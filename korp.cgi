#!/usr/bin/python

"""
korp.cgi is a CGI interface for querying the corpora that are available on the server.

Currently it acts as a wrapper for the CQP querying language of Corpus Workbench.
"""

from subprocess import Popen, PIPE
from cStringIO import StringIO
from collections import defaultdict

import random
import time
import cgi
import re
import json
import sqlite3 as sqlite

######################################################################
# These variables could be changed depending on the corpus server

# The absolute path to the CQP binary
CQP_EXECUTABLE = "/usr/local/bin/cqp"

# The absolute path to the CWB registry files
CWB_REGISTRY = "/usr/contrib/etc/cwb_registry"

# The default encoding for the cqp binary
# (this can be changed by the CGI parameter 'encoding')
CQP_ENCODING = "UTF-8"

# The maximum number of search results that can be returned per query
MAX_KWIC_ROWS = 100


######################################################################
# These variables should probably not need to be changed

# The version of this script
KORP_VERSION = "0.28"

# The available CGI commands; for each command there must be a function
# with the same name, taking one argument (the CGI form)
COMMANDS = "info query count relations".split()

def default_command(form):
    return "query" if "cqp" in form else "info"

# Special symbols used by this script; they must NOT be in the corpus
END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

# Regular expressions for parsing CGI parameters
IS_NUMBER = re.compile(r"^\d+$")
IS_IDENT = re.compile(r"^[\w-]+$")


######################################################################
# And now the functions corresponding to the CGI commands

def main():
    """The main CGI handler; reads the 'command' parameter and calls
    the same-named function with the CGI form as argument.

    Global CGI parameter are
     - command: (default: 'info' or 'query' depending on the 'cqp' parameter)
     - callback: an identifier that the result should be wrapped in
     - encoding: the encoding for interacting with the corpus (default: UTF-8)
     - indent: pretty-print the result with a specific indentation (for debugging)
     - debug: if set, return some extra information (for debugging)
    """
    starttime = time.time()
    print_header()
    form = cgi.FieldStorage()
    command = form.getfirst("command")
    if not command:
        command = default_command(form)
    try:
        if command not in COMMANDS:
            raise ValueError("'%s' is not a permitted command, try these instead: '%s'" % (command, "', '".join(COMMANDS)))
        assert_key("callback", form, IS_IDENT)
        assert_key("encoding", form, IS_IDENT)
        assert_key("indent", form, IS_NUMBER)

        # Here we call the command function:
        result = globals()[command](form)
        result["time"] = time.time() - starttime
        print_object(result, form)
    except:
        import traceback, sys
        exc = sys.exc_info()
        error = {"ERROR": {"type": exc[0].__name__,
                           "value": str(exc[1]),
                           "traceback": traceback.format_exc().splitlines(),
                           },
                 "time": time.time() - starttime}
        print_object(error, form)


def info(form):
    """Return information, either about a specific corpus
    or general information about the available corpora.
    """
    if "corpus" in form:
        return corpus_info(form)
    else:
        return general_info(form)


def general_info(form):
    """Return information about the available corpora.
    """
    corpora = runCQP("show corpora;", form)
    version = corpora.next()
    return {"cqp-version": version, "corpora": list(corpora)}


def corpus_info(form):
    """Return information about a specific corpus.
    """
    assert_key("corpus", form, IS_IDENT, True)
    corpus = form.getfirst("corpus")

    cmd = ["%s;" % corpus]
    cmd += show_attributes()
    cmd += ["info;"]

    # call the CQP binary
    lines = runCQP(cmd, form)

    # skip CQP version 
    lines.next()

    # read attributes
    attrs = read_attributes(lines)

    # corpus information
    raw_info = list(lines)
    info = {}
    
    for infoline in raw_info:
        if ":" in infoline and not infoline.endswith(":"):
            infokey, infoval = (x.strip() for x in infoline.split(":", 1))
            info[infokey] = infoval

    result = {"attrs": attrs, "info": info}
    if "debug" in form:
        result["DEBUG"] = {"cmd": cmd}
    return result


def query(form):
    """Perform a CQP query and return a number of matches.

    Each match contains position information and a list of the words and attributes in the match.

    The required parameters are
     - corpus: the CWB corpus. More than one parameter can be used.
     - cqp: the CQP query string
     - start, end: which result rows that should be returned

    The optional parameters are
     - context: how many words/sentences to the left/right should be returned
       (default '10 words')
     - show: add once for each corpus parameter (positional/strutural/alignment)
       (default only show the 'word' parameter)
     - show_struct: structural annotations for matched region. Multiple parameters possible.
     - within: only search for matches within the given s-attribute (e.g., within a sentence)
       (default: no within)
     - cut: set cutoff threshold to reduce the size of the result
       (default: no cutoff)
    """
    assert_key("cqp", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("start", form, IS_NUMBER, True)
    assert_key("end", form, IS_NUMBER, True)
    assert_key("context", form, r"^\d+ [\w-]+$")
    assert_key("show", form, IS_IDENT)
    assert_key("show_struct", form, IS_IDENT)
    assert_key("within", form, IS_IDENT)
    assert_key("cut", form, IS_NUMBER)

    ############################################################################
    # First we read all CGI parameters and translate them to CQP

    corpora = set(form.getlist("corpus"))
    shown = set(form.getlist("show"))
    shown.add("word")
    shown_structs = set(form.getlist("show_struct"))
    context = form.getfirst("context", "10 words")
    start, end = int(form.getfirst("start")), int(form.getfirst("end"))

    if end - start >= MAX_KWIC_ROWS:
        raise ValueError("At most %d KWIC rows can be returned per call." % MAX_KWIC_ROWS)

    cqp = form.getfirst("cqp").decode("utf-8")
    if "within" in form:
        cqp += " within %s" % form.getfirst("within")
    if "cut" in form:
        cqp += " cut %s" % form.getfirst("cut")

    total_hits = 0
    current_position = 0
    result = {}

    ############################################################################
    # Iterate through the corpora to find from which we will fetch our results
    
    for corpus in corpora:
    
        skip = False
    
        # No extra queries need to be made if only one corpus is selected
        if len(corpora) == 1:
            shown_local = shown
            shown_structs_local = shown_structs
            start_local = start
            end_local = end
        else:
            # First query is done to determine number of hits. This is needed for
            # pagination in multiple corpora to work.
            cmd = ["%s;" % corpus]
            # Show attributes, to determine which attributes are available
            cmd += show_attributes()
            cmd += make_query(cqp)
            cmd += ["size Last;"]
            
            lines = runCQP(cmd, form)
            
            # Skip the CQP version
            lines.next()
            
            # Read attributes and filter out unavailable ones
            attrs = read_attributes(lines)
            attrs = attrs["p"] + attrs["s"] + attrs["a"]
            shown_local = set(attr for attr in shown if attr in attrs)
            shown_structs_local = set(attr for attr in shown_structs if attr in attrs)
            
            # Read number of hits
            corpus_hits = int(lines.next())
            total_hits += corpus_hits
            
            start_local = 0
            end_local = 0
            
            # Calculate which hits from this corpus is needed, if any
            if start >= current_position and start < total_hits:
                start_local = start - current_position
                end_local = min(end - current_position, corpus_hits - 1)
            elif end >= current_position and end < total_hits:
                start_local = max(start - current_position, 0)
                end_local = end - current_position
            elif start < current_position and end >= total_hits:
                start_local = 0
                end_local = corpus_hits - 1
            else:
                skip = True

            current_position += corpus_hits
        
        # If hits from this corpus is needed, query corpos again and fetch results
        if not skip:
            
            cmd = ["%s;" % corpus]
            # This prints the attributes and their relative order:
            cmd += show_attributes()
            cmd += make_query(cqp)
            # This prints the size of the query (i.e., the number of results):
            cmd += ["size Last;"]
            cmd += ["show +%s;" % " +".join(shown_local)]
            cmd += ["set Context %s;" % context]
            cmd += ["set LeftKWICDelim '%s '; set RightKWICDelim ' %s';" % (LEFT_DELIM, RIGHT_DELIM)]
            if shown_structs_local:
                cmd += ["set PrintStructures '%s';" % ", ".join(shown_structs_local)]
            # This prints the result rows:
            cmd += ["cat Last %s %s;" % (start_local, end_local)]

            ######################################################################
            # Then we call the CQP binary, and read the results

            lines = runCQP(cmd, form)

            # Skip the CQP version
            lines.next()

            # Read the attributes and their relative order 
            attrs = read_attributes(lines)
            p_attrs = [attr for attr in attrs["p"] if attr in shown_local]
            nr_splits = len(p_attrs) - 1
            s_attrs = set(attr for attr in attrs["s"] if attr in shown_local)
            ls_attrs = set(attr for attr in attrs["s"] if attr in shown_structs_local)
            a_attrs = set(attr for attr in attrs["a"] if attr in shown_local)

            # Read the size of the query, i.e., the number of results
            nr_hits = int(lines.next())

            ######################################################################
            # Now we create the concordance (kwic = keywords in context)
            # from the remaining lines

            kwic = []
            for line in lines:
                linestructs = {}
                match = {}

                header, line = line.split(":", 1)
                if header[:3] == "-->":
                    # For aligned corpora, every other line is the aligned result
                    aligned = header[3:]
                else:
                    # This is the result row for the query corpus
                    aligned = None
                    match["position"] = int(header)

                # Handle PrintStructures
                if shown_structs_local.intersection(ls_attrs):
                    lineattr, line = line.split(":", 1)
                    lineattrs = lineattr.split("<")[1:]
                    
                    for s in lineattrs:
                        s = s[:-1]
                        
                        if s in ls_attrs:
                            s_key = s
                            s_val = None
                        else:
                            s_key, s_val = s.split(" ", 1)

                        linestructs[s_key] = s_val

                words = line.split()
                tokens = []
                n = 0
                structs = defaultdict(list)
                struct = None

                for word in words:
                
                    if struct:
                        # Structural attrs can be split in the middle (<s_n 123>),
                        # so we need to finish the structure here
                        struct_id, word = word.split(">", 1)
                        structs["open"].append(struct + " " + struct_id)
                        struct = None

                    # We use special delimiters to see when we enter and leave the match region
                    if word == LEFT_DELIM:
                        match["start"] = n
                        continue
                    elif word == RIGHT_DELIM:
                        match["end"] = n
                        continue

                    # We read all structural attributes that are opening (from the left)
                    while word[0] == "<":
                        if word[1:] in s_attrs:
                            # If we stopped in the middle of a struct (<s_n 123>),
                            # wee need to continue with the next word
                            struct = word[1:]
                            break
                        elif ">" in word and word[1:word.find(">")] in s_attrs:
                            # This is for s-attrs that have no arguments (<s>)
                            struct, word = word[1:].split(">", 1)
                            structs["open"].append(struct)
                            struct = None
                        else:
                            # What we've found is not a structural attribute
                            break

                    if struct:
                        # If we stopped in the middle of a struct (<s_n 123>),
                        # wee need to continue with the next word
                        continue

                    # Now we read all s-attrs that are closing (from the right)
                    while word[-1] == ">" and "</" in word:
                        word, struct = word[:-1].rsplit("</", 1)
                        structs["close"].insert(0, struct)
                        struct = None

                    # What's left is the word with its p-attrs
                    values = word.rsplit("/", nr_splits)
                    token = dict((attr, translate_undef(val)) for (attr, val) in zip(p_attrs, values))
                    if structs:
                        token["structs"] = structs
                        structs = defaultdict(list)
                    tokens.append(token)

                    n += 1

                if aligned:
                    # If this was an aligned row, we add it to the previous kwic row
                    if words != ["(no", "alignment", "found)"]:
                        kwic[-1].setdefault("aligned", {})[aligned] = tokens
                else:
                    # Otherwise we add a new kwic row
                    kwic_row = {"corpus": corpus, "match": match}
                    if linestructs:
                        kwic_row["structs"] = linestructs
                    kwic_row["tokens"] = tokens
                    kwic.append(kwic_row)

            result["hits"] = nr_hits
            if result.has_key("kwic"):
                result["kwic"].extend(kwic)
            else:
                result["kwic"] = kwic
            
            if "debug" in form:
                result_local["DEBUG"] = {"cqp": cqp, "cmd": cmd}
            
    if len(corpora) > 1:
        result["hits"] = total_hits
    
    return result


def count(form):
    """Perform a CQP query and return a count of the given words/attrs.

    The required parameters are
     - corpus: the CWB corpus
     - cqp: the CQP query string
     - show: add once for each corpus positional attribute

    The optional parameters are
     - within: only search for matches within the given s-attribute (e.g., within a sentence)
       (default: no within)
     - cut: set cutoff threshold to reduce the size of the result
       (default: no cutoff)
    """
    assert_key("cqp", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("show", form, IS_IDENT, True)
    assert_key("cut", form, IS_NUMBER)

    corpus = form.getfirst("corpus")
    shown = form.getlist("show")
    cqp = form.getfirst("cqp").decode("utf-8")
    if "within" in form:
        cqp += " within %s" % form.getfirst("within")
    if "cut" in form:
        cqp += " cut %s" % form.getfirst("cut")

    # TODO: we could use cwb-scan-corpus for counting:
    #   cwb-scan-corpus -q SUC2 '?word=/^en$/c' 'pos' 'pos+1' | sort -n -r
    # it's efficient, but I think more limited

    if False: # if len(shown) == 1:
        # If we only want to show one attribute, we can use CQP's internal statistics
        # (but it seems to be slower anyway, so we skip that)
        # TODO: probably it's faster for large corpora, we should perhaps test that
        cmd = ["%s;" % corpus]
        cmd += make_query(cqp)
        cmd += ["size Last;"]
        cmd += ["count Last by %s;" % shown[0]]

        lines = runCQP(cmd, form)

        # skip CQP version
        lines.next()

        # size of the query result
        nr_hits = int(lines.next())

        counts = []
        for line in lines:
            count, _pos, ngram = line.split(None, 2)
            counts.append([count, ngram.split()])

    else:
        cmd = ["%s;" % corpus]
        cmd += make_query(cqp)
        cmd += ["set LeftKWICDelim ''; set RightKWICDelim '';"]
        cmd += ["set Context 0 words;"]
        cmd += ["show -cpos -word;"]
        cmd += ["show +%s;" % " +".join(shown)]
        cmd += ["cat Last;"]

        lines = runCQP(cmd, form)

        # skip CQP version
        lines.next()

        nr_hits = 0
        counts = defaultdict(int)
        for line in lines:
            nr_hits += 1
            counts[line] += 1

        counts = [[count, ngram.split()] for (ngram, count) in counts.iteritems()]
        counts.sort(reverse=True)

    result = {"hits": nr_hits, "counts": counts}
    if "debug" in form:
        result["DEBUG"] = {"cqp": cqp, "cmd": cmd}
    return result


def relations(form):
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("lemgram", form, r"", True)
    
    corpora = set(form.getlist("corpus"))
    lemgram = "%|" + form.getfirst("lemgram").decode("UTF-8") + "|%"
    
    corporasql = []
    for corpus in corpora:
        corporasql.append("corpus = '%s'" % corpus)
    corporasql = " OR ".join(corporasql)
    
    result = {}

    conn = sqlite.connect('relations.db')
    cur = conn.cursor()
    cur.execute("""SELECT * FROM relations WHERE (""" + corporasql + """) AND head LIKE ? OR dep LIKE ? ORDER BY head, rel""", (lemgram, lemgram))
    
    for row in cur:
        r = { "head": row[0],
              "rel": row[1],
              "dep": row[2],
              "freq": row[3],
              "sources": row[4].split(";"),
              "corpus": row[5]
            }
        result.setdefault("relations", []).append(r)
    
    conn.commit()
    cur.close()
    
    return result

######################################################################
# Helper functions

def make_query(cqp):
    """Create web-safe commands for a CQP query.
    """
    querylock = random.randrange(10**8, 10**9)
    return ["set QueryLock %s;" % querylock,
            "%s;" % cqp,
            "unlock %s;" % querylock]


def translate_undef(s):
    """Translate None to '__UNDEF__'."""
    return None if s == "__UNDEF__" else s


class CQPError(Exception):
    pass


def runCQP(command, form, executable=CQP_EXECUTABLE, registry=CWB_REGISTRY):
    """Call the CQP binary with the given command, and the CGI form.
    Yield one result line at the time, disregarding empty lines.
    If there is an error, raise a CQPError exception.
    """
    encoding = form.getfirst("encoding", CQP_ENCODING)
    if not isinstance(command, basestring):
        command = "\n".join(command)
    command = "set PrettyPrint off;\n" + command
    command = command.encode(encoding)
    process = Popen([executable, "-c", "-r", registry],
                    stdin=PIPE, stdout=PIPE, stderr=PIPE)
    reply, error = process.communicate(command)
    if error:
        # remove newlines from the error string:
        error = re.sub(r"\s+", r" ", error)
        # keep only the first CQP error (the rest are consequences):
        error = re.sub(r"^CQP Error: *", r"", error)
        error = re.sub(r" *(CQP Error:).*$", r"", error)
        raise CQPError(error)
    for line in reply.decode(encoding).splitlines():
        if line:
            yield line


def show_attributes():
    """Command sequence for returning the corpus attributes."""
    return ["show cd; .EOL.;"]

def read_attributes(lines):
    """Read the CQP output from the show_attributes() command."""
    attrs = {'p': [], 's': [], 'a': []}
    for line in lines:
        if line == END_OF_LINE: break
        (typ, name, _rest) = (line + " X").split(None, 2)
        attrs[typ[0]].append(name)
    return attrs


def assert_key(key, form, regexp, required=False):
    """Check that the value of the attribute 'key' in the CGI form
    matches the specification 'regexp'. If 'required' is True, then
    the key has to be in the form.
    """
    values = form.getlist(key)
    if required and not values:
        raise KeyError("Key is required: %s" % key)
    if not all(re.match(regexp, x) for x in values):
        pattern = regexp.pattern if hasattr(regexp, "pattern") else regexp
        raise ValueError("Value(s) for key %s do(es) not match /%s/: %s" % (key, pattern, values))


def print_header():
    """Prints the JSON header."""
    print "Content-Type: application/json"
    print

def print_object(obj, form):
    """Prints an object in JSON format.
    The CGI form can contain optional parameters 'callback' and 'indent'
    which change the output format.
    """
    callback = form.getfirst("callback")
    if callback: print callback + "(",
    try:
        indent = int(form.getfirst("indent"))
        print json.dumps(obj, sort_keys=True, indent=indent),
    except:
        print json.dumps(obj, separators=(",",":"))
    if callback: print ")",
    print

if __name__ == "__main__":
    main()
