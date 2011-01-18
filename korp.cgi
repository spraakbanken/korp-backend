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
KORP_VERSION = "0.2"

# The available CGI commands; for each command there must be a function
# with the same name, taking one argument (the CGI form)
COMMANDS = "info query count".split()

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
    # TODO: convert into a structured object
    info = list(lines)

    result = {"attrs": attrs, "info": info}
    if "debug" in form:
        result["DEBUG"] = {"cmd": cmd}
    return result



def query(form):
    """Perform a CQP query and return a number of matches.

    Each match contains position information and a list of the words and attributes in the match.

    The required parameters are
     - corpus: the CWB corpus
     - cqp: the CQP query string
     - start, end: which result rows that should be returned

    The optional parameters are
     - context: how many words/sentences to the left/right should be returned
       (default '10 words')
     - show: add once for each corpus parameter (positional/strutural/alignment)
       (default only show the 'word' parameter)
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
    assert_key("within", form, IS_IDENT)
    assert_key("cut", form, IS_NUMBER)

    ######################################################################
    # First we read all CGI parameters and translate them to CQP

    corpus = form.getfirst("corpus")
    shown = set(form.getlist("show"))
    shown.add("word")
    context = form.getfirst("context", "10 words")
    start, end = int(form.getfirst("start")), int(form.getfirst("end"))

    if end - start >= MAX_KWIC_ROWS:
        raise ValueError("At most %d KWIC rows can be returned per call." % MAX_KWIC_ROWS)

    cqp = form.getfirst("cqp").decode("utf-8")
    if "within" in form:
        cqp += " within %s" % form.getfirst("within")
    if "cut" in form:
        cqp += " cut %s" % form.getfirst("cut")

    cmd = ["%s;" % corpus]
    # This prints the attributes and their relative order:
    cmd += show_attributes()
    cmd += make_query(cqp)
    # This prints the size of the query (i.e., the number of results):
    cmd += ["size Last;"]
    cmd += ["show +%s;" % " +".join(shown)]
    cmd += ["set Context %s;" % context]
    cmd += ["set LeftKWICDelim '%s '; set RightKWICDelim ' %s';" % (LEFT_DELIM, RIGHT_DELIM)]
    # This prints the result rows:
    cmd += ["cat Last %s %s;" % (start, end)]

    ######################################################################
    # Then we call the CQP binary, and read the results

    lines = runCQP(cmd, form)

    # Skip the CQP version
    lines.next()

    # Read the attributes and their relative order 
    attrs = read_attributes(lines)
    p_attrs = [attr for attr in attrs["p"] if attr in shown]
    nr_splits = len(p_attrs) - 1
    s_attrs = set(attr for attr in attrs["s"] if attr in shown)
    a_attrs = set(attr for attr in attrs["a"] if attr in shown)

    # Read the size of the query, i.e., the number of results
    nr_hits = int(lines.next())

    ######################################################################
    # Now we create the concordance (kwic = keywords in context)
    # from the remaining lines

    kwic = []
    for line in lines:
        match = {}

        header, line = line.split(":", 1)
        if header[:3] == "-->":
            # For aligned corpora, every other line is the aligned result
            aligned = header[3:]
        else:
            # This is the result row for the query corpus
            aligned = None
            match["position"] = int(header)

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
                # This is for s-attrs that have no arguments (<s>)
                struct, word = word[1:].split(">", 1)
                structs["open"].append(struct)
                struct = None

            if struct:
                # If we stopped in the middle of a struct (<s_n 123>),
                # wee need to continue with the next word
                continue

            # Now we read all s-attrs that are closing (from the right)
            while word[-1] == ">":
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
            kwic.append({"match": match, "tokens": tokens})

    result = {"hits": nr_hits, "kwic": kwic}
    if "debug" in form:
        result["DEBUG"] = {"cqp": cqp, "cmd": cmd}
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



######################################################################
# Assorted notes and TODOs about the CQP binary, and the CQP Manual
#
# 3.2: Save queries to disk
# > set dd "path-to-data-dir"   (or cqp -l .)
# > SUC2;
# > X = "...";
# > save X;
#
# Load queries from disk
# > set dd "path-to-data-dir"
# > X;
# X> ...
#
# Dump results to disk (slower)
# > dump X > "file"
# > undump X with target keyword < "file"
#
# Corpus information
# > info SUC2
#
# Don't show corpus position
# > show -cpos
#
# 4.1: labels
# 4.2: s-attrs, ... expand (left|right)? to ...
#
# 4.4: To display structural attrs for each match
# > set PrintStructures "sentence_n, text_id"
#
# 4.4: Find matches in a particular novel
# > B = [pos = "NP"] [pos = "NP"] :: match.novel_title = "David Copperfield";
# (where <novel title="A Tale of Two Cities"> ...B... </novel>)
#
# 2.9: sorting, counting
# > count by pos %c
#
# 3.4: The command group is a bit limited (can't handle sequences, only single words)
# good for co-occurences?
#
# 3.5: set operations, subset
#
# 5.2: word lists (good for pos-classes, e.g.)
# 5.3: subqueries, remember to use:    > ... expand to sentence
######################################################################


if __name__ == "__main__":
    main()
