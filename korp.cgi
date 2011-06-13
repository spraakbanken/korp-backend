#!/usr/bin/python

"""
korp.cgi is a CGI interface for querying the corpora that are available on the server.

Currently it acts as a wrapper for the CQP querying language of Corpus Workbench.
"""

from subprocess import Popen, PIPE
#from cStringIO import StringIO
from collections import defaultdict

import random
import time
import cgi
import re
import json
import MySQLdb
import zlib

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
MAX_KWIC_ROWS = 1000


######################################################################
# These variables should probably not need to be changed

# The version of this script
KORP_VERSION = "0.28"

# The available CGI commands; for each command there must be a function
# with the same name, taking one argument (the CGI form)
COMMANDS = "info query count relations relations_sentences annotationstats".split()

def default_command(form):
    return "query" if "cqp" in form else "info"

# Special symbols used by this script; they must NOT be in the corpus
END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

# Regular expressions for parsing CGI parameters
IS_NUMBER = re.compile(r"^\d+$")
IS_IDENT = re.compile(r"^[\w\-,]+$")

QUERY_DELIM = ","

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
    
    # Convert form fields to regular dictionary
    form_raw = cgi.FieldStorage()
    form = dict((field, form_raw.getvalue(field)) for field in form_raw.keys())
    
    command = form.get("command")
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
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
    
    result = {"corpora": {}}
    total_size = 0

    for corpus in corpora:
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
                if infokey == "Size":
                    total_size += int(infoval)

        result["corpora"][corpus] = {"attrs": attrs, "info": info}
    
    result["total_size"] = total_size
    
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
    #assert_key("context", form, r"^\d+ [\w-]+$")
    assert_key("show", form, IS_IDENT)
    assert_key("show_struct", form, IS_IDENT)
    assert_key("within", form, IS_IDENT)
    assert_key("cut", form, IS_NUMBER)

    ############################################################################
    # First we read all CGI parameters and translate them to CQP
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)

    shown = form.get("show", [])
    if isinstance(shown, basestring):
        shown = shown.split(QUERY_DELIM)
    shown = set(shown)
    shown.add("word")

    shown_structs = form.get("show_struct", [])
    if isinstance(shown_structs, basestring):
        shown_structs = shown_structs.split(QUERY_DELIM)
    shown_structs = set(shown_structs)
    
    defaultcontext = form.get("defaultcontext", "10 words")
    
    context = form.get("context", {})
    if context:
        if not ":" in context:
            raise ValueError("Malformed value for key 'context'.")
        context = dict(x.split(":") for x in context.split(","))
    
    start, end = int(form.get("start")), int(form.get("end"))

    if end - start >= MAX_KWIC_ROWS:
        raise ValueError("At most %d KWIC rows can be returned per call." % MAX_KWIC_ROWS)

    cqp = form.get("cqp").decode("utf-8")
    if "within" in form:
        cqp += " within %s" % form.get("within")
    if "cut" in form:
        cqp += " cut %s" % form.get("cut")

    total_hits = 0
    statistics = {}
    result = {}
    
    saved_statistics = {}
    saved_total_hits = 0
    saved_hits = form.get("querydata", "")
    if saved_hits:
        saved_hits = zlib.decompress(saved_hits.replace("\\n", "\n").replace("-", "+").replace("_", "/").decode("base64"))
        saved_crc32, saved_total_hits, stats_temp = saved_hits.split(";", 2)
        checksum = str(zlib.crc32(cqp.encode("utf-8")  + "".join(sorted(corpora))))
        if saved_crc32 == checksum:
            saved_total_hits = int(saved_total_hits)
            for pair in stats_temp.split(";"):
                c, h = pair.split(":")
                saved_statistics[c] = int(h)
    
    ############################################################################
    # Iterate through the corpora to find from which we will fetch our results
    
    start_local = start
    end_local = end
    
    for corpus in corpora:
        skip = (end_local < 0)
        
        if not saved_statistics or (saved_statistics and saved_statistics[corpus] > start_local and not skip):
            
            cmd = ["%s;" % corpus]
            # This prints the attributes and their relative order:
            cmd += show_attributes()
            cmd += make_query(cqp)
            # This prints the size of the query (i.e., the number of results):
            cmd += ["size Last;"]
            cmd += ["show +%s;" % " +".join(shown)]
            setcontext = context[corpus] if corpus in context else defaultcontext
            cmd += ["set Context %s;" % setcontext]
            cmd += ["set LeftKWICDelim '%s '; set RightKWICDelim ' %s';" % (LEFT_DELIM, RIGHT_DELIM)]
            if shown_structs:
                cmd += ["set PrintStructures '%s';" % ", ".join(shown_structs)]
            # This prints the result rows:
            if not skip:
                cmd += ["cat Last %s %s;" % (start_local, end_local)]

            ######################################################################
            # Then we call the CQP binary, and read the results

            lines = runCQP(cmd, form, attr_ignore=True)

            # Skip the CQP version
            lines.next()
            
            # Read the attributes and their relative order 
            attrs = read_attributes(lines)
            
            # Read the size of the query, i.e., the number of results
            nr_hits = int(lines.next())

            if len(corpora) == 1:
                shown_local = shown
                shown_structs_local = shown_structs
            else:
                # Filter out unavailable attributes
                all_attrs = attrs["p"] + attrs["s"] + attrs["a"]
                shown_local = set(attr for attr in shown if attr in all_attrs)
                shown_structs_local = set(attr for attr in shown_structs if attr in all_attrs)
                
        else:
            nr_hits = saved_statistics[corpus]
            skip = True
        
        statistics[corpus] = nr_hits
        total_hits += nr_hits
        
        # Calculate which hits from next corpus we need, if any
        start_local = start_local - nr_hits
        end_local = end_local - nr_hits
        if start_local < 0:
            start_local = 0

        if skip:
            continue

        p_attrs = [attr for attr in attrs["p"] if attr in shown_local]
        nr_splits = len(p_attrs) - 1
        s_attrs = set(attr for attr in attrs["s"] if attr in shown_local)
        ls_attrs = set(attr for attr in attrs["s"] if attr in shown_structs_local)
        a_attrs = set(attr for attr in attrs["a"] if attr in shown_local)

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
            if shown_structs_local.intersection(ls_attrs) and not aligned:
                lineattr, line = line.split(":  ", 1)
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
            result["DEBUG"] = {"cqp": cqp, "cmd": cmd}

    result["hits"] = total_hits
    result["corpus_hits"] = statistics
    checksum = str(zlib.crc32(cqp.encode("utf-8") + "".join(sorted(corpora))))
    result["querydata"] = zlib.compress(checksum + ";" + str(total_hits) + ";" + ";".join("%s:%d" % (c, h) for c, h in statistics.iteritems())).encode("base64").replace("+", "-").replace("/", "_")
    
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

    corpus = form.get("corpus")
    
    shown = form.get("show")
    if isinstance(shown, basestring):
        shown = shown.split(QUERY_DELIM)
    shown = set(shown)

    cqp = form.get("cqp").decode("utf-8")
    if "within" in form:
        cqp += " within %s" % form.get("within")
    if "cut" in form:
        cqp += " cut %s" % form.get("cut")

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


def annotationstats(form):
    """    """
    assert_key("annotation", form, r"", True)
    assert_key("group", form, r"", True)
    assert_key("value", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)

    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
    
    group = form.get("group").decode("utf-8")
    annotation = form.get("annotation").decode("utf-8")
    value = form.get("value").decode("utf-8")
    
    result = {"corpora": {}}
    total_stats = {"absolute": defaultdict(int),
                   "relative": defaultdict(float)}
    total_size = 0
    
    for corpus in corpora:

        cmd = ["%s;" % corpus]
        cmd += ["info; .EOL.;"]
        cmd += make_query('[%s contains "%s"]' % (annotation, value))
        cmd += ["group Last match %s;" % group]

        lines = runCQP(cmd, form)

        # skip CQP version
        lines.next()
        
        for line in lines:
            if line.startswith("Size:"):
                _, corpus_size = line.split(":")
                corpus_size = int(corpus_size.strip())
            elif line == END_OF_LINE:
                break
        
        total_size += corpus_size
        corpus_stats = {"absolute": defaultdict(int),
                        "relative": defaultdict(float)}
        for line in lines:
            wordform, count = line.split("\t")
            corpus_stats["absolute"][wordform.lower()] += int(count)
            corpus_stats["relative"][wordform.lower()] += int(count) / float(corpus_size) * 1000000
            total_stats["absolute"][wordform.lower()]  += int(count)
            
        result["corpora"][corpus] = corpus_stats
    
    for wf, count in total_stats["absolute"].iteritems():
        total_stats["relative"][wf] = count / float(total_size) * 1000000
    
    result["total"] = total_stats
        
    return result


def relations(form):
    assert_key("corpus", form, IS_IDENT, True)
    #assert_key("lemgram", form, r"", True)
    assert_key("min", form, IS_NUMBER, False)
    assert_key("max", form, IS_NUMBER, False)
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
        
    lemgram = form.get("lemgram")
    word = form.get("word")
    minfreq = form.get("min")
    maxresults = form.get("max") or 15
    maxresults = int(maxresults)
    minfreqsql = " AND freq >= %s" % minfreq if minfreq else ""
    
    assert lemgram or word, "lemgram or word missing."
    
    corporasql = []
    for corpus in corpora:
        corporasql.append("corpus = '%s'" % corpus)
    corporasql = " OR ".join(corporasql)
    
    result = {}

    conn = MySQLdb.connect(host = "localhost",
                           user = "",
                           passwd = "",
                           db = "")
    cursor = conn.cursor()
    
    if lemgram:
        headdep = "dep" if "..av." in lemgram else "head"
        cursor.execute("""SELECT * FROM relations WHERE (""" + corporasql + """) AND (""" + headdep + """ = %s)""" + minfreqsql, (lemgram,))
    elif word:
        cursor.execute("""SELECT * FROM relations WHERE (""" + corporasql + """) AND (head = %s OR head = %s OR dep = %s)""" + minfreqsql, (word + "_VB", word + "_NN", word + "_JJ"))
    
    rels = {}
    counter = {}
    
    for row in cursor:
        rels.setdefault((row[0], row[1], row[2], row[3]), {"freq": 0, "corpus": []})
        rels[(row[0], row[1], row[2], row[3])]["freq"] += row[4]
        rels[(row[0], row[1], row[2], row[3])]["corpus"].append(row[5])
    
    sortedrels = sorted(rels.items(), key=lambda x: (x[0][1], x[1]["freq"]), reverse=True)
    
    for rel in sortedrels:
        counter.setdefault(rel[0][1], 0)
        if counter[rel[0][1]] >= maxresults:
            continue
        else:
            counter[rel[0][1]] += 1
            r = { "head": rel[0][0],
                  "rel": rel[0][1],
                  "dep": rel[0][2],
                  "depextra": rel[0][3],
                  "freq": rel[1]["freq"],
                  "corpus": rel[1]["corpus"]
                }
            result.setdefault("relations", []).append(r)
    
    cursor.close()
    conn.close()
    
    return result


def relations_sentences(form):
    from copy import deepcopy
    
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("head", form, "", True)
    assert_key("dep", form, "", True)
    assert_key("rel", form, "", True)
    assert_key("start", form, IS_NUMBER, False)
    assert_key("end", form, IS_NUMBER, False)
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
    
    head = form.get("head")
    dep = form.get("dep")
    depextra = form.get("depextra") or ""
    rel = form.get("rel")
    start = int(form.get("start", "0"))
    end = int(form.get("end", "199"))
    
    corporasql = []
    for corpus in corpora:
        corporasql.append("corpus = '%s'" % corpus)
    corporasql = " OR ".join(corporasql)
    
    querystarttime = time.time()

    conn = MySQLdb.connect(host = "localhost",
                           user = "",
                           passwd = "",
                           db = "")
    cursor = conn.cursor()
    cursor.execute("""SELECT sentences, corpus FROM relations WHERE (""" + corporasql + """) AND head = %s AND dep = %s AND depextra = %s AND rel = %s""", (head, dep, depextra, rel))
    
    querytime = time.time() - querystarttime
   
    counter = 0
    corpora_dict = {}
    sids = {}
    used_corpora = set()
    for row in cursor:
        ids = [s.split(":") for s in row[0].split(";")]
        for s in ids:
            if counter >= start and counter <= end:
                sids.setdefault(s[0], []).append(s[1:3])
                used_corpora.add(row[1])
                corpora_dict.setdefault(row[1], {}).setdefault(s[0], []).append(s[1:3])
            if counter > end:
                break
            counter += 1

    cursor.close()

    if not sids:
        return {"hits": 0}
    
    cqpstarttime = time.time()
    result = {"hits": 0, "corpus_hits": {}}
    
    for corp, sids in corpora_dict.items():
        cqp = u'<sentence_id="%s"> []* </sentence_id> within sentence' % "|".join(set(sids.keys()))
        result_temp = query({"cqp": cqp, "corpus": corp, "start": "0", "end": str(end - start), "show_struct": "sentence_id", "defaultcontext": "1 sentence"})

        for i in range(len(result_temp["kwic"]) - 1, -1, -1):
            s = result_temp["kwic"][i]
            sid = s["structs"]["sentence_id"]
            r = sids[sid][0]
            s["match"]["start"] = min(map(int, r)) - 1
            s["match"]["end"] = max(map(int, r)) - 1
            
            for r in sids[sid][1:]:
                s2 = deepcopy(s)
                s2["match"]["start"] = min(map(int, r)) - 1
                s2["match"]["end"] = max(map(int, r)) - 1
                result_temp["kwic"].insert(i + 1, s2)
                result_temp["hits"] += 1
    
        result.setdefault("kwic", []).extend(result_temp["kwic"])
        result["hits"] += result_temp["hits"]
        result["corpus_hits"][corp] = result_temp["hits"]

    result["querytime"] = querytime
    result["cqptime"] = time.time() - cqpstarttime
    
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


def runCQP(command, form, executable=CQP_EXECUTABLE, registry=CWB_REGISTRY, attr_ignore=False):
    """Call the CQP binary with the given command, and the CGI form.
    Yield one result line at the time, disregarding empty lines.
    If there is an error, raise a CQPError exception.
    """
    encoding = form.get("encoding", CQP_ENCODING)
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
        if not (attr_ignore and "No such attribute:" in error):
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
    value = form.get(key, "")
    if value and not isinstance(value, list):
        value = [value]
    if required and not value:
        raise KeyError("Key is required: %s" % key)
    if not all(re.match(regexp, x) for x in value):
        pattern = regexp.pattern if hasattr(regexp, "pattern") else regexp
        raise ValueError("Value(s) for key %s do(es) not match /%s/: %s" % (key, pattern, value))


def print_header():
    """Prints the JSON header."""
    print "Content-Type: application/json"
    print

def print_object(obj, form):
    """Prints an object in JSON format.
    The CGI form can contain optional parameters 'callback' and 'indent'
    which change the output format.
    """
    callback = form.get("callback")
    if callback: print callback + "(",
    try:
        indent = int(form.get("indent"))
        print json.dumps(obj, sort_keys=True, indent=indent),
    except:
        print json.dumps(obj, separators=(",",":"))
    if callback: print ")",
    print

if __name__ == "__main__":
    main()
