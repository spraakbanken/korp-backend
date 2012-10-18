#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
korp.cgi is a CGI interface for querying the corpora that are available on the server.

Currently it acts as a wrapper for the CQP querying language of Corpus Workbench.

http://spraakbanken.gu.se/korp/
"""

from subprocess import Popen, PIPE
from collections import defaultdict
from concurrent import futures

import sys
import os
import random
import time
import cgi
import re
import json
import MySQLdb
import zlib
import urllib, urllib2, base64, md5

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

# Number of threads to use during parallel processing
PARALLEL_THREADS = 6

# The name of the MySQL database and table prefix
DBNAME = ""
DBTABLE = "relations"
# Username and password for database access
DBUSER = ""
DBPASSWORD = ""

# URL to authorization server
AUTH_SERVER = ""
AUTH_SECRET = ""

######################################################################
# These variables should probably not need to be changed

# The version of this script
KORP_VERSION = "2.0"

# The available CGI commands; for each command there must be a function
# with the same name, taking one argument (the CGI form)
COMMANDS = "info query count relations relations_sentences lemgram_count timespan authenticate".split()

def default_command(form):
    return "query" if "cqp" in form else "info"

# Special symbols used by this script; they must NOT be in the corpus
END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

# Regular expressions for parsing CGI parameters
IS_NUMBER = re.compile(r"^\d+$")
IS_IDENT = re.compile(r"^[\w\-,|]+$")

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
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) # Open unbuffered stdout
    print_header()
    
    # Convert form fields to regular dictionary
    form_raw = cgi.FieldStorage()
    form = dict((field, form_raw.getvalue(field)) for field in form_raw.keys())
    
    incremental = form.get("incremental", "").lower() == "true"
    
    if incremental:
        print "{"
            
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
        import traceback
        exc = sys.exc_info()
        error = {"ERROR": {"type": exc[0].__name__,
                           "value": str(exc[1])
                           },
                 "time": time.time() - starttime}
        if "debug" in form:
            error["ERROR"]["traceback"] = traceback.format_exc().splitlines()
        print_object(error, form)
    
    if incremental:
        print "}"

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
    """Return information about a specific corpus or corpora.
    """
    assert_key("corpus", form, IS_IDENT, True)
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = sorted(set(corpora))
    
    result = {"corpora": {}}
    total_size = 0
    total_sentences = 0
    
    cmd = []

    for corpus in corpora:
        cmd += ["%s;" % corpus]
        cmd += show_attributes()
        cmd += ["info; .EOL.;"]

    cmd += ["exit;"]

    # call the CQP binary
    lines = runCQP(cmd, form)

    # skip CQP version 
    lines.next()
    
    for corpus in corpora:
        # read attributes
        attrs = read_attributes(lines)

        # corpus information
        info = {}
        
        for line in lines:
            if line == END_OF_LINE: break
            if ":" in line and not line.endswith(":"):
                infokey, infoval = (x.strip() for x in line.split(":", 1))
                info[infokey] = infoval
                if infokey == "Size":
                    total_size += int(infoval)
                elif infokey == "Sentences":
                    total_sentences += int(infoval)

        result["corpora"][corpus] = {"attrs": attrs, "info": info}
    
    result["total_size"] = total_size
    result["total_sentences"] = total_sentences
    
    if "debug" in form:
        result["DEBUG"] = {"cmd": cmd}
    return result


def query_optimize(cqp, cqpextra):
    """ Optimizes simple queries with multiple words by converting them to an MU query.
        Optimization only works for queries with at least two tokens, or one token preceded
        by one or more wildcards. The query also must use "within".
        """
    
    q, rest = parse_cqp(cqp)
    expand = cqpextra.get("within")
    
    leading_wildcards = False
    # Remove leading and trailing wildcards since they will only slow us down
    while q and q[0].startswith("[]"):
        leading_wildcards = True
        del q[0]
    while q and q[-1].startswith("[]"):
        del q[-1]
    
    # Determine if this query may not benefit from optimization
    if len(q) == 0 or (len(q) == 1 and not leading_wildcards) or rest or not expand:
        return make_query(make_cqp(cqp, cqpextra))
    
    cmd = ["MU"]
    wildcards = {}

    for i in range(len(q) - 1):
        if q[i].startswith(u"[]"):
            n1 = n2 = None
            if q[i] == u"[]":
                n1 = n2 = 1
            elif re.search(r"{\s*(\d+)\s*,\s*(\d*)\s*}$", q[i]):
                n = re.search(r"{\s*(\d+)\s*,\s*(\d*)\s*}$", q[i]).groups()
                n1 = int(n[0])
                n2 = int(n[1]) if n[1] else 9999
            elif re.search(r"{\s*(\d*)\s*}$", q[i]):
                n1 = n2 = int(re.search(r"{\s*(\d*)\s*}$", q[i]).groups()[0])
            if not n1 == None:
                wildcards[i] = (n1, n2)
            continue
        elif re.search(r"{.*?}$", q[i]):
            # Repetition for anything other than wildcards can't be optimized
            return make_query(make_cqp(cqp, cqpextra))
        cmd[0] += " (meet %s" % (q[i])
    
    cmd[0] += " %s" % q[-1]
    
    wildcard_range = [1, 1]
    for i in range(len(q) - 2, -1, -1):
        if i in wildcards:
            wildcard_range[0] += wildcards[i][0]
            wildcard_range[1] += wildcards[i][1]
            continue
        elif i + 1 in wildcards:
            if wildcard_range[1] >= 9999:
                cmd[0] += " %s)" % expand
            else:
                cmd[0] += " %d %d)" % (wildcard_range[0], wildcard_range[1])
            wildcard_range = [1, 1]
        else:
            cmd[0] += " 1 1)"

    cmd[0] += " expand to %s;" % expand
    cmd += ["Last;"]
    cmd += make_query(make_cqp(cqp, cqpextra))

    return cmd

def query_corpus(form, corpus, cqp, cqpextra, shown, shown_structs, start, end, no_results=False):

    # Optimization
    optimize = True # bool(form.get("optimize"))
    
    # Context
    defaultcontext = form.get("defaultcontext", "10 words")
    context = form.get("context", {})
    if context:
        if not ":" in context:
            raise ValueError("Malformed value for key 'context'.")
        context = dict(x.split(":") for x in context.split(","))
    
    # Within
    defaultwithin = form.get("defaultwithin", "")
    within = form.get("within", defaultwithin)
    if within:
        if ":" in within:
            within = dict(x.split(":") for x in within.split(","))
            within = within.get(corpus, defaultwithin)
        cqpextra["within"] = within
    
    # Handle aligned corpora
    if "|" in corpus:
        linked = corpus.split("|")
        cqp = cqp.replace("<LINKED_CORPUS>", linked[1])
        corpus = linked[0]
        shown.add(linked[1].lower())
    
    # Sorting
    sort = form.get("sort")
    if sort == "left":
        sortcmd = ["sort by word on match[-1] .. match[-3];"]
    elif sort == "keyword":
        sortcmd = ["sort by word;"]
    elif sort == "right":
        sortcmd = ["sort by word on matchend[1] .. matchend[3];"]
    else:
        sortcmd = []

    # Build the CQP query
    cmd = ["%s;" % corpus]
    # This prints the attributes and their relative order:
    cmd += show_attributes()
    if optimize:
        cmd += query_optimize(cqp, cqpextra)
    else:
        cmd += make_query(make_cqp(cqp, cqpextra))
    # This prints the size of the query (i.e., the number of results):
    cmd += ["size Last;"]
    if not no_results:
        cmd += ["show +%s;" % " +".join(shown)]
        setcontext = context[corpus] if corpus in context else defaultcontext
        cmd += ["set Context %s;" % setcontext]
        cmd += ["set LeftKWICDelim '%s '; set RightKWICDelim ' %s';" % (LEFT_DELIM, RIGHT_DELIM)]
        if shown_structs:
            cmd += ["set PrintStructures '%s';" % ", ".join(shown_structs)]
        cmd += sortcmd
        # This prints the result rows:
        cmd += ["cat Last %s %s;" % (start, end)]
    cmd += ["exit;"]
    ######################################################################
    # Then we call the CQP binary, and read the results

    lines = runCQP(cmd, form, attr_ignore=True)

    # Skip the CQP version
    lines.next()
    
    # Read the attributes and their relative order 
    attrs = read_attributes(lines)
    
    # Read the size of the query, i.e., the number of results
    nr_hits = int(lines.next())
    
    return (lines, nr_hits, attrs)


def query_parse_lines(corpus, lines, attrs, shown, shown_structs):
    ######################################################################
    # Now we create the concordance (kwic = keywords in context)
    # from the remaining lines

    # Filter out unavailable attributes
    p_attrs = [attr for attr in attrs["p"] if attr in shown]
    nr_splits = len(p_attrs) - 1
    s_attrs = set(attr for attr in attrs["s"] if attr in shown)
    ls_attrs = set(attr for attr in attrs["s"] if attr in shown_structs)
    a_attrs = set(attr for attr in attrs["a"] if attr in shown)

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
        if ls_attrs and not aligned:
            lineattr, line = line.rsplit(":  ", 1)
            lineattrs = lineattr[2:-1].split("><")
            
            # Handle "><" in attribute values
            if not len(lineattrs) == len(ls_attrs):
                new_lineattrs = []
                for la in lineattrs:
                    if not la.split(" ", 1)[0] in ls_attrs:
                        new_lineattrs[-1] += "><" + la
                    else:
                        new_lineattrs.append(la)
                lineattrs = new_lineattrs
            
            for s in lineattrs:
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
                    # we need to continue with the next word
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
                # we need to continue with the next word
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
            if not "start" in match:
                # TODO: CQP bug - CQP can't handle too long sentences, skipping
                continue
            # Otherwise we add a new kwic row
            kwic_row = {"corpus": corpus, "match": match}
            if linestructs:
                kwic_row["structs"] = linestructs
            kwic_row["tokens"] = tokens
            kwic.append(kwic_row)

    return kwic


def query_and_parse(form, corpus, cqp, cqpextra, shown, shown_structs, start, end, no_results=False):
    lines, nr_hits, attrs = query_corpus(form, corpus, cqp, cqpextra, shown, shown_structs, start, end, no_results)
    kwic = query_parse_lines(corpus, lines, attrs, shown, shown_structs)
    return kwic, nr_hits
    

def which_hits(corpora, stats, start, end):
    corpus_hits = {}
    for corpus in corpora:
        hits = stats[corpus]
        if hits > start:
            corpus_hits[corpus] = (start, min(hits - 1, end))
        
        start -= hits
        end -= hits
        if start < 0: start = 0
        if end < 0: break
    
    return corpus_hits


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
     - sort: sort the results by keyword ('keyword'), left context ('left) or right context ('right')
       (default: no sorting)
     - incremental: returns the result incrementally instead of all at once
    """
    assert_key("cqp", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("start", form, IS_NUMBER, True)
    assert_key("end", form, IS_NUMBER, True)
    #assert_key("context", form, r"^\d+ [\w-]+$")
    assert_key("show", form, IS_IDENT)
    assert_key("show_struct", form, IS_IDENT)
    #assert_key("within", form, IS_IDENT)
    assert_key("cut", form, IS_NUMBER)
    assert_key("sort", form, r"(left|keyword|right)")
    assert_key("incremental", form, r"(true|false)")

    ############################################################################
    # First we read all CGI parameters and translate them to CQP
    
    incremental = form.get("incremental", "").lower() == "true"
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = sorted(set(corpora))
    
    check_authentication(corpora)

    shown = form.get("show", [])
    if isinstance(shown, basestring):
        shown = shown.split(QUERY_DELIM)
    shown = set(shown)
    shown.add("word")

    shown_structs = form.get("show_struct", [])
    if isinstance(shown_structs, basestring):
        shown_structs = shown_structs.split(QUERY_DELIM)
    shown_structs = set(shown_structs)
    
    start, end = int(form.get("start")), int(form.get("end"))

    if end - start >= MAX_KWIC_ROWS:
        raise ValueError("At most %d KWIC rows can be returned per call." % MAX_KWIC_ROWS)

    cqp = form.get("cqp").decode("utf-8")
    cqpextra = {}

    if "within" in form:
        cqpextra["within"] = form.get("within")
    if "cut" in form:
        cqpextra["cut"] = form.get("cut")

    total_hits = 0
    statistics = {}
    result = {}
    
    saved_statistics = {}
    saved_total_hits = 0
    saved_hits = form.get("querydata", "")
    if saved_hits:
        saved_hits = zlib.decompress(saved_hits.replace("\\n", "\n").replace("-", "+").replace("_", "/").decode("base64"))
        saved_crc32, saved_total_hits, stats_temp = saved_hits.split(";", 2)
        checksum = str(zlib.crc32(make_cqp(cqp, cqpextra).encode("utf-8") + "".join(sorted(corpora))))
        if saved_crc32 == checksum:
            saved_total_hits = int(saved_total_hits)
            for pair in stats_temp.split(";"):
                c, h = pair.split(":")
                saved_statistics[c] = int(h)
        
    start_local = start
    end_local = end
    
    ############################################################################
    # If saved_statistics is available, calculate which corpora need to be queried
    # and then query them in parallel.
    # If saved_statistics is NOT available, query the corpora in serial until we
    # have the needed rows, and then query the remaining corpora in parallel to get
    # number of hits.
    
    if saved_statistics:
        statistics = saved_statistics
        total_hits = sum(saved_statistics.values())
        corpora_hits = which_hits(corpora, saved_statistics, start, end)
        corpora_kwics = {}
        
        progress_count = 0
        
        # If only one corpus, it is faster to not use threads
        if len(corpora_hits) == 1:
            corpus, hits = corpora_hits.items()[0]
            result["kwic"], _ = query_and_parse(form, corpus, cqp, cqpextra, shown, shown_structs, hits[0], hits[1])
        else:
            if incremental:
                print '"progress_corpora": ["%s"],' % '", "'.join(corpora_hits.keys())
            with futures.ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
                future_query = dict((executor.submit(query_and_parse, form, corpus, cqp, cqpextra, shown, shown_structs, corpora_hits[corpus][0], corpora_hits[corpus][1]), corpus) for corpus in corpora_hits)
                
                for future in futures.as_completed(future_query):
                    corpus = future_query[future]
                    if incremental:
                        print '"progress_%d": {"corpus": "%s", "hits": %d},' % (progress_count, corpus, corpora_hits[corpus][1] - corpora_hits[corpus][0] + 1)
                        progress_count += 1
                    if future.exception() is not None:
                        raise CQPError(future.exception())
                    else:
                        kwic, _ = future.result()
                        corpora_kwics[corpus] = kwic
                
                for corpus in corpora:
                    if corpus in corpora_hits.keys():
                        if result.has_key("kwic"):
                            result["kwic"].extend(corpora_kwics[corpus])
                        else:
                            result["kwic"] = corpora_kwics[corpus]
    else:
        if incremental:
            print '"progress_corpora": ["%s"],' % '", "'.join(corpora)
        
        progress_count = 0
        rest_corpora = []
        
        # Serial until we've got all the requested rows
        for i, corpus in enumerate(corpora):
                       
            if end_local < 0:
                rest_corpora = corpora[i:]
                break
            kwic, nr_hits = query_and_parse(form, corpus, cqp, cqpextra, shown, shown_structs, start_local, end_local)
            
            statistics[corpus] = nr_hits
            total_hits += nr_hits
            
            # Calculate which hits from next corpus we need, if any
            start_local = start_local - nr_hits
            end_local = end_local - nr_hits
            if start_local < 0: start_local = 0

            if result.has_key("kwic"):
                result["kwic"].extend(kwic)
            else:
                result["kwic"] = kwic
            
            if incremental:
                print '"progress_%d": {"corpus": "%s", "hits": %d},' % (progress_count, corpus, nr_hits)
                progress_count += 1
        
        if incremental:
            print_object(result, form)
            result = {}
        
        if rest_corpora:
            if incremental:
                print ",",
            with futures.ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
                future_query = dict((executor.submit(query_corpus, form, corpus, cqp, cqpextra, shown, shown_structs, 0, 0, True), corpus) for corpus in rest_corpora)
                
                for future in futures.as_completed(future_query):
                    corpus = future_query[future]
                    if future.exception() is not None:
                        raise CQPError(future.exception())
                    else:
                        _, nr_hits, _ = future.result()
                        statistics[corpus] = nr_hits
                        total_hits += nr_hits
                        if incremental:
                            print '"progress_%d": {"corpus": "%s", "hits": %d},' % (progress_count, corpus, nr_hits)
                            progress_count += 1
        elif incremental:
            print ",",

    if "debug" in form:
        result["DEBUG"] = {"cqp": cqp}

    result["hits"] = total_hits
    result["corpus_hits"] = statistics
    result["corpus_order"] = corpora
    checksum = str(zlib.crc32(make_cqp(cqp, cqpextra).encode("utf-8") + "".join(sorted(corpora))))
    result["querydata"] = zlib.compress(checksum + ";" + str(total_hits) + ";" + ";".join("%s:%d" % (c, h) for c, h in statistics.iteritems())).encode("base64").replace("+", "-").replace("/", "_")

    return result


def count_query_worker(corpus, cqp, groupby, ignore_case, form):

    optimize = True
    cqpextra = {}
    
    if "cut" in form:
        cqpextra["cut"] = form.get("cut")

    # Within
    defaultwithin = form.get("defaultwithin", "")
    within = form.get("within", defaultwithin)
    if within:
        if ":" in within:
            within = dict(x.split(":") for x in within.split(","))
            within = within.get(corpus, defaultwithin)
        cqpextra["within"] = within

    cmd = ["%s;" % corpus]
    if optimize:
        cmd += query_optimize(cqp, cqpextra)
    else:
        cmd += make_query(make_cqp(cqp, cqpextra))
    cmd += ["size Last;"]
    cmd += ["info; .EOL.;"]
    cmd += ["""tabulate Last %s > "| sort | uniq -c | sort -nr";""" % ", ".join("match .. matchend %s%s" % (g, " %c" if g in ignore_case else "") for g in groupby)]
    cmd += ["exit;"]
    
    lines = runCQP(cmd, form)

    # skip CQP version
    lines.next()

    # size of the query result
    nr_hits = int(lines.next())

    # Get corpus size
    for line in lines:
        if line.startswith("Size:"):
            _, corpus_size = line.split(":")
            corpus_size = int(corpus_size.strip())
        elif line == END_OF_LINE:
            break

    return lines, nr_hits, corpus_size


def count(form):
    """Perform a CQP query and return a count of the given words/attrs.

    The required parameters are
     - corpus: the CWB corpus
     - cqp: the CQP query string
     - groupby: add once for each corpus positional or structural attribute

    The optional parameters are
     - within: only search for matches within the given s-attribute (e.g., within a sentence)
       (default: no within)
     - cut: set cutoff threshold to reduce the size of the result
       (default: no cutoff)
     - ignore_case: changes all values of the selected attribute to lower case
     - incremental: incrementally report the progress while executing
       (default: false)
    """
    assert_key("cqp", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("groupby", form, IS_IDENT, True)
    assert_key("cut", form, IS_NUMBER)
    assert_key("ignore_case", form, IS_IDENT)
    assert_key("incremental", form, r"(true|false)")
    
    incremental = form.get("incremental", "").lower() == "true"

    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
    
    check_authentication(corpora)
    
    groupby = form.get("groupby")
    if isinstance(groupby, basestring):
        groupby = groupby.split(QUERY_DELIM)
    
    ignore_case = form.get("ignore_case", [])
    if isinstance(ignore_case, basestring):
        ignore_case = ignore_case.split(QUERY_DELIM)
    ignore_case = set(ignore_case)
    
    start = int(form.get("start", 0))
    end = int(form.get("end", -1))
    
    cqp = form.get("cqp").decode("utf-8")

    result = {"corpora": {}}
    total_stats = {"absolute": defaultdict(int),
                   "relative": defaultdict(float),
                   "sums": {"absolute": 0, "relative": 0.0}}
    total_size = 0
    total_hits = 0
    
    # TODO: we could use cwb-scan-corpus for counting:
    #   cwb-scan-corpus -q SUC2 '?word=/^en$/c' 'pos' 'pos+1' | sort -n -r
    # it's efficient, but I think more limited

    progress_count = 0
    if incremental:
        print '"progress_corpora": ["%s"],' % '", "'.join(corpora)

    with futures.ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
        future_query = dict((executor.submit(count_query_worker, corpus, cqp, groupby, ignore_case, form), corpus) for corpus in corpora)
        
        for future in futures.as_completed(future_query):
            corpus = future_query[future]
            if future.exception() is not None:
                raise CQPError(future.exception())
            else:
                lines, nr_hits, corpus_size = future.result()

                total_size += corpus_size
                corpus_stats = {"absolute": defaultdict(int),
                                "relative": defaultdict(float),
                                "sums": {"absolute": 0, "relative": 0.0}}
                
                for i, line in enumerate(lines):
                    count, ngram = line.lstrip().split(" ", 1)
                    
                    if len(groupby) > 1:
                        groups = ngram.split("\t")
                        ngram = "/".join(groups)
                    
                    corpus_stats["absolute"][ngram] += int(count)
                    corpus_stats["relative"][ngram] += int(count) / float(corpus_size) * 1000000
                    corpus_stats["sums"]["absolute"] += int(count)
                    corpus_stats["sums"]["relative"] += int(count) / float(corpus_size) * 1000000
                    total_stats["absolute"][ngram]  += int(count)
                    total_stats["sums"]["absolute"] += int(count)
                
                result["corpora"][corpus] = corpus_stats
                
                if incremental:
                    print '"progress_%d": "%s",' % (progress_count, corpus)
                    progress_count += 1
                    total_hits += nr_hits
                    if total_hits > 1000000:
                        print '"warning":"Over 1000000 hits!",'

    result["count"] = len(total_stats["absolute"])

    if end > -1 and (start > 0 or len(total_stats["absolute"]) > (end - start) + 1):
        total_absolute = sorted(total_stats["absolute"].iteritems(), key=lambda x: x[1], reverse=True)[start:end+1]
        new_corpora = {}
        for ngram, count in total_absolute:
            total_stats["relative"][ngram] = count / float(total_size) * 1000000
                    
            for corpus in corpora:
                new_corpora.setdefault(corpus, {"absolute": {}, "relative": {}, "sums": result["corpora"][corpus]["sums"]})
                if ngram in result["corpora"][corpus]["absolute"]:
                    new_corpora[corpus]["absolute"][ngram] = result["corpora"][corpus]["absolute"][ngram]
                if ngram in result["corpora"][corpus]["relative"]:
                    new_corpora[corpus]["relative"][ngram] = result["corpora"][corpus]["relative"][ngram]
        
        result["corpora"] = new_corpora
        total_stats["absolute"] = dict(total_absolute)
    else:
        for ngram, count in total_stats["absolute"].iteritems():
            total_stats["relative"][ngram] = count / float(total_size) * 1000000
    
    total_stats["sums"]["relative"] = total_stats["sums"]["absolute"] / float(total_size) * 1000000 if total_size > 0 else 0.0
    result["total"] = total_stats
    
    if "debug" in form:
        result["DEBUG"] = {"cqp": cqp, "cmd": cmd}
        
    return result


def lemgram_count(form):
    """Returns lemgram statistics per corpus.

    The required parameters are
     - lemgram: list of lemgrams

    The optional parameters are
     - corpus: the CWB corpus/corpora
       (default: all corpora)
     - count: what to count (lemgram/prefix/suffix)
       (default: lemgram)
    """

    assert_key("lemgram", form, r"", True)
    assert_key("corpus", form, IS_IDENT)
    assert_key("count", form, r"(lemgram|prefix|suffix)")
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora) if corpora else set()
    
    check_authentication(corpora)
    
    lemgram = form.get("lemgram")
    if isinstance(lemgram, basestring):
        lemgram = lemgram.split(QUERY_DELIM)
    lemgram = set(lemgram)
    
    count = form.get("count", "lemgram")
    if isinstance(count, basestring):
        count = count.split(QUERY_DELIM)
    count = set(count)
       
    counts = {"lemgram": "freq",
              "prefix": "freq_prefix",
              "suffix": "freq_suffix"}
    
    sums = " + ".join("SUM(%s)" % counts[c] for c in count)
    
    conn = MySQLdb.connect(host = "localhost",
                           user = DBUSER,
                           passwd = DBPASSWORD,
                           db = DBNAME,
                           use_unicode = True,
                           charset = "utf8")
    cursor = conn.cursor()
    
    lemgram_sql = " lemgram IN (%s)" % "%s" % ", ".join(conn.escape(l).decode("utf-8") for l in lemgram)
    corpora_sql = " AND corpus IN (%s)" % ", ".join("%s" % conn.escape(c) for c in corpora) if corpora else ""
    
    sql = "SELECT lemgram, " + sums + " FROM lemgram_index WHERE" + lemgram_sql + corpora_sql + " GROUP BY lemgram COLLATE utf8_bin;"
    
    result = {}
    cursor.execute(sql)

    for row in cursor:
        # We need this check here, since a search for "hår" also returns "här" and "har".
        if row[0].encode("utf-8") in lemgram and int(row[1]) > 0:
            result[row[0]] = int(row[1])
    
    return result

def timespan(form):
    """Calculates timespan information for corpora.

    The required parameters are
     - corpus: the CWB corpus/corpora

    The optional parameters are
     - granularity: granularity of result (y = year, m = month, d = day)
       (default: year)
     - spans: give results as spans instead of points
       (default: points)
     - combined: combine results
       (default: results per corpus)
    """
    
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("granularity", form, r"[ymdYMD]")
    assert_key("spans", form, r"(true|false)")
    assert_key("combined", form, r"(true|false)")
    
    import datetime
    from dateutil.relativedelta import relativedelta

    def strftime(dt, fmt):
        """Python datetime.strftime < 1900 workaround, taken from https://gist.github.com/2000837"""

        if dt.year < 1900:
            # create a copy of this datetime, just in case, then set the year to
            # something acceptable, then replace that year in the resulting string
            tmp_dt = datetime.datetime(datetime.MAXYEAR, dt.month, dt.day,
                                      dt.hour, dt.minute,
                                      dt.second, dt.microsecond,
                                      dt.tzinfo)
            
            if re.search('(?<!%)((?:%%)*)(%y)', fmt):
                util.log.warning("Using %y time format with year prior to 1900 could produce unusual results!")
            
            tmp_fmt = fmt
            tmp_fmt = re.sub('(?<!%)((?:%%)*)(%y)', '\\1\x11\x11', tmp_fmt, re.U)
            tmp_fmt = re.sub('(?<!%)((?:%%)*)(%Y)', '\\1\x12\x12\x12\x12', tmp_fmt, re.U)
            tmp_fmt = tmp_fmt.replace(str(datetime.MAXYEAR), '\x13\x13\x13\x13')
            tmp_fmt = tmp_fmt.replace(str(datetime.MAXYEAR)[-2:], '\x14\x14')
            
            result = tmp_dt.strftime(tmp_fmt)
            
            if '%c' in fmt:
                # local datetime format - uses full year but hard for us to guess where.
                result = result.replace(str(datetime.MAXYEAR), str(dt.year))
            
            result = result.replace('\x11\x11', str(dt.year)[-2:])
            result = result.replace('\x12\x12\x12\x12', str(dt.year))
            result = result.replace('\x13\x13\x13\x13', str(datetime.MAXYEAR))
            result = result.replace('\x14\x14', str(datetime.MAXYEAR)[-2:])
                
            return result
            
        else:
            return dt.strftime(fmt)

    def plusminusone(date, value, df, negative=False):
        d = datetime.datetime.strptime(str(date), df)
        if negative:
            d = d - value
        else:
            d = d + value
        return int(strftime(d, df))

    def shorten(date, g):
        gs = {"y": 4, "m": 6, "d": 8, "h": 10, "n": 12, "s": 14}
        return int(str(date)[:gs[g]])
   
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
    
    #check_authentication(corpora)

    granularity = form.get("granularity", "y").lower()
    points = False if form.get("spans", "").lower() == "true" else True
    combined = True if form.get("combined", "").lower() == "true" else False
        
    if granularity == "y":
        df = "%Y"
        add = relativedelta(years=1)
    elif granularity == "m":
        df = "%Y%m"
        add = relativedelta(months=1)
    elif granularity == "d":
        df = "%Y%m%d"
        add = relativedelta(days=1)
    elif granularity == "h":
        df = "%Y%m%d%H"
        add = relativedelta(hours=1)
    elif granularity == "n":
        df = "%Y%m%d%H%M"
        add = relativedelta(minutes=1)
    elif granularity == "s":
        df = "%Y%m%d%H%M%S"
        add = relativedelta(seconds=1)
    
    conn = MySQLdb.connect(host = "localhost",
                           user = DBUSER,
                           passwd = DBPASSWORD,
                           db = DBNAME,
                           use_unicode = True,
                           charset = "utf8")
    cursor = conn.cursor()
    
    corpora_sql = "(%s)" % ", ".join("%s" % conn.escape(c) for c in corpora)
    sql = "SELECT corpus, datefrom, dateto, tokens FROM timespans WHERE corpus IN " + corpora_sql + ";"
    
    result = defaultdict(int)
    cursor.execute(sql)

    rows = defaultdict(list)
    nodes = defaultdict(set)

    for row in cursor:
        corpus = row[0]
        datefrom = shorten(row[1], granularity) if not row[1] == "" else ""
        dateto = shorten(row[2], granularity) if not row[2] == "" else ""
        tokens = row[3]

        r = {"datefrom": datefrom, "dateto": dateto, "corpus": corpus, "tokens": tokens}
        if combined: corpus = "combined"
        rows[corpus].append(r)
        nodes[corpus].add(("f", datefrom))
        nodes[corpus].add(("t", dateto))
    
    corpusnodes = dict((k, sorted(v, key=lambda x: (x[1], x[0]))) for k, v in nodes.iteritems())
    result = {}
    
    for corpus, nodes in corpusnodes.iteritems():
        result[corpus] = defaultdict(int)
    
        for i in range(0, len(nodes) - 1):
            start = nodes[i]
            end = nodes[i + 1]
            if start[0] == "t":
                start = plusminusone(start[1], add, df) if not start == "" else ""
                if start == end[1] and end[0] == "f":
                    continue
            else:
                start = start[1]
            if end[1] == "":
                end = ""
            else:
                end = end[1] if end[0] == "t" else plusminusone(end[1], add, df, True)
            
            if points and not start == "":
                result[corpus]["%d" % start] = 0
                
            for row in rows[corpus]:
                if row["datefrom"] <= start and row["dateto"] >= end:
                    if points:
                        result[corpus][str(start)] += row["tokens"]
                    else:
                        result[corpus]["%d - %d" % (start, end) if start else ""] += row["tokens"]
            if points and not end == "":
                result[corpus]["%d" % plusminusone(end, add, df, False)] = 0
    
    return result

def relations(form):
    """Calculates word picture data.

    The required parameters are
     - corpus: the CWB corpus/corpora
     - word: a word or lemgram to lookup

    The optional parameters are
     - min: cut off results with a frequency lower than this
       (default: no cut-off)
     - max: maximum number of results
       (default: 15)
     - type: type of search (word/lemgram)
       (default: word)
     - incremental: incrementally report the progress while executing
       (default: false)
    """

    import math

    assert_key("corpus", form, IS_IDENT, True)
    assert_key("word", form, "", True)
    assert_key("type", form, r"(word|lemgram)", False)
    assert_key("min", form, IS_NUMBER, False)
    assert_key("max", form, IS_NUMBER, False)
    assert_key("incremental", form, r"(true|false)")
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
    
    check_authentication(corpora)
    
    incremental = form.get("incremental", "").lower() == "true"
    
    word = form.get("word")
    search_type = form.get("type", "")
    minfreq = form.get("min")
    sortby = form.get("sortby", "mi")
    maxresults = form.get("max", 15)
    maxresults = int(maxresults)
    minfreqsql = " AND freq >= %s" % minfreq if minfreq else ""
    
    result = {}

    conn = MySQLdb.connect(host = "localhost",
                           user = DBUSER,
                           passwd = DBPASSWORD,
                           db = DBNAME,
                           use_unicode = True,
                           charset = "utf8")
    cursor = conn.cursor()
    cursor.execute("SET @@session.long_query_time = 1000;")
    
    # Get available tables
    cursor.execute("SHOW TABLES LIKE '" + DBTABLE + "_%';")
    tables = set(x[0] for x in cursor)
    # Filter out corpora which doesn't exist in database
    corpora = filter(lambda x: DBTABLE + "_" + x.upper() in tables, corpora)
    if not corpora: return {}
    
    columns = ("head", "rel", "dep", "depextra", "freq")
    selects = []
    
    if search_type == "lemgram":
        lemgram_sql = conn.escape(word).decode("utf-8")
        
        for corpus in corpora:
            corpus_table = DBTABLE + "_" + corpus.upper()
            columns1 = ", ".join(["`" + corpus_table + "`." + x for x in columns])
            columns2 = "`" + corpus_table + "_rel`.freq as rel_freq"
            columns3 = "`" + corpus_table + "_head_rel`.freq as head_rel_freq"
            columns4 = "`" + corpus_table + "_dep_rel`.freq as dep_rel_freq"
            
            selects.append((corpus.upper(), u"(SELECT " + columns1 + ", " + columns2 + ", " + columns3 + ", " + columns4 + u", " + conn.string_literal(corpus.upper()) + u" as corpus " + \
                           u"FROM `" + corpus_table + "`, `" + corpus_table + "_rel`, `" + corpus_table + "_head_rel`, `" + corpus_table + "_dep_rel` " + \
                           u"WHERE (`" + corpus_table + "`.head = " + lemgram_sql + u" COLLATE utf8_bin OR `" + corpus_table + "`.dep = " + lemgram_sql + " COLLATE utf8_bin" + u") " + \
                           minfreqsql + " AND `" + corpus_table + "`.wf = 0 " + \
                           u"AND `" + corpus_table + u"`.rel = `" + corpus_table + "_rel`.rel " + \
                           u"AND `" + corpus_table + u"`.head = `" + corpus_table + "_head_rel`.head " + \
                           u"AND `" + corpus_table + u"`.rel = `" + corpus_table + "_head_rel`.rel " + \
                           u"AND `" + corpus_table + u"`.dep = `" + corpus_table + "_dep_rel`.dep " + \
                           u"AND `" + corpus_table + u"`.depextra = `" + corpus_table + "_dep_rel`.depextra " + \
                           u"AND `" + corpus_table + u"`.rel = `" + corpus_table + "_dep_rel`.rel)"))
    else:
        suffixes = ("_VB", "_NN", "_JJ", "_PP")
        words = []
        for suffix in suffixes:
            words.append(conn.escape(word + suffix).decode("utf-8"))
        
        words_list = "(" + ", ".join(words) + " COLLATE utf8_bin)"
        word = word.decode("utf-8")
        
        for corpus in corpora:
            corpus_table = DBTABLE + "_" + corpus.upper()
            columns1 = ", ".join(["`" + corpus_table + "`." + x for x in columns])
            columns2 = "`" + corpus_table + "_rel`.freq as rel_freq"
            columns3 = "`" + corpus_table + "_head_rel`.freq as head_rel_freq"
            columns4 = "`" + corpus_table + "_dep_rel`.freq as dep_rel_freq"
    
            selects.append((corpus.upper(), u"(SELECT " + columns1 + ", " + columns2 + ", " + columns3 + ", " + columns4 + u", " + conn.string_literal(corpus.upper()) + u" AS corpus " + \
                           u"FROM `" + corpus_table + "`, `" + corpus_table + "_rel`, `" + corpus_table + "_head_rel`, `" + corpus_table + "_dep_rel` "
                           u"WHERE ((`" + corpus_table + "`.head IN " + words_list + u" AND NOT `" + corpus_table + "`.wf = 2) OR (`" + corpus_table + "`.dep IN " + words_list + " AND NOT `" + corpus_table + "`.wf = 1)) " + \
                           minfreqsql + \
                           u"AND `" + corpus_table + u"`.rel = `" + corpus_table + "_rel`.rel " + \
                           u"AND `" + corpus_table + u"`.head = `" + corpus_table + "_head_rel`.head " + \
                           u"AND `" + corpus_table + u"`.rel = `" + corpus_table + "_head_rel`.rel " + \
                           u"AND `" + corpus_table + u"`.dep = `" + corpus_table + "_dep_rel`.dep " + \
                           u"AND `" + corpus_table + u"`.depextra = `" + corpus_table + "_dep_rel`.depextra " + \
                           u"AND `" + corpus_table + u"`.rel = `" + corpus_table + "_dep_rel`.rel)"))
    
    cursor_result = []
    if incremental:
        print '"progress_corpora": ["%s"],' % '", "'.join(corpora)
        progress_count = 0
        for sql in selects:
            cursor.execute(sql[1])
            cursor_result.extend(list(cursor))
            print '"progress_%d": {"corpus": "%s"},' % (progress_count, sql[0])
            progress_count += 1
    else:    
        sql = u" UNION ALL ".join(x[1] for x in selects)
        cursor.execute(sql)
        cursor_result = cursor
    
    rels = {}
    counter = {}
    freq_rel = {}
    freq_head_rel = {}
    freq_rel_dep = {}
    
    # 0     1    2    3         4     5         6              7             8
    # head, rel, dep, depextra, freq, rel_freq, head_rel_freq, dep_rel_freq, corpus
    
    for row in cursor_result:
        #if (lemgram and (row[0] <> lemgram and row[2] <> lemgram)) or (word and not (row[0].startswith(word) or row[2].startswith(word))):
        #    continue
        rels.setdefault((row[0], row[1], row[2], row[3]), {"freq": 0, "corpus": set()})
        rels[(row[0], row[1], row[2], row[3])]["freq"] += row[4]
        rels[(row[0], row[1], row[2], row[3])]["corpus"].add(row[8])
        
        freq_rel.setdefault(row[1], {})[(row[8], row[1])] = row[5]
        freq_head_rel.setdefault((row[0], row[1]), {})[(row[8], row[1])] = row[6]
        freq_rel_dep.setdefault((row[1], row[2], row[3]), {})[(row[8], row[1])] = row[7]
    
    # Calculate MI
    for rel in rels:
        f_rel = sum(freq_rel[rel[1]].values())
        f_head_rel = sum(freq_head_rel[(rel[0], rel[1])].values())
        f_rel_dep = sum(freq_rel_dep[(rel[1], rel[2], rel[3])].values())
        rels[rel]["mi"] = rels[rel]["freq"] * math.log((f_rel * rels[rel]["freq"]) / (f_head_rel * f_rel_dep * 1.0), 2)
    
    sortedrels = sorted(rels.items(), key=lambda x: (x[0][1], x[1][sortby]), reverse=True)
    
    for rel in sortedrels:
        counter.setdefault((rel[0][1], "h"), 0)
        counter.setdefault((rel[0][1], "d"), 0)
        if search_type == "lemgram" and rel[0][0] == word:
            counter[(rel[0][1], "h")] += 1
            if counter[(rel[0][1], "h")] > maxresults:
                continue
        else:
            counter[(rel[0][1], "d")] += 1
            if counter[(rel[0][1], "d")] > maxresults:
                continue

        r = { "head": rel[0][0],
              "rel": rel[0][1],
              "dep": rel[0][2],
              "depextra": rel[0][3],
              "freq": rel[1]["freq"],
              "mi": rel[1]["mi"],
              "corpus": list(rel[1]["corpus"])
            }
        result.setdefault("relations", []).append(r)
    
    cursor.close()
    conn.close()
    
    return result


def relations_sentences(form):
    """Executes a CQP query to find sentences with a given relation from a word picture.

    The required parameters are
     - corpus: the CWB corpus/corpora
     - head: head of relation
     - rel: relation

    The optional parameters are
     - dep: dependent of relation
     - depextra: dependent prefix
     - start, end: which result rows that should be returned
     - show
     - show_struct
    """

    from copy import deepcopy
    
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("head", form, "", True)
    assert_key("rel", form, "", True)
    assert_key("dep", form, "", False)
    assert_key("depextra", form, "", False)
    assert_key("start", form, IS_NUMBER, False)
    assert_key("end", form, IS_NUMBER, False)
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = sorted(set(corpora))
    
    check_authentication(corpora)
    
    head = form.get("head")
    dep = form.get("dep", "")
    depextra = form.get("depextra", "")
    rel = form.get("rel")
    start = int(form.get("start", "0"))
    end = int(form.get("end", "99"))
    shown = form.get("show", "word")
    shown_structs = form.get("show_struct", [])
    if isinstance(shown_structs, basestring):
        shown_structs = shown_structs.split(QUERY_DELIM)
    shown_structs = set(shown_structs)
    
    querystarttime = time.time()

    conn = MySQLdb.connect(host = "localhost",
                           user = DBUSER,
                           passwd = DBPASSWORD,
                           db = DBNAME)
    cursor = conn.cursor()
    cursor.execute("SET @@session.long_query_time = 1000;")
    selects = []
    counts = []
    
    head_sql = conn.escape(head).decode("utf-8")
    dep_sql = conn.escape(dep).decode("utf-8")
    depextra_sql = conn.escape(depextra).decode("utf-8")
    rel_sql = conn.escape(rel).decode("utf-8")
    
    for corpus in corpora:
        corpus_table = DBTABLE + "_" + corpus.upper()
        corpus_table_sentences = DBTABLE + "_" + corpus.upper() + "_sentences"
        
        where = u" WHERE `" + corpus_table_sentences + u"`.id = `" + corpus_table + u"`.id AND " + \
                "`" + corpus_table + (u'`.head=%s AND ' % head_sql) + \
                "`" + corpus_table + (u'`.rel=%s AND ' % rel_sql) + \
                "`" + corpus_table + (u'`.dep=%s AND ' % dep_sql) + \
                "`" + corpus_table + (u'`.depextra=%s ' % depextra_sql)
        
        selects.append(u"(SELECT `" + corpus_table_sentences + u"`.sentence, `" + corpus_table_sentences + u"`.start, `" + corpus_table_sentences + u"`.end, " + \
                       conn.string_literal(corpus.upper()) + u" AS corpus FROM `" + corpus_table_sentences + u"`, `" + corpus_table + "`" + \
                       where + \
                       u")"
                       )
        counts.append(u"(SELECT " + conn.string_literal(corpus.upper()) + u" AS corpus, COUNT(*) FROM `" + corpus_table + u"`, `" + corpus_table_sentences + "`" + where + u")")
    
    sql_count = u" UNION ALL ".join(counts)
    cursor.execute(sql_count)
    
    corpus_hits = {}
    for row in cursor:
        corpus_hits[row[0]] = int(row[1])
    
    sql = u" UNION ALL ".join(selects) + (u" LIMIT %d, %d" % (start, end-1))
    cursor.execute(sql)
    
    querytime = time.time() - querystarttime
    corpora_dict = {}
    for row in cursor:
        # 0 sentence, 1 start, 2 end, 3 corpus
        corpora_dict.setdefault(row[3], {}).setdefault(row[0], []).append((row[1], row[2]))

    cursor.close()
    
    total_hits = sum(corpus_hits.values())

    if not corpora_dict:
        return {"hits": 0}
    
    cqpstarttime = time.time()
    result = {}
    
    for corp, sids in sorted(corpora_dict.items(), key=lambda x: x[0]):
        cqp = u'<sentence_id="%s"> []* </sentence_id> within sentence' % "|".join(set(sids.keys()))
        q = {"cqp": cqp,
             "corpus": corp,
             "start": "0",
             "end": str(end - start),
             "show_struct": ["sentence_id"] + list(shown_structs),
             "defaultcontext": "1 sentence"}
        if shown:
            q["show"] = shown
        result_temp = query(q)

        # Loop backwards since we might be adding new items
        for i in range(len(result_temp["kwic"]) - 1, -1, -1):
            s = result_temp["kwic"][i]
            sid = s["structs"]["sentence_id"]
            r = sids[sid][0]
            s["match"]["start"] = min(map(int, r)) - 1
            s["match"]["end"] = max(map(int, r))
            
            # If the same relation appears more than once in the same sentence,
            # append copies of the sentence as separate results
            for r in sids[sid][1:]:
                s2 = deepcopy(s)
                s2["match"]["start"] = min(map(int, r)) - 1
                s2["match"]["end"] = max(map(int, r))
                result_temp["kwic"].insert(i + 1, s2)
    
        result.setdefault("kwic", []).extend(result_temp["kwic"])

    result["hits"] = total_hits
    result["corpus_hits"] = corpus_hits
    result["corpus_order"] = corpora
    result["querytime"] = querytime
    result["cqptime"] = time.time() - cqpstarttime
    
    return result


######################################################################
# Helper functions

def parse_cqp(cqp):
    """ Tries to parse a CQP query, returning identified tokens and a
    boolean indicating partial failure if True."""
    
    sections = []
    last_start = 0
    in_bracket = 0
    in_quote = False
    in_curly = False
    quote_type = ""
    
    for i in range(len(cqp)):
        c = cqp[i]
        
        if c in '"\'':
            if in_quote and quote_type == c and not cqp[i - 1] == "\\":
                in_quote = False
                if not in_bracket:
                    sections.append([last_start, i])
            elif not in_quote:
                in_quote = True
                quote_type = c
                if not in_bracket:
                    last_start = i
        elif c == "[":
            if not in_bracket and not in_quote:
                last_start = i
                in_bracket = True
        elif c == "]":
            if in_bracket and not in_quote:
                sections.append([last_start, i])
                in_bracket = False
        elif c == "{" and not in_bracket and not in_quote:
            in_curly = True
        elif c == "}" and not in_bracket and not in_quote and in_curly:
            in_curly = False
            sections[-1][1] = i

    last_section = (0, 0)
    sections.append([len(cqp), len(cqp)])
    tokens = []
    rest = False

    for section in sections:
        if last_section[1] < section[0]:
            if cqp[last_section[1]+1:section[0]].strip():
                rest = True
        last_section = section
        if cqp[section[0]:section[1]+1]:
            tokens.append(cqp[section[0]:section[1]+1])
    
    return (tokens, rest)


def make_cqp(cqp, cqpextra):
    """ Combine CQP query and extra options. """
    for i in cqpextra.items():
        cqp += " %s %s" % i
    return cqp


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


class KorpAuthenticationError(Exception):
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
        # Ignore certain errors: 1) "show +attr" for unknown attr, 2) querying unknown structural attribute, 3) calculating statistics for empty results
        if not (attr_ignore and "No such attribute:" in error) and not "is not defined for corpus" in error and not "cl->range && cl->size > 0" in error and not "neither a positional/structural attribute" in error:
            raise CQPError(error)
    for line in reply.splitlines():
        # TODO: Current version of CQP can't handle extremely long sentences.
        # When fixed, remove len() check and move the decode back to reply.decode(encoding, errors="ignore").splitlines()
        if line and len(line) < 32768:
            yield line.decode(encoding, errors="ignore")


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
    print "Access-Control-Allow-Origin: *"
    print "Access-Control-Allow-Methods: GET, POST"
    print "Access-Control-Allow-Headers: Authorization"
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
        out = json.dumps(obj, sort_keys=True, indent=indent)
        out = out[1:-1] if form.get("incremental", "").lower() == "true" else out
        print out,
    except:
        out = json.dumps(obj, separators=(",",":"))
        out = out[1:-1] if form.get("incremental", "").lower() == "true" else out
        print out,
    if callback: print ")",


def authenticate(_=None):
    """Authenticates a user against an authentication server.
    """
    
    auth_header = cgi.os.environ.get('HTTP_AUTH_HEADER')

    if auth_header and auth_header.startswith("Basic "):
        user, pw = base64.b64decode(auth_header[6:]).split(":")

        postdata = {
            "username": user,
            "password": pw,
            "checksum": md5.new(user + pw + AUTH_SECRET).hexdigest()
        }

        try:
            contents = urllib2.urlopen(AUTH_SERVER, urllib.urlencode(postdata)).read()
            auth_response = json.loads(contents)
        except urllib2.HTTPError:
            raise KorpAuthenticationError("Could not contact authentication server.")
        except ValueError:
            raise KorpAuthenticationError("Invalid response from authentication server.")
        except:
            raise KorpAuthenticationError("Unexpected error during authentication.")
        
        if auth_response["authenticated"]:
            return {"corpora": ["GSLC" if x == "MINISUC" else x for x in auth_response["permitted_resources"]["corpora"]]}

    return {}


def check_authentication(corpora):
    """Takes a list of corpora, and if any of them are protected, runs authentication.
    Raises an error if authentication fails."""
    
    with open("protected.txt") as infile:
        protected = [x.strip() for x in infile.readlines()]
    c = filter(lambda x: x.upper() in protected, corpora)
    if c:
        auth = authenticate()
        if not auth:
            raise KorpAuthenticationError("You do not have access to the following corpora: %s" % ", ".join(c))

if __name__ == "__main__":
    main()

