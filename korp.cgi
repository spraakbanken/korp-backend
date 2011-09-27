#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
korp.cgi is a CGI interface for querying the corpora that are available on the server.

Currently it acts as a wrapper for the CQP querying language of Corpus Workbench.
"""

from subprocess import Popen, PIPE
from collections import defaultdict
from concurrent import futures

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

# Number of threads to use during parallel processing
PARALLEL_THREADS = 6

# The name of the MySQL database and table prefix
DBNAME = ""
DBTABLE = "relations"

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
    corpora = sorted(set(corpora))
    
    result = {"corpora": {}}
    total_size = 0
    
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

        result["corpora"][corpus] = {"attrs": attrs, "info": info}
    
    result["total_size"] = total_size
    
    if "debug" in form:
        result["DEBUG"] = {"cmd": cmd}
    return result


def query_corpus(form, corpus, cqp, shown, shown_structs, context, defaultcontext, sortcmd, start, end, no_results=False):

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
    if not no_results:
        cmd += sortcmd
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
                # CQP bug: CQP can't handle too long sentences, skipping
                continue
            # Otherwise we add a new kwic row
            kwic_row = {"corpus": corpus, "match": match}
            if linestructs:
                kwic_row["structs"] = linestructs
            kwic_row["tokens"] = tokens
            kwic.append(kwic_row)

    return kwic


def query_and_parse(form, corpus, cqp, shown, shown_structs, context, defaultcontext, sortcmd, start, end, no_results=False):
    lines, nr_hits, attrs = query_corpus(form, corpus, cqp, shown, shown_structs, context, defaultcontext, sortcmd, start, end, no_results)
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
    assert_key("sort", form, IS_IDENT)

    ############################################################################
    # First we read all CGI parameters and translate them to CQP
    
    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = sorted(set(corpora))

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

    sort = form.get("sort")
    if sort == "left":
        sortcmd = ["sort by word on match[-1] .. match[-3];"]
    elif sort == "keyword":
        sortcmd = ["sort by word;"]
    elif sort == "right":
        sortcmd = ["sort by word on matchend[1] .. matchend[3];"]
    else:
        sortcmd = []

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
        
        # If only one corpus, it is faster to not use threads
        if len(corpora_hits) == 1:
            corpus, hits = corpora_hits.items()[0]
            kwic, _ = query_and_parse(form, corpus, cqp, shown, shown_structs, context, defaultcontext, sortcmd, hits[0], hits[1])

            if result.has_key("kwic"):
                result["kwic"].extend(kwic)
            else:
                result["kwic"] = kwic
        else:
            with futures.ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
                future_query = dict((executor.submit(query_and_parse, form, corpus, cqp, shown, shown_structs, context, defaultcontext, sortcmd, corpora_hits[corpus][0], corpora_hits[corpus][1]), corpus) for corpus in corpora_hits)
                
                for future in futures.as_completed(future_query):
                    corpus = future_query[future]
                    if future.exception() is not None:
                        print '\nERROR: %r generated an exception: %s\n' % (corpus, future.exception())
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
        rest_corpora = []
        # Serial until we've got all the requested rows
        for i, corpus in enumerate(corpora):
            if end_local < 0:
                rest_corpora = corpora[i:]
                break
            kwic, nr_hits = query_and_parse(form, corpus, cqp, shown, shown_structs, context, defaultcontext, sortcmd, start_local, end_local)
            
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
        
        if rest_corpora:
            with futures.ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
                future_query = dict((executor.submit(query_corpus, form, corpus, cqp, shown, shown_structs, context, defaultcontext, sortcmd, 0, 0, True), corpus) for corpus in rest_corpora)
                
                for future in futures.as_completed(future_query):
                    corpus = future_query[future]
                    if future.exception() is not None:
                        print '\nERROR: %r generated an exception: %s\n' % (corpus, future.exception())
                    else:
                        _, nr_hits, _ = future.result()
                        statistics[corpus] = nr_hits
                        total_hits += nr_hits

    if "debug" in form:
        result["DEBUG"] = {"cqp": cqp, "cmd": cmd}

    result["hits"] = total_hits
    result["corpus_hits"] = statistics
    result["corpus_order"] = corpora
    checksum = str(zlib.crc32(cqp.encode("utf-8") + "".join(sorted(corpora))))
    result["querydata"] = zlib.compress(checksum + ";" + str(total_hits) + ";" + ";".join("%s:%d" % (c, h) for c, h in statistics.iteritems())).encode("base64").replace("+", "-").replace("/", "_")

    return result


def count_query_worker(corpus, cqp, groupby, form):

    cmd = ["%s;" % corpus]
    cmd += make_query(cqp)
    cmd += ["size Last;"]
    cmd += ["info; .EOL.;"]
    cmd += ["count Last by %s;" % groupby[0]]
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
     - groupby: add once for each corpus positional attribute

    The optional parameters are
     - within: only search for matches within the given s-attribute (e.g., within a sentence)
       (default: no within)
     - cut: set cutoff threshold to reduce the size of the result
       (default: no cutoff)
    """
    assert_key("cqp", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("groupby", form, IS_IDENT, True)
    assert_key("cut", form, IS_NUMBER)
    
    case_normalize = False

    corpora = form.get("corpus")
    if isinstance(corpora, basestring):
        corpora = corpora.split(QUERY_DELIM)
    corpora = set(corpora)
    
    groupby = form.get("groupby")
    if isinstance(groupby, basestring):
        groupby = groupby.split(QUERY_DELIM)
    groupby = list(set(groupby))
    
    start = int(form.get("start", 0))
    end = int(form.get("end", -1))
    
    cqp = form.get("cqp").decode("utf-8")
    if "within" in form:
        cqp += " within %s" % form.get("within")
    if "cut" in form:
        cqp += " cut %s" % form.get("cut")

    result = {"corpora": {}}
    total_stats = {"absolute": defaultdict(int),
                   "relative": defaultdict(float),
                   "sums": {"absolute": 0, "relative": 0.0}}
    total_size = 0

    # TODO: we could use cwb-scan-corpus for counting:
    #   cwb-scan-corpus -q SUC2 '?word=/^en$/c' 'pos' 'pos+1' | sort -n -r
    # it's efficient, but I think more limited

    # If we only want to group by one attribute, we can use CQP's internal statistics
    if len(groupby) == 1:
        with futures.ThreadPoolExecutor(max_workers=PARALLEL_THREADS) as executor:
            future_query = dict((executor.submit(count_query_worker, corpus, cqp, groupby, form), corpus) for corpus in corpora)
            
            for future in futures.as_completed(future_query):
                corpus = future_query[future]
                if future.exception() is not None:
                    print '\nERROR: %r generated an exception: %s\n' % (corpus, future.exception())
                else:
                    lines, nr_hits, corpus_size = future.result()

                    total_size += corpus_size
                    corpus_stats = {"absolute": defaultdict(int),
                                    "relative": defaultdict(float),
                                    "sums": {"absolute": 0, "relative": 0.0}}
                    
                    for i, line in enumerate(lines):
                        count, _pos, ngram = line.split(None, 2)
                        if case_normalize:
                            ngram = ngram.lower()
                        corpus_stats["absolute"][ngram] += int(count)
                        corpus_stats["relative"][ngram] += int(count) / float(corpus_size) * 1000000
                        corpus_stats["sums"]["absolute"] += int(count)
                        corpus_stats["sums"]["relative"] += int(count) / float(corpus_size) * 1000000
                        total_stats["absolute"][ngram]  += int(count)
                        total_stats["sums"]["absolute"] += int(count)
                    
                    result["corpora"][corpus] = corpus_stats
    """
    else:
        cmd = ["%s;" % corpus]
        cmd += ["info; .EOL.;"]
        cmd += make_query(cqp)
        cmd += ["set LeftKWICDelim ''; set RightKWICDelim '';"]
        cmd += ["set Context 0 words;"]
        cmd += ["show -cpos -word;"]
        cmd += ["show +%s;" % " +".join(groupby)]
        cmd += ["cat Last;"]
        cmd += ["exit;"]

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
    """

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
    
    total_stats["sums"]["relative"] = total_stats["sums"]["absolute"] / float(total_size) * 1000000
    result["total"] = total_stats
    
    if "debug" in form:
        result["DEBUG"] = {"cqp": cqp, "cmd": cmd}
        
    return result


def annotationstats(form):
    """ Deprecated. Use count() instead. """
    
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
        cmd += ["exit;"]

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

    import math

    rel_grouping = {
        "OO": "OBJ",
        "IO": "OBJ",
        "RA": "ADV",
        "TA": "ADV",
        "OA": "ADV"
    }

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
    sortby = form.get("sortby", "mi")
    maxresults = form.get("max", 15)
    maxresults = int(maxresults)
    minfreqsql = " AND freq >= %s" % minfreq if minfreq else ""
    
    assert lemgram or word, "lemgram or word missing."
    
    result = {}

    conn = MySQLdb.connect(host = "localhost",
                           user = "",
                           passwd = "",
                           db = DBNAME,
                           use_unicode = True)
    cursor = conn.cursor()
    
    # Get available tables
    cursor.execute("SHOW TABLES LIKE '" + DBTABLE + "_%';")
    tables = set(x[0] for x in cursor)
    # Filter out corpora which doesn't exist in database
    corpora = filter(lambda x: DBTABLE + "_" + x.upper() in tables, corpora)
    
    columns = "head, rel, dep, depextra, freq, freq_rel, freq_head_rel, freq_rel_dep"    
    selects = []
    if lemgram:
        lemgram_sql = conn.escape(lemgram).decode("utf-8")
        lemgram = lemgram.decode("utf-8")
        headdep = "dep" if "..av." in lemgram else "head"
        
        for corpus in corpora:
            corpus_table = DBTABLE + "_" + corpus.upper()
            selects.append(u"(SELECT " + columns + u", " + conn.string_literal(corpus.upper()) + u" as corpus FROM " + corpus_table + u" WHERE " + headdep + u" = " + lemgram_sql + minfreqsql + u")")
    elif word:
        word_vb_sql = conn.escape(word + "_VB").decode("utf-8")
        word_nn_sql = conn.escape(word + "_NN").decode("utf-8")
        word_jj_sql = conn.escape(word + "_JJ").decode("utf-8")
        
        for corpus in corpora:
            corpus_table = DBTABLE + "_" + corpus.upper()
            selects.append(u"(SELECT " + columns + u", " + conn.string_literal(corpus.upper()) + u" as corpus FROM " + corpus_table + (u" WHERE (head = %s OR head = %s OR dep = %s)" % (word_vb_sql, word_nn_sql, word_jj_sql)) + minfreqsql + ")")
    
    sql = " UNION ALL ".join(selects)
    cursor.execute(sql)
            
    rels = {}
    counter = {}
    freq_rel = {}
    freq_head_rel = {}
    freq_rel_dep = {}
    
    for row in cursor:
        val = row[2] if "..av." in lemgram else row[0]
        # TODO: Ta bort  or row[2] == "" när stöd för intransitiva verb finns i framändan
        if (lemgram and val <> lemgram) or (word and not val.startswith(word)) or row[2] == "":
            continue
        rel = rel_grouping.get(row[1], row[1])
        rels.setdefault((row[0], rel, row[2], row[3]), {"freq": 0, "corpus": set()})
        rels[(row[0], rel, row[2], row[3])]["freq"] += row[4]
        rels[(row[0], rel, row[2], row[3])]["corpus"].add(row[8])
        
        freq_rel.setdefault(rel, {})[(row[8], row[1])] = row[5]
        freq_head_rel.setdefault((row[0], rel), {})[(row[8], row[1])] = row[6]
        freq_rel_dep.setdefault((rel, row[2], row[3]), {})[(row[8], row[1])] = row[7]
    
    # Calculate MI
    for rel in rels:
        f_rel = sum(freq_rel[rel[1]].values())
        f_head_rel = sum(freq_head_rel[(rel[0], rel[1])].values())
        f_rel_dep = sum(freq_rel_dep[(rel[1], rel[2], rel[3])].values())
        rels[rel]["mi"] = rels[rel]["freq"] * math.log((f_rel * rels[rel]["freq"]) / (f_head_rel * f_rel_dep * 1.0), 2)
    
    sortedrels = sorted(rels.items(), key=lambda x: (x[0][1], x[1][sortby]), reverse=True)
    
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
                  "mi": rel[1]["mi"],
                  "corpus": list(rel[1]["corpus"])
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
    corpora = sorted(set(corpora))
    
    head = form.get("head")
    dep = form.get("dep")
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
                           user = "",
                           passwd = "",
                           db = "")
    cursor = conn.cursor()
    selects = []
    
    head_sql = conn.escape(head).decode("utf-8")
    dep_sql = conn.escape(dep).decode("utf-8")
    depextra_sql = conn.escape(depextra).decode("utf-8")
    rel_sql = conn.escape(rel).decode("utf-8")
    
    for corpus in corpora:
        selects.append(u"""(SELECT sentences, %s as corpus FROM """ % conn.string_literal(corpus.upper()) + DBTABLE + u"_" + corpus.upper() + (u""" WHERE head = %s AND dep = %s AND depextra = %s AND rel = %s""" % (head_sql, dep_sql, depextra_sql, rel_sql)) + u")")
    sql = u" UNION ALL ".join(selects)
    cursor.execute(sql)
    
    querytime = time.time() - querystarttime
   
    counter = 0
    corpora_dict = {}
    sids = {}
    total_hits = 0
    corpus_hits = {}
    for row in cursor:
        ids = [s.split(":") for s in row[0].split(";")]
        total_hits += len(ids)
        corpus_hits[row[1]] = len(ids)
        for s in ids:
            if counter >= start and counter <= end:
                sids.setdefault(s[0], []).append(s[1:3])
                corpora_dict.setdefault(row[1], {}).setdefault(s[0], []).append(s[1:3])
            if counter > end:
                break
            counter += 1

    cursor.close()

    if not sids:
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
            # append compies of the sentence as separate results
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
    # Regexp optimizer is not activated by default in CQP 3 beta
    command = "set PrettyPrint off;\nset Optimize on;\n" + command
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
        if not (attr_ignore and "No such attribute:" in error) and not "is not defined for corpus" in error:
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
    print "Access-Control-Allow-Origin: *"
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
