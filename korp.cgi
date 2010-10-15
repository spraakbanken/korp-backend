#!/usr/bin/python
# -*- coding: utf-8 -*-

from subprocess import Popen, PIPE
from cStringIO import StringIO
from collections import defaultdict

import random
import time
import cgi
import re
import json

ENCODING = "UTF-8"

CQP_EXECUTABLE = "/usr/local/bin/cqp"
CWB_REGISTRY = "/usr/contrib/etc/cwb_registry"

COMMANDS = "info query count".split()
MAX_KWIC_ROWS = 100

END_OF_LINE = "-::-EOL-::-"
LEFT_DELIM = "---:::"
RIGHT_DELIM = ":::---"

IS_NUMBER = re.compile(r"^\d+$")
IS_IDENT = re.compile(r"^[\w-]+$")


def main():
    starttime = time.time()
    print_header()
    form = cgi.FieldStorage()
    command = form.getfirst("command")
    if not command:
        command = "query" if "cqp" in form else "info"
    try:
        if command not in COMMANDS:
            raise ValueError("'%s' is not a permitted command, try these instead: '%s'" % (command, "', '".join(COMMANDS)))
        assert_key("callback", form, IS_IDENT)
        assert_key("encoding", form, IS_IDENT)
        assert_key("indent", form, IS_NUMBER)

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
    if "corpus" in form:
        return corpus_info(form)
    else:
        return general_info(form)


def general_info(form):
    corpora = runCQP("show corpora;", form)
    version = corpora.next()
    return {"cqp-version": version, "corpora": list(corpora)}


def corpus_info(form):
    assert_key("corpus", form, IS_IDENT, True)
    corpus = form.getfirst("corpus")
    lines = runCQP("%s; show cd; .EOL.; info;" % corpus, form)
    # skip CQP version 
    lines.next()
    # read attributes
    attrs = read_attributes(lines)
    # corpus info
    info = list(lines)
    return {"attrs": attrs, "info": info}


def read_attributes(lines):
    attrs = {'p': [], 's': [], 'a': []}
    for line in lines:
        if line == END_OF_LINE: break
        (typ, name, _rest) = (line + " X").split(None, 2)
        attrs[typ[0]].append(name)
    return attrs


def query(form):
    assert_key("cqp", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("start", form, IS_NUMBER, True)
    assert_key("end", form, IS_NUMBER, True)
    assert_key("context", form, r"^\d+ [\w-]+$")
    assert_key("show", form, IS_IDENT)
    assert_key("cut", form, IS_NUMBER)

    corpus = form.getfirst("corpus")
    shown = set(form.getlist("show"))
    shown.add("word")
    context = form.getfirst("context", "10 words")
    cqp = form.getfirst("cqp").decode("utf-8")
    if "cut" in form:
        cqp += " cut %s" % form.getfirst("cut")
    start, end = int(form.getfirst("start")), int(form.getfirst("end"))

    if end - start >= MAX_KWIC_ROWS:
        raise ValueError("At most %d KWIC rows can be returned per call." % MAX_KWIC_ROWS)

    cmd = ["%s;" % corpus]
    cmd += ["show cd;", ".EOL.;"]
    cmd += make_query(cqp)
    cmd += ["size Last;"]
    cmd += ["show +%s;" % " +".join(shown)]
    cmd += ["set Context %s;" % context]
    cmd += ["set LeftKWICDelim '%s '; set RightKWICDelim ' %s';" % (LEFT_DELIM, RIGHT_DELIM)]
    cmd += ["cat Last %s %s;" % (start, end)]

    lines = runCQP(cmd, form)

    # skip CQP version
    lines.next()

    # read the attributes and their relative order 
    attrs = read_attributes(lines)
    p_attrs = [attr for attr in attrs["p"] if attr in shown]
    nr_splits = len(p_attrs) - 1
    s_attrs = set(attr for attr in attrs["s"] if attr in shown)
    a_attrs = set(attr for attr in attrs["a"] if attr in shown)

    # size of the query result
    nr_hits = int(lines.next())

    # the concordance
    kwic = []
    for line in lines:
        match = {}

        header, line = line.split(":", 1)
        if header[:3] == "-->":
            aligned = header[3:]
        else:
            aligned = None
            match["position"] = int(header)

        words = line.split()
        tokens = []
        n = 0
        structs = defaultdict(list)
        struct = None
        for word in words:
            if struct:
                struct_id, word = word.split(">", 1)
                structs["open"].append(struct + " " + struct_id)
                struct = None

            if word == LEFT_DELIM:
                match["start"] = n
                continue
            elif word == RIGHT_DELIM:
                match["end"] = n
                continue

            while word[0] == "<":
                if word[1:] in s_attrs:
                    struct = word[1:]
                    break
                struct, word = word[1:].split(">", 1)
                structs["open"].append(struct)
                struct = None

            if struct:
                continue

            while word[-1] == ">":
                word, struct = word[:-1].rsplit("</", 1)
                structs["close"].insert(0, struct)
                struct = None

            values = word.rsplit("/", nr_splits)
            token = dict((attr, translate_undef(val)) for (attr, val) in zip(p_attrs, values))
            if structs:
                token["structs"] = structs
                structs = defaultdict(list)
            tokens.append(token)

            n += 1

        if aligned:
            if words != ["(no", "alignment", "found)"]:
                kwic[-1].setdefault("aligned", {})[aligned] = tokens
        else:
            kwic.append({"match": match, "tokens": tokens})

    return {"hits": nr_hits,
            "kwic": kwic,
            "cqp": cqp,
            }


def count(form):
    assert_key("cqp", form, r"", True)
    assert_key("corpus", form, IS_IDENT, True)
    assert_key("show", form, IS_IDENT, True)
    assert_key("cut", form, IS_NUMBER)

    corpus = form.getfirst("corpus")
    shown = form.getlist("show")
    cqp = form.getfirst("cqp").decode("utf-8")
    if "cut" in form:
        cqp += " cut %s" % form.getfirst("cut")

    # TODO: man kan använda cwb-scan-corpus för att räkna:
    # cwb-scan-corpus -q SUC2 '?word=/^en$/c' 'pos' 'pos+1' | sort -n -r

    if False: # len(shown) == 1:
        # use CQP's internal statistics (but it's slower)
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

    return {"hits": nr_hits,
            "counts": counts,
            }


def make_query(cqp):
    querylock = random.randrange(10**8, 10**9)
    return ["set QueryLock %s;" % querylock,
            "%s;" % cqp,
            "unlock %s;" % querylock]


def translate_undef(s):
    return None if s == "__UNDEF__" else s


class CQPError(Exception):
    pass


def runCQP(command, form, executable=CQP_EXECUTABLE, registry=CWB_REGISTRY):
    encoding = form.getfirst("encoding", ENCODING)
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


def assert_key(key, form, regexp, required=False):
    values = form.getlist(key)
    if required and not values:
        raise KeyError("Key is required: %s" % key)
    if not all(re.match(regexp, x) for x in values):
        pattern = regexp.pattern if hasattr(regexp, "pattern") else regexp
        raise ValueError("Value(s) for key %s do(es) not match /%s/: %s" % (key, pattern, values))


def print_header():
    print "Content-Type: application/json"
    print

def print_object(obj, form):
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



######################################################################
# Diverse anteckningar om CQP
#
# spara frågor till disk (avns 3.2)
# > set dd "path-to-data-dir"   (eller cqp -l .)
# > SUC2;
# > X = "...";
# > save X;
#
# ladda frågor
# > set dd "path-to-data-dir"
# > X;
# X> ...
#
# dumpa resultat (långsammare)
# > dump X > "fil"
# > undump X with target keyword < "fil"
#
# korpus-info:
# > info SUC2
#
# visa INTE korpus-position:
# > show -cpos
#
# > set PrintStructures "sentence_n, novel_title"
#
# 4.1: labels
# 4.2: s-attrs, ... expand (left|right)? to ...
#
# 4.4: xml, t.ex. metadata
# > B = [pos = "NP"] [pos = "NP"] :: match.novel_title = "David Copperfield";
#
# <novel title="A Tale of Two Cities"> ...B... </novel>
#
# 2.9: sorting, counting
# > count by pos %c
# 3.4: kommandot group är lite begränsat (klarar inte sekvenser, bara enstaka ord)
# bra för cooccurences?
#
# 3.5: set operations, subset
#
# 5.2: word lists (bra för ordklasser, t.ex.)
# 5.3: subqueries, kom ihåg att använda ... expand to sentence
######################################################################
