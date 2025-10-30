
"""
tests/corpusutils.py

Utility functions used in pytest tests for Korp, in particular for
setting up CWB corpus data.
"""


import glob
import os
import os.path
import re
import subprocess

from collections import defaultdict
from itertools import chain, zip_longest


class CWBEncoder:

    """Encode VRT data to a CWB corpus."""

    # Default positional attribute names if none specified in input
    _default_pos_attrs = [
        "word",
        "lemma",
        "pos",
        "msd",
        "deprel",
        "dephead",
        "ref",
        "lex/",
    ]
    # Default name (prefix) for the rest of the positional attributes
    # (beginning from the 9th attribute)
    _default_pos_attr_name = "attr"

    def __init__(self, corpus_root, cwb_encode=None, cwb_makeall=None):
        """Initialize with paths for corpus root, cwb-encode, cwb-makeall."""
        cwb_encode = cwb_encode or "cwb-encode"
        cwb_makeall = cwb_makeall or "cwb-makeall"
        corpus_root = os.path.abspath(corpus_root)
        self._datarootdir = os.path.join(corpus_root, "data")
        self._registrydir = os.path.join(corpus_root, "registry")
        os.makedirs(self._datarootdir)
        os.makedirs(self._registrydir)

    def encode_corpora(self, corpus_src_dir):
        """Encode all VRT data in corpus_src_dir, base name as corpus id."""
        corpus_ids = []
        for vrt_file in glob.glob(os.path.join(corpus_src_dir, "*.vrt")):
            corpus_id = os.path.splitext(os.path.basename(vrt_file))[0]
            self.encode_corpus(corpus_id, vrt_file)
            corpus_ids.append(corpus_id)
        return corpus_ids

    def encode_corpus(self, corpus_id, vrt_file):
        """Encode vrt_file with corpus_id."""
        self.encode_vrt_file(corpus_id, vrt_file)
        self.cwb_makeall(corpus_id)
        self.copy_info_file(corpus_id, vrt_file)

    def encode_vrt_file(self, corpus_id, vrt_file):
        """Run cwb-encode for vrt_file for corpus_id."""

        def interleave(s, seq):
            """Return [s, seq[0], s, seq[1], ... , s, seq[-1]."""
            return [*chain(*zip_longest([], seq, fillvalue=s))]

        attrs = self._get_attrs(vrt_file)
        # print(attrs)
        datadir = os.path.join(self._datarootdir, corpus_id)
        os.makedirs(datadir)
        subprocess.run([
            "cwb-encode",
            "-f", vrt_file,
            "-d", datadir,
            "-R", os.path.join(self._registrydir, corpus_id),
            "-xsB",
            "-c", "utf8",
            "-p", "-",
            *interleave("-P", attrs["positional"]),
            *interleave("-S", attrs["structural"])
        ]).check_returncode()

    def _get_attrs(self, vrt_file):
        """Get the positional and strucutral attribute info from vrt_file.

        Assumes that vrt_file contains comments of the following kind
        before the first data line (token or structural attribute):
        <!-- #vrt positional-attributes: attr1 attr2 ... -->
        <!-- #vrt structural-attributes: text:0+a1+a2 sentence:0+a3+a4 ... -->
        Returns dict
        {
            "positional": ["attr1", "attr2", ...],
            "structural": ["text:0+a1+a2", "sentence:0+a3+a4", ...]
        }
        """

        def make_struct_spec(name, depth, attrnames):
            """Make structural attribute specification for cwb-encode.

            name is the name of the structural attribute, depth is the
            recursive nesting depth (1-based) and attrnames the
            annotation names.
            """
            return (f"{name}:{depth - 1}"
                    + "".join(f"+{attrname}" for attrname in attrnames))

        attrs = {
            "positional": [],
            "structural": [],
        }
        # Processing line before the first token or structure tag
        in_header = True
        # Annotation names of each structural attribute: values are
        # dicts with dummy values to substitute ordered sets
        struct_attrs = defaultdict(dict)
        # The number of each structural attribute currently open
        open_structs = defaultdict(int)
        # For each structural attribute, the maximum nesting depth
        struct_maxdepth = defaultdict(int)
        with open(vrt_file, "r") as vrtf:
            for line in vrtf:
                if line[0] == "<":
                    if in_header and line.startswith(
                            ("<!-- #vrt positional-attributes:",
                             "<!-- #vrt structural-attributes:")):
                        attrs[line.split()[2].split("-")[0]] = (
                            line.partition(":")[2].strip(" ->\n").split())
                        if attrs["structural"] and attrs["positional"]:
                            return attrs
                    elif line[1] not in "!?":
                        in_header = False
                        structname = re.search(r"\w+", line).group(0)
                        if line[1] == "/":
                            open_structs[structname] -= 1
                        else:
                            open_structs[structname] += 1
                            struct_maxdepth[structname] = max(
                                struct_maxdepth[structname],
                                open_structs[structname])
                            # Should we also allow attribute values
                            # enclosed in single quotes?
                            attrnames = re.findall(r'(\w+?)="(?:[^"]*)"', line)
                            # Use dict with dummy values as a substitute
                            # for an ordered set
                            struct_attrs[structname].update(
                                dict((attrname, None)
                                     for attrname in attrnames))
                elif line[0] != "\n":
                    in_header = False
                    if not attrs["positional"]:
                        pos_attr_count = line.count("\t") + 1
                        attrs["positional"] = (
                            self._default_pos_attrs[:pos_attr_count]
                            + [self._default_pos_attr_name + str(attrnum + 1)
                               for attrnum in range(
                                       len(self._default_pos_attrs),
                                       pos_attr_count)])
                        if attrs["structural"]:
                            return attrs
        attrs["structural"] = [make_struct_spec(structname,
                                                struct_maxdepth[structname],
                                                struct_attrs[structname])
                               for structname in struct_attrs.keys()]
        return attrs

    def cwb_makeall(self, corpus_id):
        """Run cwb-makeall for corpus corpus_id."""
        subprocess.run([
            "cwb-makeall",
            "-r", self._registrydir,
            corpus_id
        ]).check_returncode()

    def copy_info_file(self, corpus_id, vrt_file):
        """Copy corpus.info from source dir to the corpus data dir as .info."""
        info_file = os.path.splitext(vrt_file)[0] + ".info"
        if os.path.isfile(info_file):
            subprocess.run([
                "cp",
                "-p",
                info_file,
                os.path.join(self._datarootdir, corpus_id, ".info")
            ])
