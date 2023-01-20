
"""
tests/corpusutils.py

Utility functions used in pytest tests for Korp, in particular for
setting up CWB corpus data.
"""


import glob
import os
import os.path
import subprocess

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

    def __init__(self, corpus_root, cwb_encode=None, cwb_make=None):
        """Initialize with paths for corpus root, cwb-encode, cwb-make."""
        cwb_encode = cwb_encode or "cwb-encode"
        cwb_make = cwb_make or "cwb-make"
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
        self.cwb_make(corpus_id)

    def encode_vrt_file(self, corpus_id, vrt_file):
        """Run cwb-encode for vrt_file for corpus_id."""

        def interleave(s, seq):
            """Return [s, seq[0], s, seq[1], ... , s, seq[-1]."""
            return [*chain(*zip_longest([], seq, fillvalue=s))]

        attrs = self._get_attrs(vrt_file)
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
        before the first token line:
        <!-- #vrt positional-attributes: attr1 attr2 ... -->
        <!-- #vrt structural-attributes: text:0+a1+a2 sentence:0+a3+a4 ... -->
        Returns dict
        {
            "positional": ["attr1", "attr2", ...],
            "structural": ["text:0+a1+a2", "sentence:0+a3+a4", ...]
        }
        """
        attrs = {
            "positional": [],
            "structural": [],
        }
        with open(vrt_file, "r") as vrtf:
            for line in vrtf:
                if not line.startswith("<"):
                    if not attrs["positional"]:
                        pos_attr_count = line.count("\t") + 1
                        attrs["positional"] = (
                            self._default_pos_attrs[:pos_attr_count])
                    return attrs
                elif (line.startswith("<!-- #vrt positional-attributes:") or
                      line.startswith("<!-- #vrt structural-attributes:")):
                    attrs[line.split()[2].split("-")[0]] = (
                        line.partition(":")[2].strip(" ->\n").split())
        return attrs

    def cwb_make(self, corpus_id):
        """Run cwb-make for corpus corpus_id."""
        subprocess.run([
            "cwb-make",
            "-r", self._registrydir,
            corpus_id
        ]).check_returncode()
