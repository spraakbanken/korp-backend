
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

import xml.etree.ElementTree as et

from collections import defaultdict
from itertools import chain, zip_longest
from xml.sax.saxutils import escape

import yaml


def is_feature_set_value(val):
    """Return True if val is a CWB feature-set value."""
    return val and val[0] == val[-1] == "|"


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
        "lex",
    ]
    # Default name (prefix) for the rest of the positional attributes
    # (beginning from the 9th attribute)
    _default_pos_attr_name = "attr"

    def __init__(self, corpus_root, cwb_encode=None, cwb_makeall=None):
        """Initialize with paths for corpus root, cwb-encode, cwb-makeall."""
        cwb_encode = cwb_encode or "cwb-encode"
        cwb_makeall = cwb_makeall or "cwb-makeall"
        corpus_root = os.path.abspath(corpus_root)
        self._corpus_root = corpus_root
        self._datarootdir = os.path.join(corpus_root, "data")
        self._registrydir = os.path.join(corpus_root, "registry")
        self._tmpdir = os.path.join(corpus_root, "tmp")
        os.makedirs(self._datarootdir)
        os.makedirs(self._registrydir)
        os.makedirs(self._tmpdir)

    def encode_corpora(self, corpus_src_dir):
        """Encode VRT and XML data in corpus_src_dir, base name as corpus id."""
        corpus_ids = []
        cachedir, cached_corpora = self._init_cwb_cache(corpus_src_dir)
        vrt_files = glob.glob(os.path.join(corpus_src_dir, "*.vrt"))
        # Convert XML files to VRT and encode the VRT files
        for xml_file in glob.glob(os.path.join(corpus_src_dir, "*.xml")):
            corpus_id = os.path.splitext(os.path.basename(xml_file))[0]
            if self._cached_corpus_is_outdated(cachedir, corpus_id, xml_file):
                vrt_file = os.path.join(self._tmpdir, f"{corpus_id}.vrt")
                self.xml_file_to_vrt(xml_file, vrt_file)
                vrt_files.append(vrt_file)
        for vrt_file in vrt_files:
            corpus_id = os.path.splitext(os.path.basename(vrt_file))[0]
            if self._cached_corpus_is_outdated(cachedir, corpus_id, vrt_file):
                self.encode_corpus(corpus_id, vrt_file, corpus_src_dir)
                if cachedir is not None:
                    self._copy_corpus(self._corpus_root, cachedir, corpus_id)
            corpus_ids.append(corpus_id)
        # print(corpus_ids)
        return corpus_ids

    def _init_cwb_cache(self, corpus_src_dir):
        """Initialize CWB data caching, copy corpora from cache.

        Create a CWB data cache directory as
        corpus_src_dir/../cwb-cache if it does not yet exits. If the
        cache directory exists, copy corpora from the cache to the
        corpus data directory.

        Return the cache directory and a set of cached corpora ids;
        None and empty set if the cache directory cannot be created.
        """
        cachedir = os.path.join(os.path.dirname(corpus_src_dir), "cwb-cache")
        try:
            if not os.path.exists(cachedir):
                os.makedirs(cachedir)
                os.makedirs(os.path.join(cachedir, "data"))
                os.makedirs(os.path.join(cachedir, "registry"))
        except OSError:
            return None, set()
        cached_corpora = self._copy_corpora_from_cache(cachedir)
        return cachedir, cached_corpora

    def _copy_corpora_from_cache(self, cachedir):
        """Copy corpus CWB data and registry under cachedir to test corpus dir.

        Return ids of copied corpora as a set.
        """
        cached_corpora = set(
            os.path.basename(regfile)
            for regfile in glob.glob(os.path.join(cachedir, "registry", "*")))
        for corpus_id in cached_corpora:
            self._copy_corpus(cachedir, self._corpus_root, corpus_id)
        return cached_corpora

    def _copy_corpus(self, source, target, corpus_id):
        """Copy CWB data for corpus corpus_id from source to target dir.

        The data and info paths in the corpus registry file are
        adjusted for target.
        """
        subprocess.run(["cp", "-dpr",
                        os.path.join(source, "data", corpus_id),
                        os.path.join(target, "data")])
        with open(os.path.join(source, "registry", corpus_id), "r") as in_regf:
            with open(os.path.join(target, "registry", corpus_id),
                      "w") as out_regf:
                for line in in_regf:
                    if line.startswith(("HOME", "INFO")):
                        line = line.replace(source, target)
                    out_regf.write(line)

    def _cached_corpus_is_outdated(self, cachedir, corpus_id, src_file):
        """Test if CWB data for corpus_id in cachedir is older than source.

        Consider cached data outdated if cachedir does not exist, if
        the corpus registry file does not exist or if it is older than
        the corpus source file, its info file or its attributes YAML
        file.
        """

        def get_mtime(fname):
            """Get modification time for file fname; 0 if it does not exist."""
            if not os.path.exists(fname):
                return 0
            return os.path.getmtime(fname)

        if cachedir is None:
            return True
        cached_regfile = os.path.join(cachedir, "registry", corpus_id)
        if not os.path.exists(cached_regfile):
            return True
        cached_regfile_mtime = os.path.getmtime(cached_regfile)
        src_file_noext = os.path.splitext(src_file)[0]
        return (
            cached_regfile_mtime < get_mtime(src_file)
            or cached_regfile_mtime < get_mtime(f"{src_file_noext}.info")
            or cached_regfile_mtime < get_mtime(f"{src_file_noext}.attrs.yaml"))

    def encode_corpus(self, corpus_id, vrt_file, corpus_src_dir):
        """Encode vrt_file with corpus_id."""
        self.encode_vrt_file(corpus_id, vrt_file, corpus_src_dir)
        self.cwb_makeall(corpus_id)
        self.copy_info_file(corpus_id, corpus_src_dir)

    def encode_vrt_file(self, corpus_id, vrt_file, corpus_src_dir):
        """Run cwb-encode for vrt_file for corpus_id."""

        def interleave(s, seq):
            """Return [s, seq[0], s, seq[1], ... , s, seq[-1]."""
            return [*chain(*zip_longest([], seq, fillvalue=s))]

        attrs = self._get_attrs(vrt_file, corpus_src_dir)
        # print(attrs)
        datadir = os.path.join(self._datarootdir, corpus_id)
        os.makedirs(datadir, exist_ok=True)
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

    def _get_attrs(self, fname, attrsfile_dir=None):
        """Return positional and structural attributes for corpus file fname.

        If the file corpus.attrs.yaml exists for corpus file
        corpus.vrt, use the attribute information in it as the basis
        (_get_attrs_from_attrsfile). If the attributes file does not
        exist or it does not list both positional and structural
        attributes, amend the attribute information with that in (or
        inferred from) corpus.vrt (_get_attrs_from_vrt).

        If attrsfile_dir is not None, read the .attrs.yaml file from
        there instead of the directory of fname.

        Returns dict
            { "positional": ["attr1", "attr2", ...],
              "structural": ["text:0+a1+a2", "sentence:0+a3+a4", ...] }
        so the attribute specifications can be used as values for
        cwb-encode -P and -S declarations.
        """
        attrs_dir = attrsfile_dir or os.path.dirname(fname)
        attrs_fname = os.path.join(
            attrs_dir,
            os.path.splitext(os.path.basename(fname))[0] + ".attrs.yaml")
        attrs = {}
        if os.path.exists(attrs_fname):
            attrs = self._get_attrs_from_attrsfile(attrs_fname)
        if "positional" not in attrs or "structural" not in attrs:
            attrs = self._get_attrs_from_vrt(fname, attrs)
        return attrs

    def _get_attrs_from_attrsfile(self, attrs_fname):
        """Return attribute information declared in YAML file attrs_fname.

        The content of attrs_fname should be as follows:
            pos_attributes: ["attr1", "attr2", ...]
            struct_attributes:
            - text: ["a1", "a2", ...]
            - sentence: ["a3", "a4", ...]
            ...
        In addition, if a structural attribute can be recursively
        nested, its name should be followed by the recursive nesting
        depth, separated by a space or colon:
            - div 2: ["a5", ...]
            - np:2: []
        """
        with open(attrs_fname, "r") as attrsf:
            attr_info = yaml.safe_load(attrsf) or {}
        attrs = {}
        if "pos_attributes" in attr_info:
            attrs["positional"] = attr_info["pos_attributes"]
        if "struct_attributes" in attr_info:
            attrs["structural"] = []
            for struct_attrs in attr_info["struct_attributes"]:
                for structname, attrnames in struct_attrs.items():
                    parts = re.split(r"[:\s]+", structname, 1)
                    structname = parts[0]
                    depth = parts[1] if len(parts) > 1 else "0"
                    attrs["structural"].append(
                        self._make_struct_spec(structname, depth, attrnames))
        return attrs

    def _make_struct_spec(self, name, depth, attrnames):
        """Make structural attribute specification for cwb-encode.

        name is the name of the structural attribute, depth is the
        recursive nesting depth and attrnames the annotation names.
        """
        return (f"{name}:{depth}"
                + "".join(f"+{attrname}" for attrname in attrnames))

    def _get_attrs_from_vrt(self, vrt_file, attrs=None):
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

        If attrs is dict and already contains "positional" or
        "structural", preserve their values intact.
        """
        attrs = attrs or {}
        for key in ["positional", "structural"]:
            attrs.setdefault(key, [])
        # Processing line before the first token or structure tag
        in_header = True
        # Annotation names of each structural attribute: values are
        # dicts of booleans indicating whether the annotation is
        # feature-set-valued (all values begin and end with "|") or
        # not
        struct_attrs = defaultdict(lambda: defaultdict(lambda: True))
        # The number of each structural attribute currently open
        open_structs = defaultdict(int)
        # For each structural attribute, the maximum nesting depth
        struct_maxdepth = defaultdict(int)
        # List of positional attributes: values are booleans
        # indicating if all the attribute values are feature-set
        # values
        pos_attr_is_featset = []
        pos_attr_count = 0
        with open(vrt_file, "r") as vrtf:
            for line in vrtf:
                if line[0] == "<":
                    if in_header and line.startswith(
                            ("<!-- #vrt positional-attributes:",
                             "<!-- #vrt structural-attributes:")):
                        type_ = line.split()[2].split("-")[0]
                        if not attrs[type_]:
                            attrs[type_] = (
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
                            attrname_vals = dict(re.findall(r'(\w+?)="([^"]*)"',
                                                            line))
                            struct_attrs[structname].update(
                                dict((attrname,
                                      (struct_attrs[structname][attrname]
                                       and is_feature_set_value(attrval)))
                                     for attrname, attrval
                                     in attrname_vals.items()))
                elif line[0] != "\n":
                    if not attrs["positional"]:
                        # A positional-attributes comment was not
                        # encountered before the first token, so go
                        # through all the data to find out which of
                        # them are feature-set-valued
                        pos_attrs = line[:-1].split("\t")
                        if pos_attr_count == 0:
                            pos_attr_count = len(pos_attrs)
                            pos_attr_is_featset = pos_attr_count * [True]
                            in_header = False
                        for attrnum, attr in enumerate(pos_attrs):
                            pos_attr_is_featset[attrnum] = (
                                pos_attr_is_featset[attrnum]
                                and is_feature_set_value(attr))
        if not attrs["positional"]:
            pos_attr_names = (
                self._default_pos_attrs[:pos_attr_count]
                + [self._default_pos_attr_name + str(attrnum + 1)
                   for attrnum in range(len(self._default_pos_attrs),
                                        pos_attr_count)])
            attrs["positional"] = list(self._make_set_valued(
                dict((pos_attr_names[attrnum], pos_attr_is_featset[attrnum])
                     for attrnum in range(pos_attr_count))))
        if not attrs["structural"]:
            attrs["structural"] = [
                self._make_struct_spec(
                    structname, struct_maxdepth[structname] - 1,
                    self._make_set_valued(struct_attrs[structname]))
                for structname in struct_attrs.keys()]
        return attrs

    def _make_set_valued(self, attrdict):
        """Return list of attribute names with "/" suffixed to set-valued ones.

        attrdict is a dict whose keys are (positional) attribute names
        and boolean values indicate whether the attribute is
        feature-set-valued or not.
        """
        return (attr + ("/" if is_set_valued else "")
                for attr, is_set_valued in attrdict.items())

    def xml_file_to_vrt(self, xml_fname, vrt_fname):
        """Convert XML file named xml_fname to a VRT file named vrt_fname.

        The input XML file is assumed to be in a Sparv XML export
        format where each token is represented as a "token" element,
        the word form as its text content and positional attributes as
        attributes. The output VRT file has a positional-attributes
        comment, but no structural-attributes comment, as structural
        attributes are to be inferred from the VRT file.
        """
        with open(xml_fname, "r") as xmlf:
            xml = xmlf.read()
        vrt = self.xml_to_vrt(xml)
        with open(vrt_fname, "w") as vrtf:
            vrtf.write(vrt)

    def xml_to_vrt(self, xml_str):
        """Convert Sparv export XML xml_str to VRT and return as a str."""
        # Positional attributes: key is the attribute name and value
        # indicates if all the attribute values are feature-set
        # values. Feature-set positional attributes are also inferred
        # in _get_attrs_from_vrt, so this duplicates that
        # functionality. However, if removed the functionality here,
        # we would need to pass the attribute names from here to
        # _get_attrs_from_vrt in some other way than via the
        # positional-attributes comment.
        pos_attrs = {"word": False}

        def convert_elem(elem):
            """Recursively convert XML elem to intermediate VRT lines.

            Return a list of items corresponding to VRT lines. Tag
            lines as strings but token lines are dicts of attributes,
            as a token element may lack some attributes but all the
            VRT tokens need to have the same number of attributes in
            the same order.
            """
            nonlocal pos_attrs
            if elem.tag == "token":
                pos_attrs.update(dict((key, (pos_attrs.get(key, True)
                                             and is_feature_set_value(val)))
                                      for key, val in elem.items()))
                line = dict(elem.items())
                line["word"] = elem.text
                return [line]
            else:
                lines = [f"<{elem.tag}"
                         + "".join(f" {name}=\"" + escape(val, {"\"": "&quot;"})
                                   + "\""
                                   for name, val in elem.items())
                         + ">\n"]
                for subelem in elem:
                    lines.extend(convert_elem(subelem))
                lines.append(f"</{elem.tag}>\n")
                return lines

        lines = convert_elem(et.XML(xml_str))
        pos_attrs_s = " ".join(self._make_set_valued(pos_attrs))
        for linenr, line in enumerate(lines):
            if isinstance(line, dict):
                lines[linenr] = ("\t".join(escape(line.get(attr, ""))
                                           for attr in pos_attrs)
                                 + "\n")
        lines[0:0] = [f"<!-- #vrt positional-attributes: {pos_attrs_s} -->\n"]
        return "".join(lines)

    def cwb_makeall(self, corpus_id):
        """Run cwb-makeall for corpus corpus_id."""
        subprocess.run([
            "cwb-makeall",
            "-r", self._registrydir,
            corpus_id
        ]).check_returncode()

    def copy_info_file(self, corpus_id, corpus_src_dir):
        """Copy corpus.info from source dir to the corpus data dir as .info."""
        info_file = os.path.join(corpus_src_dir, corpus_id + ".info")
        if os.path.isfile(info_file):
            subprocess.run([
                "cp",
                "-p",
                info_file,
                os.path.join(self._datarootdir, corpus_id, ".info")
            ])
