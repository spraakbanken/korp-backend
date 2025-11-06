
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
        self._tmpdir = os.path.join(corpus_root, "tmp")
        os.makedirs(self._datarootdir)
        os.makedirs(self._registrydir)
        os.makedirs(self._tmpdir)

    def encode_corpora(self, corpus_src_dir):
        """Encode VRT and XML data in corpus_src_dir, base name as corpus id."""
        corpus_ids = []
        vrt_files = glob.glob(os.path.join(corpus_src_dir, "*.vrt"))
        # Convert XML files to VRT and encode the VRT files
        for xml_file in glob.glob(os.path.join(corpus_src_dir, "*.xml")):
            vrt_file = os.path.join(
                self._tmpdir,
                os.path.splitext(os.path.basename(xml_file))[0] + ".vrt")
            self.xml_file_to_vrt(xml_file, vrt_file)
            vrt_files.append(vrt_file)
        for vrt_file in vrt_files:
            corpus_id = os.path.splitext(os.path.basename(vrt_file))[0]
            self.encode_corpus(corpus_id, vrt_file, corpus_src_dir)
            corpus_ids.append(corpus_id)
        return corpus_ids

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

    def _get_attrs(self, fname, attrsfile_dir=None):
        """Return positional and structural attributes for corpus file fname.

        If the file corpus.attrs.yaml exists for corpus file
        corpus.vrt, return the attribute information in it
        (_get_attrs_from_attrsfile); otherwise, return the attribute
        information in (or inferred from) corpus.vrt
        (_get_attrs_from_vrt).

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
        if os.path.exists(attrs_fname):
            return self._get_attrs_from_attrsfile(attrs_fname)
        else:
            return self._get_attrs_from_vrt(fname)

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
        attrs = {
            "positional": attr_info.get("pos_attributes", []),
            "structural": [],
        }
        for struct_attrs in attr_info.get("struct_attributes", []):
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

    def _get_attrs_from_vrt(self, vrt_file):
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
        attrs = {
            "positional": [],
            "structural": [],
        }
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
                            attrname_vals = dict(re.findall(r'(\w+?)="([^"]*)"',
                                                            line))
                            struct_attrs[structname].update(
                                dict((attrname,
                                      (struct_attrs[structname][attrname]
                                       and is_feature_set_value(attrval)))
                                     for attrname, attrval
                                     in attrname_vals.items()))
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
        # indicates if all the attribute values are feature-set values
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
