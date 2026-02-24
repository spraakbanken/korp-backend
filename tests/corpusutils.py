"""Utility functions used in pytest tests for Korp, in particular for setting up CWB corpus data."""

import re
import subprocess
import xml.etree.ElementTree as et  # noqa: N813
from collections import defaultdict
from itertools import chain, zip_longest
from pathlib import Path
from typing import ClassVar
from xml.sax.saxutils import escape

import yaml


def is_feature_set_value(val: str) -> bool:
    """Return True if `val` is a CWB feature-set value."""
    return bool(val and val[0] == val[-1] == "|")


class CWBEncoder:
    """Encode VRT data to a CWB corpus."""

    # Default positional attribute names if none specified in input
    _default_pos_attrs: ClassVar = [
        "word",
        "lemma",
        "pos",
        "msd",
        "deprel",
        "dephead",
        "ref",
        "lex",
    ]
    # Default name (prefix) for the rest of the positional attributes (beginning from the 9th attribute)
    _default_pos_attr_name = "attr"

    def __init__(self, corpus_root: Path, cwb_encode: str | None = None, cwb_makeall: str | None = None) -> None:
        """Initialize with paths for corpus root, cwb-encode, cwb-makeall."""
        cwb_encode = cwb_encode or "cwb-encode"
        cwb_makeall = cwb_makeall or "cwb-makeall"
        corpus_root = corpus_root.expanduser().resolve()
        self._corpus_root = corpus_root
        self._datarootdir = corpus_root / "data"
        self._registrydir = corpus_root / "registry"
        self._tmpdir = corpus_root / "tmp"
        self._datarootdir.mkdir(parents=True, exist_ok=True)
        self._registrydir.mkdir(parents=True, exist_ok=True)
        self._tmpdir.mkdir(parents=True, exist_ok=True)

    def encode_corpora(self, corpus_src_dir: Path) -> list[str]:
        """Encode VRT and XML data in corpus_src_dir, base name as corpus id.

        Returns:
            List of corpus ids.
        """
        corpus_ids = []
        cache_dir, _cached_corpora = self._init_cwb_cache(corpus_src_dir)
        vrt_files = list(corpus_src_dir.glob("*.vrt"))
        # Convert XML files to VRT and encode the VRT files
        for xml_file in corpus_src_dir.glob("*.xml"):
            corpus_id = xml_file.stem
            if self._cached_corpus_is_outdated(cache_dir, corpus_id, xml_file):
                vrt_file = self._tmpdir / f"{corpus_id}.vrt"
                self.xml_file_to_vrt(xml_file, vrt_file)
                vrt_files.append(vrt_file)
        for vrt_file in vrt_files:
            corpus_id = vrt_file.stem
            if self._cached_corpus_is_outdated(cache_dir, corpus_id, vrt_file):
                self.encode_corpus(corpus_id, vrt_file, corpus_src_dir)
                if cache_dir is not None:
                    self._copy_corpus(self._corpus_root, cache_dir, corpus_id)
            corpus_ids.append(corpus_id)
        return corpus_ids

    def _init_cwb_cache(self, corpus_src_dir: Path) -> tuple[Path | None, set[str]]:
        """Initialize CWB data caching, copy corpora from cache.

        Create a CWB data cache directory as `corpus_src_dir/../cwb-cache` if it does not yet exits. If the cache
        directory exists, copy corpora from the cache to the corpus data directory.

        Args:
            corpus_src_dir: The directory where the source corpus files are located.

        Returns:
            A tuple containing the path to the cache directory and a set of cached corpus IDs. If the cache directory
                cannot be created, returns (None, set()).
        """
        cache_dir = corpus_src_dir.parent / "cwb-cache"
        try:
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
                (cache_dir / "data").mkdir(parents=True, exist_ok=True)
                (cache_dir / "registry").mkdir(parents=True, exist_ok=True)
        except OSError:
            return None, set()
        cached_corpora = self._copy_corpora_from_cache(cache_dir)
        return cache_dir, cached_corpora

    def _copy_corpora_from_cache(self, cache_dir: Path) -> set[str]:
        """Copy corpus CWB data and registry under `cache_dir` to test corpus dir.

        Returns:
            A set of corpus IDs that were copied from the cache.
        """
        cached_corpora = {regfile.name for regfile in (cache_dir / "registry").glob("*")}
        for corpus_id in cached_corpora:
            self._copy_corpus(cache_dir, self._corpus_root, corpus_id)
        return cached_corpora

    @staticmethod
    def _copy_corpus(source: Path, target: Path, corpus_id: str) -> None:
        """Copy CWB data for corpus `corpus_id` from `source` to `target` dir.

        The data and info paths in the corpus registry file are adjusted for target.

        Args:
            source: The directory where the corpus data is currently located.
            target: The directory where the corpus data should be copied to.
            corpus_id: The ID of the corpus to be copied.
        """
        subprocess.run(["cp", "-dpr", source / "data" / corpus_id, target / "data"], check=True)
        with (
            (source / "registry" / corpus_id).open("r") as in_regf,
            (target / "registry" / corpus_id).open("w") as out_regf,
        ):
            for line in in_regf:
                if line.startswith(("HOME", "INFO")):
                    line = line.replace(str(source), str(target))  # noqa: PLW2901
                out_regf.write(line)

    @staticmethod
    def _cached_corpus_is_outdated(cache_dir: Path | None, corpus_id: str, src_file: Path) -> bool:
        """Test if CWB data for `corpus_id` in `cache_dir` is older than source.

        Consider cached data outdated if `cache_dir` does not exist, if the corpus registry file does not exist or if it
        is older than the corpus source file, its info file or its attributes YAML file.

        Args:
            cache_dir: The directory where the cached corpus data is located.
            corpus_id: The ID of the corpus to check.
            src_file: The source file for the corpus, either a VRT or an XML file.

        Returns:
            True if the cached corpus data is outdated, False otherwise.
        """

        def get_mtime(fname: Path) -> float:
            """Return modification time for file fname; 0 if it does not exist."""
            if not fname.exists():
                return 0
            return fname.stat().st_mtime

        if cache_dir is None:
            return True
        cached_regfile = cache_dir / "registry" / corpus_id
        if not cached_regfile.exists():
            return True
        cached_regfile_mtime = cached_regfile.stat().st_mtime
        src_file_noext = src_file.with_suffix("")
        return (
            cached_regfile_mtime < get_mtime(src_file)
            or cached_regfile_mtime < get_mtime(src_file_noext.with_suffix(".info"))
            or cached_regfile_mtime < get_mtime(src_file_noext.with_suffix(".attrs.yaml"))
        )

    def encode_corpus(self, corpus_id: str, vrt_file: Path, corpus_src_dir: Path) -> None:
        """Encode `vrt_file` with `corpus_id`."""
        self.encode_vrt_file(corpus_id, vrt_file, corpus_src_dir)
        self.cwb_makeall(corpus_id)
        self.copy_info_file(corpus_id, corpus_src_dir)

    def encode_vrt_file(self, corpus_id: str, vrt_file: Path, corpus_src_dir: Path) -> None:
        """Run cwb-encode for `vrt_file` for `corpus_id`."""

        def interleave(s: str, seq: list[str]) -> list[str]:
            """Return [s, seq[0], s, seq[1], ... , s, seq[-1]."""
            return [*chain(*zip_longest([], seq, fillvalue=s))]

        attrs = self._get_attrs(vrt_file, corpus_src_dir)
        data_dir = self._datarootdir / corpus_id
        data_dir.mkdir(exist_ok=True, parents=True)
        subprocess.run(
            [
                "cwb-encode",
                "-f",
                vrt_file,
                "-d",
                data_dir,
                "-R",
                self._registrydir / corpus_id,
                "-xsB",
                "-c",
                "utf8",
                "-p",
                "-",
                *interleave("-P", attrs["positional"]),
                *interleave("-S", attrs["structural"]),
            ],
            check=True,
        )

    def _get_attrs(self, fname: Path, attrsfile_dir: Path | None = None) -> dict[str, list[str]]:
        """Return positional and structural attributes for corpus file fname.

        If the file corpus.attrs.yaml exists for corpus file corpus.vrt, use the attribute information in it as the
        basis (`_get_attrs_from_attrsfile`). If the attributes file does not exist or it does not list both positional
        and structural attributes, amend the attribute information with that in (or inferred from) corpus.vrt
        (`_get_attrs_from_vrt`).

        If `attrsfile_dir` is not `None`, read the .attrs.yaml file from there instead of the directory of fname.

        Returns dict
            { "positional": ["attr1", "attr2", ...],
              "structural": ["text:0+a1+a2", "sentence:0+a3+a4", ...] }
        so the attribute specifications can be used as values for cwb-encode -P and -S declarations.
        """
        attrs_dir = attrsfile_dir or fname.parent
        attrs_fname = attrs_dir / Path(fname.name).with_suffix(".attrs.yaml")
        attrs = {}
        if attrs_fname.exists():
            attrs = self._get_attrs_from_attrsfile(attrs_fname)
        if "positional" not in attrs or "structural" not in attrs:
            attrs = self._get_attrs_from_vrt(fname, attrs)
        return attrs

    def _get_attrs_from_attrsfile(self, attrs_fname: Path) -> dict[str, list[str]]:
        """Return attribute information declared in YAML file attrs_fname.

        The content of attrs_fname should be as follows:
            pos_attributes: ["attr1", "attr2", ...]
            struct_attributes:
            - text: ["a1", "a2", ...]
            - sentence: ["a3", "a4", ...]
            ...
        In addition, if a structural attribute can be recursively nested, its name should be followed by the recursive
        nesting depth, separated by a space or colon:
            - div 2: ["a5", ...]
            - np:2: []
        """
        with attrs_fname.open("r", encoding="utf-8") as attrsf:
            attr_info = yaml.safe_load(attrsf) or {}
        attrs = {}
        if "pos_attributes" in attr_info:
            attrs["positional"] = attr_info["pos_attributes"]
        if "struct_attributes" in attr_info:
            attrs["structural"] = []
            for struct_attrs in attr_info["struct_attributes"]:
                for structname, attrnames in struct_attrs.items():
                    parts = re.split(r"[:\s]+", structname, maxsplit=1)
                    structname = parts[0]  # noqa: PLW2901
                    depth = int(parts[1]) if len(parts) > 1 else 0
                    attrs["structural"].append(self._make_struct_spec(structname, depth, attrnames))
        return attrs

    @staticmethod
    def _make_struct_spec(name: str, depth: int, attrnames: list[str]) -> str:
        """Make structural attribute specification for cwb-encode.

        Args:
            name: The name of the structural attribute.
            depth: The recursive nesting depth of the structural attribute.
            attrnames: The annotation names of the structural attribute.

        Returns:
            A string of the form "name:depth+a1+a2+...+an" where a1, a2, ..., an are the annotation names of the
            structural attribute.
        """
        return f"{name}:{depth}" + "".join(f"+{attrname}" for attrname in attrnames)

    def _get_attrs_from_vrt(self, vrt_file: Path, attrs: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
        """Get the positional and strucutral attribute info from `vrt_file`.

        Assumes that `vrt_file` contains comments of the following kind before the first data line (token or structural
        attribute):

        <!-- #vrt positional-attributes: attr1 attr2 ... -->
        <!-- #vrt structural-attributes: text:0+a1+a2 sentence:0+a3+a4 ... -->

        Args:
            vrt_file: The VRT file to get the attribute information from.
            attrs: If not `None`, a dict with possibly already some of the attribute information. The attribute
                information in `attrs` is preserved and only missing information is added based on the content of
                `vrt_file`. The attribute information in `vrt_file` is ignored if the corresponding attribute
                information is already present in `attrs`.

        Returns:
            A dict with the positional and structural attribute information:
                {
                    "positional": ["attr1", "attr2", ...],
                    "structural": ["text:0+a1+a2", "sentence:0+a3+a4", ...]
                }
        """
        attrs = attrs or {}
        for key in ["positional", "structural"]:
            attrs.setdefault(key, [])
        # Processing line before the first token or structure tag
        in_header = True
        # Annotation names of each structural attribute: values are dicts of booleans indicating whether the annotation
        # is feature-set-valued (all values begin and end with "|") or not
        struct_attrs = defaultdict(lambda: defaultdict(lambda: True))
        # The number of each structural attribute currently open
        open_structs = defaultdict(int)
        # For each structural attribute, the maximum nesting depth
        struct_maxdepth = defaultdict(int)
        # List of positional attributes: values are booleans indicating if all the attribute values are feature-set
        # values
        pos_attr_is_featset = []
        pos_attr_count = 0
        with vrt_file.open("r", encoding="utf-8") as vrtf:
            for line in vrtf:
                if line[0] == "<":
                    if in_header and line.startswith(
                        ("<!-- #vrt positional-attributes:", "<!-- #vrt structural-attributes:")
                    ):
                        type_ = line.split()[2].split("-")[0]
                        if not attrs[type_]:
                            attrs[type_] = line.partition(":")[2].strip(" ->\n").split()
                        if attrs["structural"] and attrs["positional"]:
                            return attrs
                    elif line[1] not in "!?":
                        in_header = False
                        structname = re.search(r"\w+", line).group(0)  # type: ignore
                        if line[1] == "/":
                            open_structs[structname] -= 1
                        else:
                            open_structs[structname] += 1
                            struct_maxdepth[structname] = max(struct_maxdepth[structname], open_structs[structname])
                            # Should we also allow attribute values enclosed in single quotes?
                            attrname_vals = dict(re.findall(r'(\w+?)="([^"]*)"', line))
                            struct_attrs[structname].update(
                                {
                                    attrname: (struct_attrs[structname][attrname] and is_feature_set_value(attrval))
                                    for attrname, attrval in attrname_vals.items()
                                }
                            )
                elif line[0] != "\n" and not attrs["positional"]:
                    # A positional-attributes comment was not encountered before the first token, so go through all the
                    # data to find out which of them are feature-set-valued
                    pos_attrs = line[:-1].split("\t")
                    if pos_attr_count == 0:
                        pos_attr_count = len(pos_attrs)
                        pos_attr_is_featset = pos_attr_count * [True]
                        in_header = False
                    for attrnum, attr in enumerate(pos_attrs):
                        pos_attr_is_featset[attrnum] = pos_attr_is_featset[attrnum] and is_feature_set_value(attr)
        if not attrs["positional"]:
            pos_attr_names = self._default_pos_attrs[:pos_attr_count] + [
                self._default_pos_attr_name + str(attrnum + 1)
                for attrnum in range(len(self._default_pos_attrs), pos_attr_count)
            ]
            attrs["positional"] = list(
                self._make_set_valued(
                    {pos_attr_names[attrnum]: pos_attr_is_featset[attrnum] for attrnum in range(pos_attr_count)}
                )
            )
        if not attrs["structural"]:
            attrs["structural"] = [
                self._make_struct_spec(
                    structname, struct_maxdepth[structname] - 1, self._make_set_valued(struct_attrs[structname])
                )
                for structname in struct_attrs
            ]
        return attrs

    @staticmethod
    def _make_set_valued(attr_dict: dict[str, bool]) -> list[str]:
        """Return list of attribute names with "/" suffixed to set-valued ones.

        Args:
            attr_dict: A dict where keys are (positional) attribute names and values are booleans indicating whether the
                attribute is feature-set-valued or not.

        Returns:
            A list of attribute names, where set-valued attribute names are suffixed with "/".
        """
        return [attr + ("/" if is_set_valued else "") for attr, is_set_valued in attr_dict.items()]

    def xml_file_to_vrt(self, xml_fname: Path, vrt_fname: Path) -> None:
        """Convert XML file `xml_fname` to a VRT file named `vrt_fname`.

        The input XML file is assumed to be in a Sparv XML export format where each token is represented as a "token"
        element, the word form as its text content and positional attributes as attributes. The output VRT file has a
        positional-attributes comment, but no structural-attributes comment, as structural attributes are to be inferred
        from the VRT file.
        """
        xml = xml_fname.read_text(encoding="utf-8")
        vrt = self.xml_to_vrt(xml)
        vrt_fname.write_text(vrt, encoding="utf-8")

    def xml_to_vrt(self, xml_str: str) -> str:
        """Convert XML string `xml_str` to a VRT string.

        Args:
            xml_str: The XML string to be converted, assumed to be in a Sparv XML export format where each token is
                represented as a "token" element, the word form as its text content and positional attributes as
                attributes.

        Returns:
            The converted VRT string, with a positional-attributes comment, but no structural-attributes comment, as
            structural attributes are to be inferred from the VRT file.
        """
        # Positional attributes: key is the attribute name and value indicates if all the attribute values are
        # feature-set values. Feature-set positional attributes are also inferred in _get_attrs_from_vrt, so this
        # duplicates that functionality. However, if removed the functionality here, we would need to pass the attribute
        # names from here to _get_attrs_from_vrt in some other way than via the positional-attributes comment.
        pos_attrs = {"word": False}

        def convert_elem(elem: et.Element) -> list[str | dict]:
            """Recursively convert XML `elem` to intermediate VRT lines.

            Args:
                elem: The XML element to be converted.

            Returns:
                A list of items corresponding to VRT lines. Tag lines as strings but token lines are dicts of
                attributes, as a token element may lack some attributes but all the VRT tokens need to have the same
                number of attributes in the same order.
            """
            nonlocal pos_attrs
            if elem.tag == "token":
                pos_attrs.update(
                    {key: (pos_attrs.get(key, True) and is_feature_set_value(val)) for key, val in elem.items()}
                )
                line = dict(elem.items())
                line["word"] = elem.text or ""
                return [line]
            lines: list[str | dict] = [
                f"<{elem.tag}"
                + "".join(f' {name}="' + escape(val, {'"': "&quot;"}) + '"' for name, val in elem.items())
                + ">\n"
            ]
            for subelem in elem:
                lines.extend(convert_elem(subelem))
            lines.append(f"</{elem.tag}>\n")
            return lines

        lines = convert_elem(et.XML(xml_str))
        lines_str = []
        pos_attrs_s = " ".join(self._make_set_valued(pos_attrs))
        for line in lines:
            if isinstance(line, dict):
                lines_str.append("\t".join(escape(line.get(attr, "")) for attr in pos_attrs) + "\n")
            else:
                lines_str.append(line)
        lines_str[0:0] = [f"<!-- #vrt positional-attributes: {pos_attrs_s} -->\n"]
        return "".join(lines_str)

    def cwb_makeall(self, corpus_id: str) -> None:
        """Run cwb-makeall for corpus `corpus_id`."""
        subprocess.run(["cwb-makeall", "-r", self._registrydir, corpus_id], check=True)

    def copy_info_file(self, corpus_id: str, corpus_src_dir: Path) -> None:
        """Copy `corpus.info` from source dir to the corpus data dir as `.info`."""
        info_file = corpus_src_dir / (corpus_id + ".info")
        if info_file.is_file():
            subprocess.run(["cp", "-p", info_file, self._datarootdir / corpus_id / ".info"], check=True)
