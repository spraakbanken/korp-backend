"""Unit tests for test utility module tests.corpusutils.

These tests are self-contained in that they do not use corpus source data in tests/data/corpora/src; instead, test
corpus data is created in the tests themselves.
"""

import pathlib
from collections.abc import Callable, Generator
from pathlib import Path
from typing import ClassVar

import pytest

from tests.corpusutils import CWBEncoder


@pytest.fixture(scope="session")
def corpus_test_data_root(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return a temporary directory for CWB files for the session.

    Also create subdirectory "src" under the temporary directory.
    """
    rootdir = tmp_path_factory.mktemp("corpora")
    (rootdir / "src").mkdir(parents=True)
    return rootdir


@pytest.fixture(scope="session")
def encoder(corpus_test_data_root: Path) -> CWBEncoder:
    """Return a CWBEncoder instance for the session."""
    return CWBEncoder(corpus_test_data_root)


@pytest.fixture(scope="session")
def corpus_file_name(corpus_test_data_root: Path) -> Callable:
    """Return a function returning the full path for a corpus source file.

    The returned function takes as its arguments the base file name of the corpus source file.
    """

    def _corpusfile_name(filename: str) -> Path:
        """Return the full path for the corpus source file `filename`."""
        return corpus_test_data_root / "src" / filename

    return _corpusfile_name


@pytest.fixture
def corpus_file(corpus_file_name: Callable[[str], Path]) -> Generator[Callable[[str, str], Path], None, None]:
    """Yield a function creating a corpus source file.

    The returned function takes as its arguments the base name of the file and the content of the file as a string. The
    created files are removed in tear-down.
    """
    created_files = []

    def _corpus_file(name: str, content: str) -> Path:
        """Create corpus source file named name with content.

        Returns:
            The full path of the created file.
        """
        path = corpus_file_name(name)
        with path.open("w", encoding="utf-8") as f:
            f.write(content)
            created_files.append(path)
            return path

    yield _corpus_file
    for path in created_files:
        pathlib.Path(path).unlink()


class TestGetAttrs:
    """Tests for CWBEncoder._get_attrs."""

    # VRT file content
    vrt_content = """<text>
<paragraph id="p1">
<sentence id="s1">
<span n="1">
<span n="2">
This\t|this|
</span>
<span n="3">
<span n="4">
is\t|be|
</span>
a\t|a|
</span>
test\t|test|
.\t|.|
</span>
</sentence>
<sentence id="s2" a="|2|3|">
<span n="5">
Great\t|great|
!\t|!|
</span>
</sentence>
</paragraph>
</text>
"""
    # Positional and structural attributes inferred from VRT
    pos_attrs_inferred: ClassVar = ["word", "lemma/"]
    struct_attrs_inferred: ClassVar = ["text:0", "paragraph:0+id", "sentence:0+id+a/", "span:2+n"]
    # Positional and structural attributes comments; note that to test that the attributes are taken from the comments
    # and not inferred from VRT, these differ from the inferred ones
    pos_attr_comment = "<!-- #vrt positional-attributes: word lemma1 -->\n"
    struct_attr_comment = "<!-- #vrt structural-attributes:text:0+id paragraph:0+id sentence:0+id1 -->\n"
    # Positional and structural attributes from the comments
    pos_attrs_from_comment: ClassVar = ["word", "lemma1"]
    struct_attrs_from_comment: ClassVar = ["text:0+id", "paragraph:0+id", "sentence:0+id1"]
    # YAML attributes file content, positional and structural separately, to make it easier to test with only one of
    # those; similarly to the above, the specified attributes differ from the actual attributes in vrt_content for
    # testing purposes
    attrs_content_pos = """
pos_attributes:
- word
- lemma2
"""
    attrs_content_struct = """
struct_attributes:
- text: []
- paragraph:
  - id1
- sentence:
  - id
  - a
- div 2:
  - n/
"""
    # The same as above, but with a colon separating the recursive nesting depth from the structural attribute name
    attrs_content_struct_colon = attrs_content_struct.replace("div 2", "div:2")
    # Empty positional and structural attributes
    attrs_content_pos_empty = "\npos_attributes: []\n"
    attrs_content_struct_empty = "\nstruct_attributes: []\n"
    # Positional and structural attributes from YAML attributes file
    pos_attrs_from_attrsfile: ClassVar = ["word", "lemma2"]
    struct_attrs_from_attrsfile: ClassVar = ["text:0", "paragraph:0+id1", "sentence:0+id+a", "div:2+n/"]

    @staticmethod
    def _assert_get_attrs_result(
        vrt_content: str,
        attrs_content: str | None,
        result_pos: list[str],
        result_struct: list[str],
        corpusfile: Callable[[str, str], Path],
        encoder: CWBEncoder,
    ) -> None:
        """Assert that `encoder._get_attrs` returns the expected positional and structural attributes.

        Given VRT file content `vrt_content` and attributes YAML file content `attrs_content`, assert that
        `encoder._get_attrs` returns positional attributes `result_pos` and structural attributes `result_struct`. The
        values of `corpusfile` and `encoder` should be the respective Pytest fixtures: since this is not a test in
        itself, it cannot request fixtures directly.
        """
        vrt_filename = corpusfile("corpus.vrt", vrt_content)
        if attrs_content is not None:
            corpusfile("corpus.attrs.yaml", attrs_content)
        assert encoder._get_attrs(vrt_filename) == {
            "positional": result_pos,
            "structural": result_struct,
        }

    @pytest.mark.parametrize(
        ("comment", "header", "result_pos", "result_struct"),
        [
            ("infer attributes", "", pos_attrs_inferred, struct_attrs_inferred),
            (
                "positional and structural attributes comments",
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_comment,
                struct_attrs_from_comment,
            ),
            ("only positional-attributes comment", pos_attr_comment, pos_attrs_from_comment, struct_attrs_inferred),
            ("only structural-attributes comment", struct_attr_comment, pos_attrs_inferred, struct_attrs_from_comment),
        ],
    )
    def test_get_attrs_vrt(
        self,
        comment: str,  # noqa: ARG002
        header: str,
        result_pos: list[str],
        result_struct: list[str],
        corpus_file: Callable,
        encoder: CWBEncoder,
    ) -> None:
        """Test with a VRT file with the given header.

        The content of the VRT file is self.vrt_content, prepended with header (the possible positional and structural
        attributes comments), result_pos is the resulting positional attributes and result_struct structural attributes.
        comment is only for documentation.
        """
        self._assert_get_attrs_result(header + self.vrt_content, None, result_pos, result_struct, corpus_file, encoder)

    def test_get_attrs_vrt_many_pos_attrs(self, corpus_file: Callable[[str, str], Path], encoder: CWBEncoder) -> None:
        """Test with VRT with more than 8 positional attributes, no comment.

        The source VRT file has no positional-attributes comment, so `_get_attrs` returns default names.
        """
        pos_attrs = CWBEncoder._default_pos_attrs + [CWBEncoder._default_pos_attr_name + str(i) for i in range(9, 11)]
        feat_set_pos_attr_nums = [4, 7, 9]
        for i in feat_set_pos_attr_nums:
            pos_attrs[i] += "/"
        self._assert_get_attrs_result(
            (
                "<text>\n<sentence>\n"
                + "\t".join(f"|{i}|" if i in feat_set_pos_attr_nums else f"{i}" for i in range(10))
                + "\n</sentence>\n</text>"
            ),
            None,
            pos_attrs,
            ["text:0", "sentence:0"],
            corpus_file,
            encoder,
        )

    @pytest.mark.parametrize(
        ("comment", "attrsfile", "header", "result_pos", "result_struct"),
        [
            (
                "positional and structural attributes, no VRT comments",
                attrs_content_pos + attrs_content_struct,
                "",
                pos_attrs_from_attrsfile,
                struct_attrs_from_attrsfile,
            ),
            (
                "positional and structural attributes, VRT comments",
                attrs_content_pos + attrs_content_struct,
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_attrsfile,
                struct_attrs_from_attrsfile,
            ),
            (
                "positional and structural attributes (colon), VRT comments",
                attrs_content_pos + attrs_content_struct_colon,
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_attrsfile,
                struct_attrs_from_attrsfile,
            ),
            (
                "positional attributes only, no VRT comments",
                attrs_content_pos,
                "",
                pos_attrs_from_attrsfile,
                struct_attrs_inferred,
            ),
            (
                "positional attributes only, VRT comments",
                attrs_content_pos,
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_attrsfile,
                struct_attrs_from_comment,
            ),
            (
                "structural attributes only, no VRT comments",
                attrs_content_struct,
                "",
                pos_attrs_inferred,
                struct_attrs_from_attrsfile,
            ),
            (
                "structural attributes only, VRT comments",
                attrs_content_struct,
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_comment,
                struct_attrs_from_attrsfile,
            ),
            (
                "empty positional, non-empty structural attributes, VRT comments",
                attrs_content_pos_empty + attrs_content_struct,
                pos_attr_comment + struct_attr_comment,
                [],
                struct_attrs_from_attrsfile,
            ),
            (
                "non-empty positional, empty structural attributes, VRT comments",
                attrs_content_pos + attrs_content_struct_empty,
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_attrsfile,
                [],
            ),
            (
                "empty positional attributes only, VRT comments",
                attrs_content_pos_empty,
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_comment,
                struct_attrs_from_comment,
            ),
            (
                "empty structural attributes only, VRT comments",
                attrs_content_struct_empty,
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_comment,
                struct_attrs_from_comment,
            ),
            ("no attributes, no VRT comments", "", "", pos_attrs_inferred, struct_attrs_inferred),
            (
                "no attributes, VRT comments",
                "",
                pos_attr_comment + struct_attr_comment,
                pos_attrs_from_comment,
                struct_attrs_from_comment,
            ),
        ],
    )
    def test_get_attrs_attrsfile(
        self,
        header: str,
        comment: str,  # noqa: ARG002
        attrsfile: str,
        result_pos: list,
        result_struct: list,
        corpus_file: Callable[[str, str], Path],
        encoder: CWBEncoder,
    ) -> None:
        """Test with an attributes YAML file with the specified content.

        The arguments are similar to those of test_get_attrs_vrt, but
        attrsfile is the content of the attributes YAML file to
        accompany the VRT file.
        """
        self._assert_get_attrs_result(
            header + self.vrt_content, attrsfile, result_pos, result_struct, corpus_file, encoder
        )
