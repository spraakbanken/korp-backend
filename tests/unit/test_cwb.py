"""Unit tests for `korp.cwb.CWB.run_cqp`.

The tests in `TestRunCqp` run the real CQP binary on the test corpora; those in `TestRunCqpMocked` replace
`subprocess.Popen` with a fake.
"""

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from korp.config import Settings
from korp.cqp import END_OF_LINE, CQPError
from korp.cwb import CWB
from korp.dependencies import AbortSignal
from tests.testutils import get_info_value

RunCqp = Callable[..., list[str]]

# Corpus used by the tests that need a known corpus; its `.info` file has `Sentences: 2`
TEST_CORPUS = "testcorpus"


def _make_cwb(registry: Path) -> CWB:
    """Return a `CWB` for `registry`, with the executables from the settings or found on `PATH`.

    The settings are read like in the application, including `.env`, so that the tests use the same CQP and locale as
    the functional tests.
    """
    settings = Settings()
    executable = settings.CQP_EXECUTABLE or shutil.which("cqp")
    scan_executable = settings.CWB_SCAN_EXECUTABLE or shutil.which("cwb-scan-corpus")
    if executable is None or scan_executable is None:
        pytest.skip("cqp or cwb-scan-corpus not found")
    return CWB(
        executable=Path(executable),
        scan_executable=Path(scan_executable),
        registry=registry,
        locale=settings.LC_COLLATE,
        encoding=settings.CQP_ENCODING,
    )


@pytest.fixture(scope="session")
def cwb(request: pytest.FixtureRequest, corpus_registry_dir: Path) -> CWB:
    """Return a `CWB` object using the test corpus registry, which `corpora` fills.

    `corpora` is requested only after `_make_cwb`, so that the tests are skipped instead of failing in encoding the
    corpora if CQP is not found.
    """
    cwb = _make_cwb(corpus_registry_dir)
    request.getfixturevalue("corpora")
    return cwb


@pytest.fixture
def run_cqp(cwb: CWB) -> RunCqp:
    """Return a function for getting the result of CQP commands as a list of lines.

    The returned function takes a CQP `command` or a list of commands, the id of the `corpus` to select before them
    (default: `TEST_CORPUS`) and keyword arguments passed to `CWB.run_cqp`. It returns the output lines of `command`
    run with `cwb.run_cqp`.
    """

    def _run_cqp(command: str | list[str], corpus: str = TEST_CORPUS, **kwargs: Any) -> list[str]:
        commands = [command] if isinstance(command, str) else command
        return list(cwb.run_cqp([f"{corpus.upper()};", *commands], **kwargs))

    return _run_cqp


class TestRunCqp:
    """Tests for `CWB.run_cqp` with the real CQP."""

    @staticmethod
    def test_run_cqp_simple(run_cqp: RunCqp) -> None:
        """Run `info` and a simple query."""
        # CQP prints its version first in child mode (-c); `.EOL.;` prints `END_OF_LINE` to separate the `info` output
        # from the concordance of `<sentence> [];`, which has one line per sentence
        result = run_cqp(["info;", ".EOL.;", "<sentence> [];", "cat Last;"])

        assert result[0].startswith("CQP version")
        size = int(get_info_value("Sentences", result) or 0)
        assert size
        assert len(result) - result.index(END_OF_LINE) - 1 == size

    @staticmethod
    def test_run_cqp_no_such_attribute(run_cqp: RunCqp) -> None:
        """Showing a non-existent attribute raises `CQPError`."""
        with pytest.raises(CQPError) as excinfo:
            run_cqp("show +_zzz;")

        assert str(excinfo.value) == "No such attribute: _zzz"

    @staticmethod
    def test_run_cqp_attr_ignore(run_cqp: RunCqp) -> None:
        """Showing a non-existent attribute with `attr_ignore` does not raise and outputs only the CQP version."""
        result = run_cqp("show +_zzz;", attr_ignore=True)

        assert len(result) == 1
        assert result[0].startswith("CQP version")

    @staticmethod
    def test_run_cqp_broken_registry_file(registry_with_broken_file: Path, corpora: list[str]) -> None:
        """A registry file that CQP cannot parse makes only selecting that corpus fail; CQP does not list it."""
        cwb = _make_cwb(registry_with_broken_file)

        lines = list(cwb.run_cqp("show corpora;"))

        assert lines[0].startswith("CQP version")
        assert set(lines[1:]) == {corpus.upper() for corpus in corpora}
        with pytest.raises(CQPError, match="Corpus ``BROKENCORPUS'' is undefined"):
            list(cwb.run_cqp("BROKENCORPUS;"))

    @staticmethod
    def test_run_cqp_syntax_error(run_cqp: RunCqp) -> None:
        """A syntax error raises `CQPError`."""
        with pytest.raises(CQPError, match="CQP Syntax Error:"):
            run_cqp("---;")

    @staticmethod
    def test_run_cqp_syntax_error_non_ascii(run_cqp: RunCqp) -> None:
        """A syntax error at a non-ASCII character raises `CQPError`, although CQP echoes only part of the character."""
        with pytest.raises(CQPError, match="CQP Syntax Error:"):
            run_cqp("[word='x'] ä;")


class FakePopen:
    """A stand-in for a `subprocess.Popen` process that records its calls and returns canned output."""

    def __init__(self, args: list[Any], kwargs: dict[str, Any], stdout: bytes, stderr: bytes, timeouts: int) -> None:
        """Record the arguments and set the canned output.

        Args:
            args: The positional arguments passed to `Popen`.
            kwargs: The keyword arguments passed to `Popen`.
            stdout: The canned stdout returned by `communicate`.
            stderr: The canned stderr returned by `communicate`.
            timeouts: The number of calls to `communicate` that raise `TimeoutExpired` before the output is returned.
        """
        self.args = args
        self.kwargs = kwargs
        self.stdout = stdout
        self.stderr = stderr
        self.timeouts = timeouts
        self.input: bytes | None = None
        self.pid = 12345
        self.killed = False

    def communicate(
        self,
        input: bytes | None = None,  # ruff: ignore[builtin-argument-shadowing]
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        """Record the input and return the canned output, after first timing out `timeouts` times.

        Args:
            input: Data sent to the process; named as in `Popen.communicate`, so that it can also be passed by keyword.
            timeout: The timeout passed to `TimeoutExpired`.

        Returns:
            The canned stdout and stderr.

        Raises:
            ValueError: If input is sent again, as in `Popen.communicate` after a timeout.
            subprocess.TimeoutExpired: If the call should time out.
        """
        if input is not None:
            if self.input is not None:
                raise ValueError("Cannot send input after starting communication")
            self.input = input
        if self.timeouts > 0:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("cqp", timeout or 0)
        return self.stdout, self.stderr

    def kill(self) -> None:
        """Record that the process was killed."""
        self.killed = True


FakePopenFactory = Callable[..., list[FakePopen]]


@pytest.fixture
def fake_popen(monkeypatch: pytest.MonkeyPatch) -> FakePopenFactory:
    """Return a function that replaces `subprocess.Popen` in `korp.cwb` with one creating `FakePopen`s.

    The returned function takes the canned `stdout`, `stderr` and number of `timeouts` of the processes (see
    `FakePopen`) and returns a list to which the processes are appended when they are created.
    """

    def _fake_popen(stdout: bytes = b"", stderr: bytes = b"", timeouts: int = 0) -> list[FakePopen]:
        processes: list[FakePopen] = []

        def _popen(args: list[Any], **kwargs: Any) -> FakePopen:
            # Pass `kwargs` as a dict, as it contains the `Popen` arguments `stdout` and `stderr`
            process = FakePopen(args, kwargs, stdout, stderr, timeouts)
            processes.append(process)
            return process

        monkeypatch.setattr("korp.cwb.subprocess.Popen", _popen)
        return processes

    return _fake_popen


@pytest.fixture
def mock_cwb(tmp_path: Path) -> CWB:
    """Return a `CWB` that does not need any real executables."""
    return CWB(Path("cqp"), Path("cwb-scan-corpus"), tmp_path, "sv_SE.UTF-8", "UTF-8")


class TestRunCqpMocked:
    """Tests for `CWB.run_cqp` with `subprocess.Popen` mocked."""

    @staticmethod
    def test_command_prefix(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """The command is prefixed with `set PrettyPrint off;` and passed to CQP with the registry and locale."""
        processes = fake_popen(stdout=b"a\n")

        assert list(mock_cwb.run_cqp("info;")) == ["a"]

        process = processes[0]
        assert process.input == b"set PrettyPrint off;\ninfo;"
        assert process.args == [Path("cqp"), "-c", "-r", mock_cwb.registry]
        assert process.kwargs["env"]["LC_COLLATE"] == "sv_SE.UTF-8"

    @staticmethod
    def test_command_list_joined(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """A list of commands is joined with newlines."""
        processes = fake_popen()

        assert list(mock_cwb.run_cqp(["A;", "B;"])) == []

        assert processes[0].input == b"set PrettyPrint off;\nA;\nB;"

    @staticmethod
    def test_command_encoding(tmp_path: Path, fake_popen: FakePopenFactory) -> None:
        """The command and the output use the configured encoding; empty lines are dropped."""
        cwb = CWB(Path("cqp"), Path("cwb-scan-corpus"), tmp_path, "sv_SE.UTF-8", "latin-1")
        processes = fake_popen(stdout="ä\n\nö\n".encode("latin-1"))

        assert list(cwb.run_cqp("[word='ä'];")) == ["ä", "ö"]

        assert processes[0].input == "set PrettyPrint off;\n[word='ä'];".encode("latin-1")

    @staticmethod
    def test_error_raises_cqp_error(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """Stderr output raises `CQPError`, with whitespace normalized and only the first error kept."""
        fake_popen(stderr=b"CQP Error:  first\n problem\nCQP Error: second\n")

        with pytest.raises(CQPError) as excinfo:
            list(mock_cwb.run_cqp("x;"))

        assert str(excinfo.value) == "first problem"

    @staticmethod
    @pytest.mark.parametrize(
        "message",
        [
            "CQP Error: Attribute x is not defined for corpus Y",
            "CQP Error: cl->range && cl->size > 0",
            "CQP Error: [_zzz=...] is neither a positional/structural attribute nor a label",
            "CQP Error: CL: major error, cannot compose string: invalid UTF8 string passed to cl_string_canonical...",
        ],
    )
    def test_ignored_errors(mock_cwb: CWB, fake_popen: FakePopenFactory, message: str) -> None:
        """Errors listed in `_IGNORED_ERRORS` are not raised and the output is still returned."""
        fake_popen(stdout=b"out\n", stderr=message.encode())

        assert list(mock_cwb.run_cqp("x;")) == ["out"]

    @staticmethod
    def test_registry_errors_ignored(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """Registry errors are not raised, and the first CQP error after them is."""
        registry_errors = b"REGISTRY ERROR (r/x): syntax error\nREGISTRY ERROR (r/x): Error parsing the main Registry\n"
        fake_popen(stdout=b"out\n", stderr=registry_errors)

        assert list(mock_cwb.run_cqp("x;")) == ["out"]

        fake_popen(stderr=registry_errors + b"CQP Error:\n\tCorpus ``X'' is undefined\n")

        with pytest.raises(CQPError) as excinfo:
            list(mock_cwb.run_cqp("x;"))
        assert str(excinfo.value) == "Corpus ``X'' is undefined"

    @staticmethod
    def test_attr_ignore(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """The error "No such attribute" is ignored only with `attr_ignore`."""
        fake_popen(stdout=b"out\n", stderr=b"CQP Error: No such attribute: zzz")

        with pytest.raises(CQPError, match="No such attribute"):
            list(mock_cwb.run_cqp("x;"))
        assert list(mock_cwb.run_cqp("x;", attr_ignore=True)) == ["out"]

    @staticmethod
    @pytest.mark.parametrize("checks_before_abort", [0, 2])
    def test_abort(
        mock_cwb: CWB,
        fake_popen: FakePopenFactory,
        monkeypatch: pytest.MonkeyPatch,
        checks_before_abort: int,
    ) -> None:
        """If the abort signal is set while CQP runs, CQP and its children are killed and nothing is yielded.

        The signal is found set after `checks_before_abort` checks in which it is not set.
        """
        # The first `communicate` precedes the first check, so every call before the abort times out
        processes = fake_popen(stdout=b"out\n", timeouts=checks_before_abort + 1)
        children = [Mock(), Mock()]
        # `psutil.Process(pid).children(recursive=True)` returns `children`
        process_class = Mock(return_value=Mock(children=Mock(return_value=children)))
        monkeypatch.setattr("korp.cwb.psutil.Process", process_class)
        is_set = iter([False] * checks_before_abort + [True])
        abort_signal = cast(AbortSignal, SimpleNamespace(is_set=lambda: next(is_set)))

        assert list(mock_cwb.run_cqp("x;", abort_signal=abort_signal)) == []
        process = processes[0]
        assert process.killed
        process_class.assert_called_once_with(process.pid)
        process_class.return_value.children.assert_called_once_with(recursive=True)
        for child in children:
            child.kill.assert_called_once_with()

    @staticmethod
    @pytest.mark.parametrize("is_set", [None, False], ids=["no_signal", "signal_not_set"])
    def test_timeout_without_abort(
        mock_cwb: CWB, fake_popen: FakePopenFactory, monkeypatch: pytest.MonkeyPatch, is_set: bool | None
    ) -> None:
        """If CQP times out but the abort signal is not set, waiting continues and the output is yielded."""
        # More than one timeout, so that `communicate` is retried; resending the input would raise `ValueError`
        processes = fake_popen(stdout=b"out\n", timeouts=3)
        process_class = Mock()
        monkeypatch.setattr("korp.cwb.psutil.Process", process_class)
        abort_signal = None if is_set is None else cast(AbortSignal, SimpleNamespace(is_set=lambda: is_set))

        assert list(mock_cwb.run_cqp("x;", abort_signal=abort_signal)) == ["out"]
        process = processes[0]
        assert not process.killed
        assert process.input == b"set PrettyPrint off;\nx;"
        process_class.assert_not_called()

    @staticmethod
    def test_error_whitespace_stripped(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """The newline that ends the CQP error message is not left as a trailing space."""
        fake_popen(stderr=b"CQP Error:\n\tNo such attribute: zzz\n")

        with pytest.raises(CQPError) as excinfo:
            list(mock_cwb.run_cqp("x;"))

        assert str(excinfo.value) == "No such attribute: zzz"

    @staticmethod
    def test_error_invalid_encoding(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """Stderr invalid in the configured encoding raises `CQPError`, with invalid bytes replaced by U+FFFD.

        CQP echoes only the first byte of a multibyte character outside a string in a syntax error.
        """
        fake_popen(stderr=b"CQP Error:\n\tCQP Syntax Error: syntax error\n\t\n[word='x'] \xc3 <--\n")

        with pytest.raises(CQPError) as excinfo:
            list(mock_cwb.run_cqp("x;"))

        assert str(excinfo.value) == "CQP Syntax Error: syntax error [word='x'] \ufffd <--"

    @staticmethod
    def test_ignored_error_invalid_encoding(mock_cwb: CWB, fake_popen: FakePopenFactory) -> None:
        """An ignored error is ignored even if stderr is invalid in the configured encoding."""
        fake_popen(
            stdout=b"out\n",
            stderr=(
                b"CQP Error:\n\t``zzz'' is neither a positional/structural attribute nor a label reference\n"
                b"CQP Error:\n\tCQP Syntax Error: syntax error\n\t\n[zzz='x' & \xc3 <--\n"
            ),
        )

        assert list(mock_cwb.run_cqp("x;")) == ["out"]
