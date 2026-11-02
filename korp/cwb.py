"""Module for interfacing with Corpus Workbench."""

import os
import re
import subprocess
from collections.abc import Iterable, Iterator
from pathlib import Path

import psutil

from korp import cqp, utils
from korp.dependencies import AbortSignal


class CWB:
    """Class for interfacing with the Corpus Workbench (CWB) command-line tools."""

    ABORT_TIMEOUT = 1  # seconds
    MAX_LINE_LENGTH = 65536  # Maximum length of a line from cwb-scan-corpus output

    # Error substrings to ignore in CQP output
    _IGNORED_ERRORS = (
        "is not defined for corpus",
        "cl->range && cl->size > 0",
        "neither a positional/structural attribute",
        "CL: major error, cannot compose string: invalid UTF8 string passed to cl_string_canonical...",
    )

    # A line of the errors that CQP prints at startup, for every command, for each registry file it cannot parse
    _REGISTRY_ERROR_LINE = re.compile(r"^REGISTRY ERROR \(.*(?:\n|$)", re.MULTILINE)

    # CQP error message for selecting a corpus that is not in the registry or whose data cannot be read; match only
    # the beginning of a message, as other error output that CQP prints before the next error is appended to it
    _UNDEFINED_CORPUS_MESSAGE = re.compile(r"Corpus ``(.+?)'' is undefined")

    # A (normalized) corpus id that CQP can select, unless it is one of `_CQP_KEYWORDS`
    _VALID_CORPUS_ID = re.compile(r"[a-z_][a-z0-9_-]*")

    # Ids that match `_VALID_CORPUS_ID` but whose uppercase form is a keyword in CQP (cqp/parser.l), so that selecting
    # them is a syntax error
    _CQP_KEYWORDS = frozenset({"mu", "null", "tab"})

    def __init__(self, executable: Path, scan_executable: Path, registry: Path, locale: str, encoding: str) -> None:
        """Initialize CWB interface.

        Args:
            executable: Path to the CQP binary.
            scan_executable: Path to the cwb-scan-corpus binary.
            registry: Path to the corpus registry directory.
            locale: Locale setting for collation.
            encoding: Character encoding for CQP communication.
        """
        self.executable = executable
        self.scan_executable = scan_executable
        self.registry = registry
        self.locale = locale
        self.encoding = encoding

    def _communicate_with_abort(
        self,
        process: subprocess.Popen,
        input_data: bytes | None = None,
        abort_signal: AbortSignal | None = None,
    ) -> tuple[bytes, bytes] | None:
        """Communicate with a subprocess, periodically checking for abort signals.

        Args:
            process: The subprocess to communicate with.
            input_data: Optional data to send to the process's stdin.
            abort_signal: An optional abort event to stop the process.

        Returns:
            A tuple of (stdout, stderr) bytes, or None if the process was aborted.
        """
        try:
            return process.communicate(input_data, timeout=self.ABORT_TIMEOUT)
        except subprocess.TimeoutExpired:
            pass
        while True:
            if abort_signal and abort_signal.is_set():
                children = psutil.Process(process.pid).children(recursive=True)
                process.kill()
                for child in children:
                    child.kill()
                return None
            try:
                return process.communicate(timeout=self.ABORT_TIMEOUT)
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _iter_lines(data: str, max_length: int | None = None) -> Iterator[str]:
        r"""Iterate over non-empty lines in data.

        Uses str.split("\n") instead of splitlines() to avoid splitting on special characters in corpus data.

        Args:
            data: The string to split into lines.
            max_length: If set, skip lines longer than this value.

        Yields:
            Non-empty lines from the data.
        """
        for line in data.split("\n"):
            if line and (max_length is None or len(line) < max_length):
                yield line

    @staticmethod
    def _iter_error_messages(error: str) -> Iterator[str]:
        """Iterate over the error messages in CQP error output.

        A message extends to the next `CQP Error:`, so it may also contain other error output, such as a `CL:` message.
        Output before the first `CQP Error:` is a message of its own.

        Args:
            error: The error output of CQP.

        Yields:
            The error messages without the `CQP Error:` prefix, with whitespace, including newlines, normalized to
            single spaces.
        """
        error = re.sub(r"\s+", " ", error).strip()
        for message in re.split(r" *CQP Error: *", error):
            if message:
                yield message

    def _run_cqp_process(
        self, command: str | list[str], abort_signal: AbortSignal | None = None
    ) -> tuple[str, str] | None:
        """Run the cqp binary in child mode (`-c`) with the given command(s), after `set PrettyPrint off;`.

        In child mode, CQP continues after an error, which `find_undefined_corpora` relies on.

        Args:
            command: The CQP command(s) to execute.
            abort_signal: An optional abort event to stop the process.

        Returns:
            The output and the error output of CQP, or None if the process was aborted. The output is decoded ignoring
            invalid bytes, the error output replacing them.
        """
        env = os.environ.copy()
        env["LC_COLLATE"] = self.locale
        if isinstance(command, list):
            command = "\n".join(command)
        command_bytes = f"set PrettyPrint off;\n{command}".encode(self.encoding)

        process = subprocess.Popen(
            [self.executable, "-c", "-r", self.registry],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        result = self._communicate_with_abort(process, command_bytes, abort_signal)
        if result is None:
            return None
        reply, error = result
        return reply.decode(self.encoding, errors="ignore"), error.decode(self.encoding, errors="replace")

    def run_cqp(
        self, command: str | list[str], attr_ignore: bool = False, abort_signal: AbortSignal | None = None
    ) -> Iterator[str]:
        """Call the cqp binary with the given command(s).

        Args:
            command: The CQP command(s) to execute.
            attr_ignore: Whether to ignore attribute-related errors.
            abort_signal: An optional abort event to stop the process.

        Yields:
            Lines of output from the CQP command. Empty lines are ignored.

        Raises:
            CQPError: If an error occurs during execution of the command. Registry errors are ignored.
        """
        result = self._run_cqp_process(command, abort_signal)
        if result is None:
            return

        reply, error = result
        if error:
            # A registry file that CQP cannot parse makes CQP print a registry error on every run, whatever the
            # command. Selecting that corpus is a separate error, so registry errors can be ignored here.
            error = self._REGISTRY_ERROR_LINE.sub("", error)
            # Normalize whitespace and keep only the first CQP error (the rest are consequences)
            error = re.sub(r"\s+", " ", error)
            error = re.sub(r"^CQP Error: *", "", error)
            error = re.sub(r" *CQP Error:.*$", "", error)
            error = error.strip()

            ignore_error = (attr_ignore and "No such attribute:" in error) or any(
                ignored in error for ignored in self._IGNORED_ERRORS
            )
            if error and not ignore_error:
                raise cqp.CQPError(error)

        yield from self._iter_lines(reply)

    def find_undefined_corpora(self, corpora: Iterable[str], select_in_cqp: bool = True) -> list[str]:
        """Return the corpora that are undefined, that is, that cannot be used in CQP.

        Args:
            corpora: The ids of the corpora to check. They are compared after normalizing with
                `utils.normalize_corpus_id`, so they are case-insensitive.
            select_in_cqp: If True, select each corpus in CQP. This also finds corpora that have a registry file but
                whose data CQP cannot read, but it takes longer with many corpora. If False, only check that each
                corpus has a registry file.

        Returns:
            The undefined corpora in `corpora`, as given and in the same order. Ids that CQP cannot select, such as
            `a.b` or `mu`, are always undefined.

        Raises:
            CQPError: If `select_in_cqp` and CQP terminates before it has selected every corpus. The message also
                contains the other CQP errors, except registry errors.
        """
        corpora = list(corpora)
        # CQP does not report an invalid id as undefined (it reports a syntax error or nothing), so it would seem
        # defined: treat it as undefined without passing it to CQP
        normalized = {corpus: utils.normalize_corpus_id(corpus) for corpus in corpora}
        valid = [
            corpus
            for corpus in corpora
            if self._VALID_CORPUS_ID.fullmatch(normalized[corpus]) and normalized[corpus] not in self._CQP_KEYWORDS
        ]
        if not valid:
            return corpora
        if not select_in_cqp:
            # Listing the registry directory once takes about as long as checking a few hundred files one by one, so
            # it is fast enough for a few corpora and faster for many
            with os.scandir(self.registry) as entries:
                registry_files = {entry.name for entry in entries if entry.is_file()}
            defined = {corpus for corpus in valid if normalized[corpus] in registry_files}
        else:
            # In child mode (`-c`), CQP continues after an error, so a single run reports every undefined corpus.
            # CQP reports only the corpora that are undefined, so a corpus that CQP never got to, because it crashed
            # or was killed, would seem defined. Output an end-of-line marker after the corpora to check that CQP
            # selected all of them. Without an abort signal, the result is never None.
            reply, error = self._run_cqp_process(
                [*(f"{normalized[corpus].upper()};" for corpus in valid), ".EOL.;", "exit;"]
            ) or ("", "")
            undefined_ids: set[str] = set()
            other_errors: list[str] = []
            # Registry errors are not related to the corpora selected
            for message in self._iter_error_messages(self._REGISTRY_ERROR_LINE.sub("", error)):
                if match := self._UNDEFINED_CORPUS_MESSAGE.match(message):
                    undefined_ids.add(utils.normalize_corpus_id(match.group(1)))
                else:
                    other_errors.append(message)
            if cqp.END_OF_LINE not in self._iter_lines(reply):
                raise cqp.CQPError(" ".join(["CQP terminated before checking all the corpora.", *other_errors]))
            defined = {corpus for corpus in valid if normalized[corpus] not in undefined_ids}
        return [corpus for corpus in corpora if corpus not in defined]

    def run_cwb_scan(self, corpus: str, attrs: list[str], abort_signal: AbortSignal | None = None) -> Iterator[str]:
        """Call the cwb-scan-corpus binary with the given arguments.

        Args:
            corpus: The corpus to scan.
            attrs: List of attributes to retrieve.
            abort_signal: An optional abort event to stop the process.

        Yields:
            Lines of output from the cwb-scan-corpus command. Empty lines are ignored.

        Raises:
            CQPError: If an error occurs during execution of the command.
        """
        process = subprocess.Popen(
            [self.scan_executable, "-q", "-r", self.registry, corpus, *attrs],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        result = self._communicate_with_abort(process, abort_signal=abort_signal)
        if result is None:
            return

        reply, error = result
        if error:
            error = re.sub(r"\s+", " ", error.decode(self.encoding, errors="replace")).strip()
            raise cqp.CQPError(error)

        yield from self._iter_lines(reply.decode(self.encoding, errors="ignore"), max_length=self.MAX_LINE_LENGTH)

    @staticmethod
    def show_attributes() -> list[str]:
        """Return the CQP command to show corpus attributes."""
        return ["show cd; .EOL.;"]

    @staticmethod
    def read_attributes(lines: Iterable[str]) -> dict[str, list[str]]:
        """Read the CQP output from the show_attributes() command.

        Args:
            lines: Iterable of output lines from CQP.

        Returns:
            A dictionary with keys 'p', 's', and 'a' for positional attributes, structural attributes, and aligned
                corpora, each containing a list of names.
        """
        attrs = {"p": [], "s": [], "a": []}
        for line in lines:
            if line == cqp.END_OF_LINE:
                break
            typ, name, *_ = line.split(None, 2)
            attrs[typ[0]].append(name)
        return attrs
