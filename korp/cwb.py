"""Module for interfacing with Corpus Workbench."""

import os
import re
import subprocess
from collections.abc import Iterable, Iterator

import psutil

from korp import cqp
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

    def __init__(self, executable: str, scan_executable: str, registry: str, locale: str, encoding: str) -> None:
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
            CQPError: If an error occurs during execution of the command.
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
            return

        reply, error = result
        if error:
            error = error.decode(self.encoding)
            # Normalize whitespace and keep only the first CQP error (the rest are consequences)
            error = re.sub(r"\s+", " ", error)
            error = re.sub(r"^CQP Error: *", "", error)
            error = re.sub(r" *CQP Error:.*$", "", error)

            ignore_error = (attr_ignore and "No such attribute:" in error) or any(
                ignored in error for ignored in self._IGNORED_ERRORS
            )
            if not ignore_error:
                raise cqp.CQPError(error)

        yield from self._iter_lines(reply.decode(self.encoding, errors="ignore"))

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
            error = re.sub(r"\s+", " ", error.decode())
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
