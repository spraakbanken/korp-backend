"""Module for interfacing with Corpus Workbench."""

import os
import re
import subprocess
from collections.abc import Iterable, Iterator

import psutil

from korp import utils


class CWB:
    """Class for interfacing with the Corpus Workbench (CWB) command-line tools."""

    ABORT_TIMEOUT = 1  # seconds
    MAX_LINE_LENGTH = 65536  # Maximum length of a line from cwb-scan-corpus output

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

    def run_cqp(
        self, command: str | list[str], attr_ignore: bool = False, abort_signal: utils.AbortSignal | None = None
    ) -> Iterator[str]:
        """Call the cqp binary with the given command(s).

        Args:
            command: The CQP command(s) to execute.
            attr_ignore: Whether to ignore attribute-related errors.
            abort_signal: An optional abort event to stop the process.

        Yields:
            Lines of output from the CQP command. Empty lines are ignored.

        Raises:
            utils.CQPError: If an error occurs during execution of the command.
        """
        env = os.environ.copy()
        env["LC_COLLATE"] = self.locale
        command_string = "\n".join(command) if not isinstance(command, str) else command

        command_string = "set PrettyPrint off;\n" + command_string
        command_string = command_string.encode(self.encoding)
        process = subprocess.Popen(
            [self.executable, "-c", "-r", self.registry],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        # Use a loop and timeout to be able to kill aborted searches
        try:
            reply, error = process.communicate(command_string, timeout=self.ABORT_TIMEOUT)
        except subprocess.TimeoutExpired:
            while True:
                if abort_signal and abort_signal.is_set():
                    # Kill cqp process and its children
                    children = psutil.Process(process.pid).children(recursive=True)
                    process.kill()
                    for child in children:
                        child.kill()
                    return
                try:
                    reply, error = process.communicate(timeout=self.ABORT_TIMEOUT)
                except subprocess.TimeoutExpired:
                    continue
                break

        if error:
            error = error.decode(self.encoding)
            # Remove newlines from the error string:
            error = re.sub(r"\s+", r" ", error)
            # Keep only the first CQP error (the rest are consequences):
            error = re.sub(r"^CQP Error: *", r"", error)
            error = re.sub(r" *(CQP Error:).*$", r"", error)
            # Ignore certain errors:
            # 1) "show +attr" for unknown attr,
            # 2) querying unknown structural attribute,
            # 3) calculating statistics for empty results
            if (
                not (attr_ignore and "No such attribute:" in error)
                and "is not defined for corpus" not in error
                and "cl->range && cl->size > 0" not in error
                and "neither a positional/structural attribute" not in error
                and "CL: major error, cannot compose string: invalid UTF8 string passed to cl_string_canonical..."
                not in error
            ):
                raise utils.CQPError(error)
        for line in reply.decode(self.encoding, errors="ignore").split(
            "\n"
        ):  # We don't use splitlines() since it might split on special characters in the data
            if line:
                yield line

    def run_cwb_scan(
        self, corpus: str, attrs: list[str], abort_signal: utils.AbortSignal | None = None
    ) -> Iterator[str]:
        """Call the cwb-scan-corpus binary with the given arguments.

        Args:
            corpus: The corpus to scan.
            attrs: List of attributes to retrieve.
            abort_signal: An optional abort event to stop the process.

        Yields:
            Lines of output from the cwb-scan-corpus command. Empty lines are ignored.

        Raises:
            utils.CQPError: If an error occurs during execution of the command.
        """
        process = subprocess.Popen(
            [self.scan_executable, "-q", "-r", self.registry, corpus, *attrs],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Use a loop and timeout to be able to kill aborted searches
        timeout = 1
        while True:
            if abort_signal and abort_signal.is_set():
                process.kill()
                return
            try:
                reply, error = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                continue
            break

        if error:
            # Remove newlines from the error string:
            error = re.sub(r"\s+", r" ", error.decode())
            raise utils.CQPError(error)
        for line in reply.decode(self.encoding, errors="ignore").split(
            "\n"
        ):  # We don't use splitlines() since it might split on special characters in the data
            if line and len(line) < self.MAX_LINE_LENGTH:
                yield line

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
            if line == utils.END_OF_LINE:
                break
            typ, name, *_ = line.split(None, 2)
            attrs[typ[0]].append(name)
        return attrs
