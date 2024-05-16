import os
import psutil
import subprocess
import re

from korp import utils


class CWB:

    def __init__(self):
        self.executable = None
        self.scan_executable = None
        self.registry = None
        self.locale = None
        self.encoding = None

    def init(self, executable, scan_executable, registry, locale, encoding):
        self.executable = executable
        self.scan_executable = scan_executable
        self.registry = registry
        self.locale = locale
        self.encoding = encoding

    def run_cqp(self, command, attr_ignore=False, abort_event=None, errors="strict"):
        """Call the CQP binary with the given command, and the request data.
        Yield one result line at the time, disregarding empty lines.
        If there is an error, raise a CQPError exception, unless the
        parameter errors is "ignore" or "report" (report errors at the
        beginning of the output as lines beginning with "CQP Error:").
        """
        env = os.environ.copy()
        env["LC_COLLATE"] = self.locale
        if not isinstance(command, str):
            command = "\n".join(command)
        command = "set PrettyPrint off;\n" + command
        command = command.encode(self.encoding)
        process = subprocess.Popen([self.executable, "-c", "-r", self.registry],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env)

        # Use a loop and timeout to be able to kill aborted searches
        timeout = 1
        try:
            reply, error = process.communicate(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            while True:
                if abort_event and abort_event.is_set():
                    # Kill cqp process and its children
                    children = psutil.Process(process.pid).children(recursive=True)
                    process.kill()
                    for child in children:
                        child.kill()
                    return
                try:
                    reply, error = process.communicate(timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
                break

        if error and errors != "ignore":
            error = error.decode(self.encoding)
            # Remove newlines from the error string:
            error = re.sub(r"\s+", r" ", error)
            if errors == "report":
                # Each error on its own line beginning with "CQP Error:"
                error = re.sub(r" +(CQP Error: *)", r"\n\1", error)
                for line in error.split("\n"):
                    yield line
            else:
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
                "\n"):  # We don't use splitlines() since it might split on special characters in the data
            if line:
                yield line

    def run_cwb_scan(self, corpus, attrs, abort_event=None):
        """Call the cwb-scan-corpus binary with the given arguments.
        Yield one result line at the time, disregarding empty lines.
        If there is an error, raise a CQPError exception.
        """
        process = subprocess.Popen([self.scan_executable, "-q", "-r", self.registry, corpus] + attrs,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Use a loop and timeout to be able to kill aborted searches
        timeout = 1
        while True:
            if abort_event and abort_event.is_set():
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
            # Ignore certain errors:
            # 1) "show +attr" for unknown attr,
            # 2) querying unknown structural attribute,
            # 3) calculating statistics for empty results
            raise utils.CQPError(error)
        for line in reply.decode(self.encoding, errors="ignore").split(
                "\n"):  # We don't use splitlines() since it might split on special characters in the data
            if line and len(line) < 65536:
                yield line

    @staticmethod
    def show_attributes():
        """Command sequence for returning the corpus attributes."""
        return ["show cd; .EOL.;"]

    @staticmethod
    def read_attributes(lines):
        """Read the CQP output from the show_attributes() command."""
        attrs = {'p': [], 's': [], 'a': []}
        for line in lines:
            if line == utils.END_OF_LINE:
                break
            (typ, name, _rest) = (line + " X").split(None, 2)
            attrs[typ[0]].append(name)
        return attrs


cwb = CWB()
