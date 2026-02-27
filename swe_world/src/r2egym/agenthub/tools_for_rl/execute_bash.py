#!/root/.venv/bin/python

"""
Description: Execute a bash command in the terminal, with Python version compatibility.

Parameters:
  --command (string, optional): The bash command to execute. For example: --command 'python my_script.py'. If not provided, will show help.
"""

import argparse
import subprocess
import sys
import os
import signal
import selectors
from types import SimpleNamespace

BLOCKED_BASH_COMMANDS = ["git", "ipython", "jupyter", "nohup"]


def _terminate_process(p: subprocess.Popen):
    """Terminate the whole process group (best-effort)."""
    try:
        pgid = os.getpgid(p.pid)
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        try:
            p.terminate()
        except Exception:
            pass


def _kill_process(p: subprocess.Popen):
    """Kill the whole process group (best-effort)."""
    try:
        pgid = os.getpgid(p.pid)
        os.killpg(pgid, signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def run_command_limited(cmd: str, max_output_bytes: int, timeout_s: int = None):
    """
    Run command with bounded memory:
    - stream-read stdout/stderr
    - cap total captured bytes (stdout+stderr)
    - if exceeded, terminate the process to stop spamming output
    """
    # Start a new process group so we can kill children too
    preexec_fn = os.setsid if hasattr(os, "setsid") else None

    p = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,          # bytes mode => easier to cap by bytes
        bufsize=0,
        preexec_fn=preexec_fn,
        executable="/bin/bash",
    )

    sel = selectors.DefaultSelector()
    sel.register(p.stdout, selectors.EVENT_READ, data="stdout")
    sel.register(p.stderr, selectors.EVENT_READ, data="stderr")

    bufs = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = False
    total = 0

    try:
        while sel.get_map():
            if timeout_s is not None:
                # crude timeout check (simple + good enough)
                # If you want precise timeout, track start time.
                pass

            events = sel.select(timeout=0.1)
            if not events:
                if p.poll() is not None:
                    # process ended; continue draining pipes
                    pass
                continue

            for key, _ in events:
                stream_name = key.data
                f = key.fileobj
                chunk = f.read(8192)
                if not chunk:
                    try:
                        sel.unregister(f)
                    except Exception:
                        pass
                    continue

                remaining = max_output_bytes - total
                if remaining > 0:
                    take = chunk[:remaining]
                    bufs[stream_name] += take
                    total += len(take)

                if total >= max_output_bytes:
                    truncated = True
                    _terminate_process(p)
                    try:
                        p.wait(timeout=2)
                    except Exception:
                        _kill_process(p)
                    # stop reading further to enforce strict cap
                    sel.close()
                    break

        # Ensure process is reaped
        if p.poll() is None:
            p.wait()

    finally:
        try:
            sel.close()
        except Exception:
            pass

    def _decode(b: bytes) -> str:
        return b.decode("utf-8", errors="replace")

    stdout = _decode(bytes(bufs["stdout"]))
    stderr = _decode(bytes(bufs["stderr"]))

    if truncated:
        note = f"\n...[TRUNCATED: captured max {max_output_bytes} bytes (stdout+stderr combined)]\n"
        # 把提示放到 stdout 末尾更直观（也可放 stderr）
        stdout += note

    return SimpleNamespace(returncode=p.returncode, stdout=stdout, stderr=stderr, truncated=truncated)


def main():
    parser = argparse.ArgumentParser(description="Execute a bash command.")
    parser.add_argument(
        "command",
        type=str,
        help="The command (and optional arguments) to execute. For example: 'python my_script.py'",
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=int(os.environ.get("BASH_MAX_OUTPUT_BYTES", "16384")),  # 默认 20KB
        help="Max bytes to capture for stdout+stderr combined. Also configurable via BASH_MAX_OUTPUT_BYTES.",
    )
    args = parser.parse_args()

    first_token = args.command.strip().split()[0]
    if first_token in BLOCKED_BASH_COMMANDS:
        print(
            f"Bash command '{first_token}' is not allowed. "
            "Please use a different command or tool."
        )
        sys.exit(1)

    if "git show" in args.command.strip() or "git log" in args.command.strip():
        print(
            "Bash command 'git show' and 'git log' is not allowed. "
            "Please use a different command or tool."
        )
        sys.exit(1)

    result = run_command_limited(args.command, max_output_bytes=args.max_output_bytes)

    if result.returncode != 0:
        print("Error executing command:\n")
        print("[STDOUT]\n")
        print(result.stdout.strip(), "\n")
        print("[STDERR]\n")
        print(result.stderr.strip())
        sys.exit(result.returncode)

    print("[STDOUT]\n")
    print(result.stdout.strip(), "\n")
    print("[STDERR]\n")
    print(result.stderr.strip())


if __name__ == "__main__":
    main()
