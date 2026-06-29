from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path


def stream_output(name: str, text: str) -> dict:
    return {
        "name": name,
        "output_type": "stream",
        "text": text.splitlines(keepends=True),
    }


def error_output(exc: BaseException) -> dict:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return {
        "output_type": "error",
        "ename": type(exc).__name__,
        "evalue": str(exc),
        "traceback": [line.rstrip("\n") for line in tb],
    }


def execute_notebook(path: Path, stop_on_error: bool = True) -> None:
    notebook = json.loads(path.read_text())
    namespace = {
        "__name__": "__notebook__",
        "__file__": str(path.resolve()),
        "NOTEBOOK_PATH": path.resolve(),
    }

    execution_count = 1
    for idx, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        cell["execution_count"] = execution_count
        execution_count += 1

        stdout = io.StringIO()
        stderr = io.StringIO()
        outputs: list[dict] = []
        started = time.time()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{path.name}:cell-{idx}", "exec"), namespace)
        except BaseException as exc:
            if stdout.getvalue():
                outputs.append(stream_output("stdout", stdout.getvalue()))
            if stderr.getvalue():
                outputs.append(stream_output("stderr", stderr.getvalue()))
            outputs.append(error_output(exc))
            cell["outputs"] = outputs
            path.write_text(json.dumps(notebook, indent=1))
            if stop_on_error:
                raise
        else:
            if stdout.getvalue():
                outputs.append(stream_output("stdout", stdout.getvalue()))
            if stderr.getvalue():
                outputs.append(stream_output("stderr", stderr.getvalue()))
            elapsed = time.time() - started
            outputs.append(stream_output("stdout", f"[cell {idx} completed in {elapsed:.2f}s]\n"))
            cell["outputs"] = outputs

    notebook.setdefault("metadata", {})
    notebook["metadata"].setdefault("language_info", {"name": "python"})
    path.write_text(json.dumps(notebook, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute code cells in a notebook without nbconvert.")
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
    execute_notebook(args.notebook.resolve(), stop_on_error=not args.keep_going)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        print(f"Notebook execution failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
