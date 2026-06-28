#!/usr/bin/env python
"""Run a light predictable-drift experiment from existing outputs when possible."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPONENT_ROOT.parent

sys.path.insert(0, str(COMPONENT_ROOT))

from fts_diffusion.evaluation.drift import block_shuffle_returns  # noqa: E402


def _read_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def _nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _safe_output_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return path
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.name}_{idx}")
        if not candidate.exists() or not any(candidate.iterdir()):
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(f"Could not create a non-overwriting output directory near {path}")


def _sample_candidates(run_dir: Path | None) -> list[Path]:
    patterns = [
        "outputs/**/samples/*.csv",
        "outputs/**/*generated*.csv",
        "outputs/**/*sample*.csv",
    ]
    found: list[Path] = []
    if run_dir and run_dir.exists():
        run_patterns = ["samples/*.csv", "**/*generated*.csv", "**/*sample*.csv"]
        for pattern in run_patterns:
            found.extend(path for path in run_dir.glob(pattern) if path.is_file())
    search_roots = [Path.cwd(), REPO_ROOT]
    for root in search_roots:
        for pattern in patterns:
            found.extend(path for path in root.glob(pattern) if path.is_file())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    if run_dir:
        run_resolved = run_dir.resolve()
        unique.sort(key=lambda p: (not str(p.resolve()).startswith(str(run_resolved)), str(p)))
    else:
        unique.sort(key=str)
    return unique


def _checkpoint_candidates(run_dir: Path | None) -> list[Path]:
    if not run_dir or not run_dir.exists():
        return []
    suffixes = ("*.pt", "*.pth", "*.ckpt", "*.pth.pt", "*.pth.pth")
    found: list[Path] = []
    for suffix in suffixes:
        found.extend(path for path in run_dir.glob(f"**/{suffix}") if path.is_file())
    return sorted(set(found))


def _config_candidates(run_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if run_dir:
        candidates.append(run_dir / "resolved_config.yaml")
    candidates.append(REPO_ROOT / "Config/default.yaml")
    candidates.append(Path("Config/default.yaml"))
    return candidates


def _infer_real_data(run_dir: Path | None) -> tuple[Path | None, Path | None, str | None]:
    for config_path in _config_candidates(run_dir):
        config = _read_yaml(config_path)
        data_path = _nested_get(config, ["data", "path"])
        input_type = _nested_get(config, ["data", "input_type"])
        if data_path:
            path = Path(data_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.exists():
                return path, config_path, input_type
    return None, None, None


def _try_sampling(run_dir: Path, seeds: list[int], length: int) -> list[Path]:
    script = REPO_ROOT / "scripts/sample_fts_diffusion.py"
    if not script.exists():
        return []
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for seed in seeds:
        output_path = sample_dir / f"drift_light_seed_{seed}.csv"
        if output_path.exists():
            created.append(output_path)
            continue
        commands = [
            [
                sys.executable,
                str(script),
                "--run-dir",
                str(run_dir),
                "--output",
                str(output_path),
                "--seed",
                str(seed),
                "--length",
                str(length),
            ],
            [
                sys.executable,
                str(script),
                "--config",
                str(run_dir / "resolved_config.yaml"),
                "--output",
                str(output_path),
                "--seed",
                str(seed),
            ],
        ]
        for command in commands:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode == 0 and output_path.exists():
                created.append(output_path)
                break
    return created


def _write_toy_data(output_dir: Path, seed: int, length: int = 2048) -> tuple[Path, list[Path]]:
    rng = np.random.default_rng(seed)
    iid = rng.normal(0.0, 1.0, size=length)
    ar = np.empty(length, dtype=float)
    noise = rng.normal(0.0, 1.0, size=length)
    ar[0] = noise[0]
    for idx in range(1, length):
        ar[idx] = 0.4 * ar[idx - 1] + noise[idx]
    null = block_shuffle_returns(iid, block_size=21, seed=seed + 100)

    data_dir = output_dir / "toy_inputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    real_path = data_dir / "iid_gaussian_returns.csv"
    ar_path = data_dir / "ar1_phi_0_4_returns.csv"
    null_path = data_dir / "block_shuffled_iid_returns.csv"
    pd.DataFrame({"return": iid}).to_csv(real_path, index=False)
    pd.DataFrame({"return": ar}).to_csv(ar_path, index=False)
    pd.DataFrame({"return": null}).to_csv(null_path, index=False)
    return real_path, [ar_path, null_path]


def _run_evaluator(
    real_csv: Path,
    synthetic_csvs: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
    config: Path | None = None,
    input_type: str | None = None,
    value_column: str | None = None,
    synthetic_input_type: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(COMPONENT_ROOT / "scripts/evaluate_predictable_drift.py"),
        "--real-csv",
        str(real_csv),
        "--synthetic-csv",
        *[str(path) for path in synthetic_csvs],
        "--eval-split",
        "test",
        "--window-size",
        str(args.window_size),
        "--step-size",
        str(args.step_size),
        "--min-window",
        str(args.min_window),
        "--block-size",
        str(args.block_size),
        "--null-reps",
        str(args.null_reps),
        "--seed",
        str(args.seed),
        "--max-lag",
        str(args.max_lag),
        "--output-dir",
        str(output_dir),
    ]
    if args.run_dir:
        command.extend(["--run-dir", str(args.run_dir)])
    if config:
        command.extend(["--config", str(config)])
    if input_type:
        command.extend(["--input-type", input_type])
    if synthetic_input_type:
        command.extend(["--synthetic-input-type", synthetic_input_type])
    if value_column:
        command.extend(["--value-column", value_column])
    return subprocess.run(command, text=True, capture_output=True, check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/drift_smoke"))
    parser.add_argument("--max-synthetic", type=int, default=5)
    parser.add_argument("--null-reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=252)
    parser.add_argument("--step-size", type=int, default=21)
    parser.add_argument("--min-window", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=21)
    parser.add_argument("--max-lag", type=int, default=50)
    parser.add_argument("--sample-length", type=int, default=2048)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = _safe_output_dir(args.output_dir)
    real_path, config_path, input_type = _infer_real_data(args.run_dir)
    samples = _sample_candidates(args.run_dir)
    checkpoints = _checkpoint_candidates(args.run_dir)
    mode = "existing_samples"

    if not samples and checkpoints and args.run_dir:
        mode = "sampled_from_checkpoints"
        samples = _try_sampling(args.run_dir, [args.seed, args.seed + 1, args.seed + 2], args.sample_length)

    if not real_path or not samples:
        mode = "toy_smoke"
        real_path, samples = _write_toy_data(output_dir, seed=args.seed, length=args.sample_length)
        config_path = None
        input_type = "return"
        synthetic_input_type = "return"
        value_column = "return"
    else:
        synthetic_input_type = "return"
        value_column = None

    selected_samples = samples[: args.max_synthetic]
    result = _run_evaluator(
        real_csv=real_path,
        synthetic_csvs=selected_samples,
        output_dir=output_dir,
        args=args,
        config=config_path,
        input_type=input_type,
        value_column=value_column,
        synthetic_input_type=synthetic_input_type,
    )

    metadata = {
        "mode": mode,
        "requested_output_dir": str(args.output_dir),
        "actual_output_dir": str(output_dir),
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "real_data_found": str(real_path) if real_path else None,
        "config_used": str(config_path) if config_path else None,
        "existing_samples_found": [str(path) for path in _sample_candidates(args.run_dir)],
        "checkpoints_found": [str(path) for path in checkpoints],
        "selected_synthetic": [str(path) for path in selected_samples],
        "evaluator_returncode": result.returncode,
        "evaluator_stdout": result.stdout,
        "evaluator_stderr": result.stderr,
    }
    (output_dir / "light_experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if result.returncode != 0:
        failure_report = [
            "# Light Predictable-Drift Experiment",
            "",
            "The evaluator failed; see `light_experiment_metadata.json` for captured stdout/stderr.",
            "",
            f"Mode: `{mode}`",
            f"Return code: `{result.returncode}`",
            "",
            "```text",
            result.stderr,
            "```",
            "",
        ]
        (output_dir / "report.md").write_text("\n".join(failure_report), encoding="utf-8")
        print(result.stdout, end="")
        print(result.stderr, file=sys.stderr, end="")
        return result.returncode

    print(result.stdout, end="")
    print(f"Light experiment mode: {mode}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
