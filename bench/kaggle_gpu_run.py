"""One-shot GPU verification + benchmark run, designed for a Kaggle T4 session.

Usage inside a Kaggle notebook (GPU accelerator ON):

    !git clone -q https://github.com/<you>/helios.git && cd helios && \
        pip install -q triton && python bench/kaggle_gpu_run.py

What it does, in order, stopping at the first failure so a broken rung is never
papered over by a later result:

  1. reports the device, so every number below has provenance
  2. runs the CPU test suite -- the engine must still be correct on GPU-adjacent
     paths before anything is measured
  3. runs the Triton parity gate; on success the kernel is promoted from
     "written" to "verified" and its speedup is recorded
  4. runs the engine end-to-end on a REAL model (Qwen2.5-0.5B) rather than the
     toy, which is what finally makes absolute tokens/second meaningful
  5. runs the ablation suite and writes JSON artifacts

Everything it prints is also written to artifacts/gpu_run.json so the claims can
be traced (spec section 19.3).
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def banner(msg: str) -> None:
    print(f"\n{'=' * 68}\n{msg}\n{'=' * 68}", flush=True)


def run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    # Child processes need `helios` importable. The package lives under python/
    # and is not pip-installed here, so PYTHONPATH is set explicitly -- omitting
    # it silently skipped toy-model creation and cascaded into three failed
    # benchmark suites on the first real T4 run.
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "python") + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True, env=env
    )
    out = proc.stdout + proc.stderr
    print(out[-4000:], flush=True)
    return proc.returncode, out


def device_report() -> dict:
    import torch

    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{cap[0]}.{cap[1]}"
        info["total_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
    try:
        import triton

        info["triton"] = triton.__version__
    except ImportError:
        info["triton"] = None
    try:
        info["commit"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
    except Exception:
        info["commit"] = "unknown"
    return info


def check_staleness() -> dict:
    """Compare the working tree against origin, and say so if it is behind.

    Exists because a stale checkout burned a full GPU session: Kaggle's
    /kaggle/working persists, so `git clone` into an existing directory fails
    with "destination path already exists", the notebook keeps going, and the run
    silently executes last session's code. The failures it then reports are
    already-fixed bugs, which is worse than no run at all -- you debug the past.

    Best-effort: a network failure here must not block a run, so it degrades to
    "unknown" rather than raising.
    """
    out: dict = {"local": "unknown", "remote": "unknown", "behind": 0}
    try:
        out["local"] = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
        subprocess.run(
            ["git", "fetch", "--quiet", "origin"], cwd=str(ROOT),
            capture_output=True, timeout=60,
        )
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
        out["remote"] = (
            subprocess.check_output(
                ["git", "rev-parse", f"origin/{branch}"], cwd=str(ROOT),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
        if out["local"] != out["remote"]:
            count = subprocess.check_output(
                ["git", "rev-list", "--count", f"HEAD..origin/{branch}"],
                cwd=str(ROOT), stderr=subprocess.DEVNULL,
            ).decode().strip()
            out["behind"] = int(count or 0)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    record: dict = {}

    banner("1. DEVICE")
    info = device_report()
    for k, v in info.items():
        print(f"  {k}: {v}")
    record["env"] = info

    stale = check_staleness()
    record["staleness"] = stale
    if stale.get("behind"):
        print(
            "\n"
            "!!! THIS CHECKOUT IS STALE !!!\n"
            f"  local:  {stale['local'][:8]}\n"
            f"  origin: {stale['remote'][:8]}  ({stale['behind']} commit(s) ahead)\n"
            "\n"
            "  Kaggle's /kaggle/working persists between sessions, so a "
            "`git clone` into an existing directory fails silently and you run "
            "OLD code -- including bugs that are already fixed upstream.\n"
            "\n"
            "  Fix it with:\n"
            "      !cd helios && git pull\n"
            "  or start clean:\n"
            "      !rm -rf helios && git clone https://github.com/<you>/helios.git\n"
        )
        return 1

    if not info["cuda_available"]:
        print(
            "\nNo CUDA device. This script is for a GPU session -- on CPU the "
            "Triton gate would skip and the numbers would not mean anything.\n"
            "In Kaggle: Settings -> Accelerator -> GPU T4 x2."
        )
        return 1
    if info["triton"] is None:
        print("\nTriton not installed. Run: pip install -q triton")
        return 1

    banner("2. CPU TEST SUITE (correctness must hold first)")
    rc, _ = run([sys.executable, "-m", "pytest", "tests", "-q",
                 "--ignore=tests/parity/test_triton_parity.py"])
    record["cpu_suite_passed"] = rc == 0
    if rc != 0:
        print("\nEngine tests failed. Fix before measuring anything.")
        return 1

    banner("3. TRITON PARITY GATE (promotes the kernel to 'verified')")
    rc, out = run([sys.executable, "-m", "pytest",
                   "tests/parity/test_triton_parity.py", "-v", "-s"])
    record["triton_verified"] = rc == 0
    for line in out.splitlines():
        if "speedup" in line:
            record["triton_speedup_line"] = line.strip()
            print(f"  >>> {line.strip()}")
    if rc != 0:
        print(
            "\nTriton parity FAILED. The kernel stays 'written, unverified'.\n"
            "Do not claim it. The failure output above is the bug report."
        )

    banner("4. REAL MODEL END-TO-END (makes absolute tok/s meaningful)")
    model_id = "Qwen/Qwen2.5-0.5B"
    local = ROOT / "artifacts" / "qwen05b"
    if not (local / "config.json").exists():
        print(f"  downloading {model_id} ...", flush=True)
        rc, _ = run([
            sys.executable, "-c",
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_id}', local_dir=r'{local}', "
            "allow_patterns=['*.json','*.safetensors','*.txt'])",
        ])
        if rc != 0:
            print("  download failed; skipping the real-model run")
            local = None
    if local and (local / "config.json").exists():
        rc, out = run([
            sys.executable, "bench/real_model_check.py",
            "--model", str(local), "--device", "cuda",
            "--kv-mb", "2048", "--max-tokens", "24", "--max-model-len", "1024",
        ])
        record["real_model_ran"] = rc == 0
        for line in out.splitlines():
            if line.strip().startswith("text:") or "tok/s" in line:
                record.setdefault("real_model_lines", []).append(line.strip())

    banner("5. BENCHMARK SUITE ON GPU")
    bench_model = ROOT / "artifacts" / "bench_model"
    if not (bench_model / "config.json").exists():
        run([sys.executable, "-m", "helios.cli", "make-toy-model",
             "--out", str(bench_model)])
    if not (bench_model / "config.json").exists():
        print("  could not create the benchmark model; skipping step 5")
        record["benchmarks_ran"] = False
        bench_model = None
    if bench_model is not None:
        for suite in ("all", "prefill-heavy", "prefix-cache"):
            run([sys.executable, "bench/loadgen.py", "--model", str(bench_model),
                 "--suite", suite, "--requests", "32", "--out", "artifacts"])
        # The KV-quantization suite needs its own request count: the pool has to
        # bind for the concurrency win to exist at all, and 32 requests in the
        # default pool does not bind (see the note in loadgen.py).
        run([sys.executable, "bench/loadgen.py", "--model", str(bench_model),
             "--suite", "kv-quant", "--requests", "16", "--prompt-len", "64",
             "--output-len", "24", "--out", "artifacts"])
        run([sys.executable, "bench/report.py"])
        record["benchmarks_ran"] = True

    banner("6. THE OPEN QUESTION THIS DEVICE CAN ANSWER")
    print(
        "  alpha for quantization-asymmetric speculation was already measured on\n"
        "  CPU (0.6548, docs/QASSD.md) because it is a property of two weight\n"
        "  matrices, not of hardware. What a GPU adds:\n"
        "\n"
        "    a) does halving the KV cache pay for its dequantization when\n"
        "       bandwidth IS the bottleneck? Compare helios_kv_int8_cramped\n"
        "       against helios_kv_fp_cramped in the artifacts just written --\n"
        "       on CPU int8 was 27% SLOWER despite 2.31x more resident sequences.\n"
        "\n"
        "    b) alpha on a larger model. The 0.5B result clears the 0.6 gate but\n"
        "       misses the 0.78 target; quantization error falls with model size,\n"
        "       so a 7B model should do better. Untested. To run it here:\n"
        "\n"
        "         python bench/measure_alpha.py --model <7b-dir> --device cuda\n"
        "\n"
        "  Neither is claimed anywhere until it is measured (spec section 19.3)."
    )
    if local and (local / "config.json").exists():
        rc, out = run([
            sys.executable, "bench/measure_alpha.py", "--model", str(local),
            "--device", "cuda", "--gamma", "4", "--max-new-tokens", "32",
            "--out", "artifacts/alpha_gpu.json",
        ])
        record["alpha_gpu_ran"] = rc == 0
        for line in out.splitlines():
            if "ALPHA" in line or "GATE" in line:
                record.setdefault("alpha_gpu_lines", []).append(line.strip())

    out_path = ROOT / "artifacts" / "gpu_run.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    banner("SUMMARY")
    for k, v in record.items():
        if k != "env":
            print(f"  {k}: {v}")
    print(f"\nwrote {out_path}")
    print(
        "\nNext: commit artifacts/ and docs/BENCHMARKS.md, then update\n"
        "docs/SCOPE.md -- move the Triton kernel from 'NOT built' to 'Built'\n"
        "ONLY if step 3 passed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
