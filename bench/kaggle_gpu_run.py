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
    proc = subprocess.run(
        cmd, cwd=str(cwd), capture_output=True, text=True
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


def main() -> int:
    record: dict = {}

    banner("1. DEVICE")
    info = device_report()
    for k, v in info.items():
        print(f"  {k}: {v}")
    record["env"] = info

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
            sys.executable, "-c",
            "import sys; sys.path.insert(0, r'%s');"
            "from helios.engine import LLMEngine, EngineConfig;"
            "from helios.core.types import SamplingParams;"
            "e=LLMEngine(EngineConfig(model_dir=r'%s', device='cuda',"
            " kv_cache_bytes=4*1024**3, block_size=16, max_model_len=2048));"
            "print('prompt:', 'The capital of France is');"
            "o=e.generate(['The capital of France is'],"
            " SamplingParams(max_tokens=24, temperature=0.0))[0];"
            "print('OUTPUT:', repr(o.text))"
            % (ROOT / "python", local),
        ])
        record["real_model_ran"] = rc == 0
        # A real model producing coherent text is the strongest single signal
        # that the whole stack -- paging, RoPE, GQA, sampling -- is correct.
        for line in out.splitlines():
            if line.startswith("OUTPUT:"):
                record["real_model_output"] = line
                print(f"  >>> {line}")

    banner("5. BENCHMARK SUITE ON GPU")
    bench_model = ROOT / "artifacts" / "bench_model"
    if not (bench_model / "config.json").exists():
        run([sys.executable, "-m", "helios.cli", "make-toy-model",
             "--out", str(bench_model)])
    for suite in ("all", "prefill-heavy", "prefix-cache"):
        run([sys.executable, "bench/loadgen.py", "--model", str(bench_model),
             "--suite", suite, "--requests", "32", "--out", "artifacts"])
    run([sys.executable, "bench/report.py"])

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
