#!/usr/bin/env python
"""Measure VoxCPM2 inference resource usage on this host.

Samples (in a background thread, 200ms cadence):
  - GPU utilization % and VRAM used by THIS process (via nvidia-smi compute apps)
  - Process RSS (resident memory)
  - Process CPU %% (across cores; can exceed 100% on multi-core)
Reports peak/avg for each phase: load, warmup, batch.
"""
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
REF_WAV = ROOT.parent / "examples" / "reference_speaker_male.wav"
TEST_SET = json.loads((ROOT / "test_set.json").read_text(encoding="utf-8"))

# Pick a representative subset: short/medium/long across scripts.
BENCH_CODES = ["en", "zh", "ar", "ja", "hi", "ru"]

PID = os.getpid()

def gpu_sample(pid: int) -> tuple[float, float]:
    """Return (gpu_util_pct, vram_mib_for_pid). Uses nvidia-smi pmon for per-process."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        )
        vram = 0.0
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if int(parts[0]) == pid:
                vram = float(parts[1])
                break
        util = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        )
        return float(util.strip()), vram
    except Exception:
        return 0.0, 0.0

def proc_sample() -> tuple[float, float]:
    """Return (cpu_pct_sum_across_cores, rss_mib). Reads /proc/<pid>/stat and statm."""
    try:
        with open(f"/proc/{PID}/statm") as f:
            rss_pages = int(f.read().split()[1])
        rss_mib = rss_pages * (os.sysconf("SC_PAGE_SIZE") / 1024 / 1024)
        return rss_mib
    except Exception:
        return 0.0

class Sampler(threading.Thread):
    def __init__(self, label: str):
        super().__init__(daemon=True)
        self.label = label
        self.stop_evt = threading.Event()
        self.gpu_util = []
        self.vram = []
        self.rss = []
        self.cpu = []

    def run(self):
        last_total = None
        last_proc = None
        page = os.sysconf("SC_CLK_TCK")
        ncpu = os.cpu_count() or 1
        while not self.stop_evt.is_set():
            try:
                util, vram = gpu_sample(PID)
                with open(f"/proc/{PID}/statm") as f:
                    rss_pages = int(f.read().split()[1])
                rss = rss_pages * (os.sysconf("SC_PAGE_SIZE") / 1024 / 1024)
                with open(f"/proc/{PID}/stat") as f:
                    parts = f.read().split()
                # fields 14,15 = utime, stime in clock ticks
                proc_ticks = int(parts[13]) + int(parts[14])
                with open("/proc/stat") as f:
                    total_line = f.readline().split()
                total_ticks = sum(int(x) for x in total_line[1:8])
                if last_total is not None:
                    dt = (total_ticks - last_total) / ncpu
                    if dt > 0:
                        cpu_pct = 100.0 * (proc_ticks - last_proc) / dt
                        self.cpu.append(cpu_pct)
                last_total, last_proc = total_ticks, proc_ticks
                self.gpu_util.append(util)
                self.vram.append(vram)
                self.rss.append(rss)
            except Exception:
                pass
            self.stop_evt.wait(0.2)

    def stop_and_summary(self) -> dict:
        self.stop_evt.set()
        self.join(timeout=2)
        def stats(lst):
            if not lst: return {"peak": 0, "avg": 0, "samples": 0}
            return {"peak": round(max(lst), 1), "avg": round(statistics.mean(lst), 1), "samples": len(lst)}
        return {
            "phase": self.label,
            "gpu_util_pct": stats(self.gpu_util),
            "vram_mib": stats(self.vram),
            "rss_mib": stats(self.rss),
            "cpu_pct": stats(self.cpu),
        }

# ---- run ----
report = {}

print(f"[host] PID={PID}, ncpu={os.cpu_count()}, torch={torch.__version__}, cuda={torch.cuda.get_device_name(0)}")

s = Sampler("model_load")
s.start()
t0 = time.time()
from voxcpm import VoxCPM
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
sr = model.tts_model.sample_rate
load_time = time.time() - t0
torch.cuda.synchronize()
time.sleep(0.5)
report["model_load"] = s.stop_and_summary()
report["model_load"]["seconds"] = round(load_time, 2)
print(f"[load] model in {load_time:.1f}s, peak VRAM={report['model_load']['vram_mib']['peak']} MiB")

# Warmup (first generate is slow due to JIT)
s = Sampler("warmup")
s.start()
t0 = time.time()
_ = model.generate(text="This is a warm-up generation.", reference_wav_path=str(REF_WAV),
                   cfg_value=2.0, inference_timesteps=10)
torch.cuda.synchronize()
warm_time = time.time() - t0
report["warmup"] = s.stop_and_summary()
report["warmup"]["seconds"] = round(warm_time, 2)
print(f"[warmup] {warm_time:.2f}s, peak VRAM={report['warmup']['vram_mib']['peak']} MiB")

# Batch of representative languages, post-warmup
s = Sampler("batch")
s.start()
batch = []
t_batch = time.time()
for code in BENCH_CODES:
    text = TEST_SET[code]["text"]
    t0 = time.time()
    wav = model.generate(text=text, reference_wav_path=str(REF_WAV),
                         cfg_value=2.0, inference_timesteps=10)
    torch.cuda.synchronize()
    dt = time.time() - t0
    dur = len(wav) / sr
    rtf = dt / dur
    batch.append({"code": code, "duration_s": round(dur, 2), "gen_s": round(dt, 2), "rtf": round(rtf, 3)})
    print(f"  {code}: {dt:.2f}s for {dur:.2f}s audio (RTF={rtf:.2f})")
batch_time = time.time() - t_batch
report["batch"] = s.stop_and_summary()
report["batch"]["seconds"] = round(batch_time, 2)
report["batch"]["per_item"] = batch

# Mean RTF over the warmed-up batch
mean_rtf = statistics.mean(b["rtf"] for b in batch)
report["batch"]["mean_rtf"] = round(mean_rtf, 3)

(ROOT / "perf_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("\n[done] wrote perf_report.json")
print(f"[done] mean post-warmup RTF over {len(batch)} items: {mean_rtf:.3f}")
