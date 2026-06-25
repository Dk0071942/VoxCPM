#!/usr/bin/env python
"""Sample VRAM/RSS/CPU/GPU during Nano-vLLM-VoxCPM inference."""
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REF_WAV = ROOT.parent / "examples" / "reference_speaker_male.wav"
TEST_SET = json.loads((ROOT / "test_set.json").read_text(encoding="utf-8"))
BENCH_CODES = ["en", "zh", "ar", "ja", "hi", "ru"]
MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d"

PID = os.getpid()
NCPU = os.cpu_count() or 1
PAGE = os.sysconf("SC_PAGE_SIZE")


def list_descendant_pids(pid: int) -> list[int]:
    out = {pid}
    try:
        r = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True, timeout=2)
        if r.stdout.strip():
            for c in r.stdout.split():
                cp = int(c)
                out.add(cp)
                # recurse one more level
                rr = subprocess.run(["pgrep", "-P", str(cp)], capture_output=True, text=True, timeout=2)
                if rr.stdout.strip():
                    for cc in rr.stdout.split():
                        out.add(int(cc))
    except Exception:
        pass
    return list(out)


def gpu_sample(pids: list[int]):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        )
        vram = 0.0
        pid_set = set(pids)
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            try:
                if int(parts[0]) in pid_set:
                    vram += float(parts[1])
            except ValueError:
                continue
        util = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=2,
        )
        return float(util.strip()), vram
    except Exception:
        return 0.0, 0.0


def proc_rss_and_cpu(pids: list[int], last_total: int | None, last_proc: int | None):
    rss = 0.0
    proc_ticks = 0
    for p in pids:
        try:
            with open(f"/proc/{p}/statm") as f:
                rss_pages = int(f.read().split()[1])
            rss += rss_pages * (PAGE / 1024 / 1024)
            with open(f"/proc/{p}/stat") as f:
                parts = f.read().split()
            proc_ticks += int(parts[13]) + int(parts[14])
        except FileNotFoundError:
            continue
    with open("/proc/stat") as f:
        total_line = f.readline().split()
    total_ticks = sum(int(x) for x in total_line[1:8])
    cpu_pct = None
    if last_total is not None and last_proc is not None:
        dt = (total_ticks - last_total) / NCPU
        if dt > 0:
            cpu_pct = 100.0 * (proc_ticks - last_proc) / dt
    return rss, cpu_pct, total_ticks, proc_ticks


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
        while not self.stop_evt.is_set():
            try:
                pids = list_descendant_pids(PID)
                util, vram = gpu_sample(pids)
                rss, cpu_pct, last_total, last_proc = proc_rss_and_cpu(pids, last_total, last_proc)
                self.gpu_util.append(util)
                self.vram.append(vram)
                self.rss.append(rss)
                if cpu_pct is not None:
                    self.cpu.append(cpu_pct)
            except Exception:
                pass
            self.stop_evt.wait(0.2)

    def stop_and_summary(self):
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


def main() -> None:
    report = {}
    print(f"[host] PID={PID}, ncpu={NCPU}")

    s = Sampler("model_load")
    s.start()
    t0 = time.time()
    from nanovllm_voxcpm import VoxCPM
    server = VoxCPM.from_pretrained(model=MODEL_PATH, devices=[0])
    load_time = time.time() - t0
    ref_bytes = REF_WAV.read_bytes()
    ref_latents = server.encode_latents(ref_bytes, "wav")
    time.sleep(0.5)
    report["model_load"] = s.stop_and_summary()
    report["model_load"]["seconds"] = round(load_time, 2)
    print(f"[load] engine ready in {load_time:.1f}s, peak VRAM={report['model_load']['vram_mib']['peak']} MiB")

    SR = 48000

    s = Sampler("warmup")
    s.start()
    t0 = time.time()
    _ = list(server.generate(target_text="This is a warm-up generation.", ref_audio_latents=ref_latents, cfg_value=2.0))
    warm_time = time.time() - t0
    report["warmup"] = s.stop_and_summary()
    report["warmup"]["seconds"] = round(warm_time, 2)
    print(f"[warmup] {warm_time:.2f}s, peak VRAM={report['warmup']['vram_mib']['peak']} MiB")

    s = Sampler("batch")
    s.start()
    batch = []
    t_batch = time.time()
    for code in BENCH_CODES:
        text = TEST_SET[code]["text"]
        t0 = time.time()
        chunks = list(server.generate(target_text=text, ref_audio_latents=ref_latents, cfg_value=2.0))
        dt = time.time() - t0
        wav = np.concatenate(chunks) if chunks else np.zeros(0)
        dur = len(wav) / SR
        rtf = dt / dur if dur > 0 else 0.0
        batch.append({"code": code, "duration_s": round(dur, 2), "gen_s": round(dt, 2), "rtf": round(rtf, 3)})
        print(f"  {code}: {dt:.2f}s for {dur:.2f}s audio (RTF={rtf:.2f})")
    batch_time = time.time() - t_batch
    report["batch"] = s.stop_and_summary()
    report["batch"]["seconds"] = round(batch_time, 2)
    report["batch"]["per_item"] = batch
    mean_rtf = statistics.mean(b["rtf"] for b in batch)
    report["batch"]["mean_rtf"] = round(mean_rtf, 3)

    try:
        server.stop()
    except Exception:
        pass

    (ROOT / "perf_report_nanovllm.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] wrote perf_report_nanovllm.json")
    print(f"[done] mean post-warmup RTF over {len(batch)} items: {mean_rtf:.3f}")


if __name__ == "__main__":
    main()
