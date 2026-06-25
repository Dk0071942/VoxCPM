#!/usr/bin/env python
"""Generate voice-cloned samples via Nano-vLLM-VoxCPM inference engine.

The Nano-vLLM-VoxCPM API takes a pre-encoded reference latent (bytes), not a
wav path — we read the wav once, encode latents, then reuse them across all
30 languages.
"""
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
TEST_SET = ROOT / "test_set.json"
OUT_DIR = ROOT / "outputs_nanovllm"
REF_WAV = ROOT.parent / "examples" / "reference_speaker_male.wav"
MODEL_PATH = "/home/ubuntu/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/bffb3df5a29440629464e5e839f4d214c8714c3d"
SR = 48000


def main() -> None:
    from nanovllm_voxcpm import VoxCPM

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEST_SET, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print(f"[init] starting Nano-vLLM-VoxCPM ({len(cases)} languages)")
    server = VoxCPM.from_pretrained(model=MODEL_PATH, devices=[0])

    ref_bytes = REF_WAV.read_bytes()
    ref_latents = server.encode_latents(ref_bytes, "wav")
    print(f"[init] engine ready. encoded ref latent ({len(ref_latents)} bytes) from {REF_WAV.name}")

    summary = []
    t_total = time.time()
    for i, (code, item) in enumerate(cases.items(), 1):
        name, text = item["name"], item["text"]
        out_path = OUT_DIR / f"{code}_{name.lower().replace(' ', '_')}.wav"
        t0 = time.time()
        try:
            chunks = list(server.generate(
                target_text=text,
                ref_audio_latents=ref_latents,
                cfg_value=2.0,
            ))
            wav = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
            sf.write(str(out_path), wav, SR)
            dt = time.time() - t0
            dur = len(wav) / SR
            rtf = dt / dur if dur > 0 else 0.0
            print(f"[{i:02d}/{len(cases)}] {code:>3} {name:<12} dur={dur:5.2f}s gen={dt:5.2f}s rtf={rtf:.2f}")
            summary.append({"code": code, "name": name, "text": text, "file": out_path.name,
                            "duration_s": round(dur, 2), "gen_time_s": round(dt, 2),
                            "rtf": round(rtf, 2), "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            dt = time.time() - t0
            print(f"[{i:02d}/{len(cases)}] {code:>3} {name:<12} FAILED after {dt:.2f}s: {exc}")
            summary.append({"code": code, "name": name, "text": text, "file": None,
                            "duration_s": None, "gen_time_s": round(dt, 2),
                            "rtf": None, "status": f"error: {exc}"})

    try:
        server.stop()
    except Exception:
        pass

    with open(ROOT / "summary_nanovllm.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[done] total {time.time()-t_total:.1f}s, summary written to summary_nanovllm.json")


if __name__ == "__main__":
    main()
