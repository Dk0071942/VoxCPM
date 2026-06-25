#!/usr/bin/env python
"""Generate voice-cloned samples in every supported language using standard VoxCPM2 (PyTorch backend)."""
import json
import time
from pathlib import Path

import soundfile as sf
from voxcpm import VoxCPM

ROOT = Path(__file__).resolve().parent
TEST_SET = ROOT / "test_set.json"
OUT_DIR = ROOT / "outputs_voxcpm"
REF_WAV = ROOT.parent / "examples" / "reference_speaker_male.wav"

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(TEST_SET, "r", encoding="utf-8") as f:
    cases = json.load(f)

print(f"[init] loading VoxCPM2 ({len(cases)} languages to synthesize)")
model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
sr = model.tts_model.sample_rate
print(f"[init] model ready. sample_rate={sr} reference={REF_WAV}")

summary = []
t_total = time.time()
for i, (code, item) in enumerate(cases.items(), 1):
    name, text = item["name"], item["text"]
    out_path = OUT_DIR / f"{code}_{name.lower().replace(' ', '_')}.wav"
    t0 = time.time()
    try:
        wav = model.generate(
            text=text,
            reference_wav_path=str(REF_WAV),
            cfg_value=2.0,
            inference_timesteps=10,
        )
        sf.write(str(out_path), wav, sr)
        dt = time.time() - t0
        dur = len(wav) / sr
        rtf = dt / dur if dur > 0 else 0.0
        status = "ok"
        print(f"[{i:02d}/{len(cases)}] {code:>3} {name:<12} dur={dur:5.2f}s gen={dt:5.2f}s rtf={rtf:.2f}")
        summary.append({"code": code, "name": name, "text": text, "file": out_path.name,
                        "duration_s": round(dur, 2), "gen_time_s": round(dt, 2),
                        "rtf": round(rtf, 2), "status": status})
    except Exception as exc:  # noqa: BLE001
        dt = time.time() - t0
        print(f"[{i:02d}/{len(cases)}] {code:>3} {name:<12} FAILED after {dt:.2f}s: {exc}")
        summary.append({"code": code, "name": name, "text": text, "file": None,
                        "duration_s": None, "gen_time_s": round(dt, 2),
                        "rtf": None, "status": f"error: {exc}"})

with open(ROOT / "summary_voxcpm.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"[done] total {time.time()-t_total:.1f}s, summary written to summary_voxcpm.json")
