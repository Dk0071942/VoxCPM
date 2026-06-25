#!/usr/bin/env python
"""Build a self-contained index.html comparing VoxCPM (PyTorch) vs Nano-vLLM-VoxCPM."""
import json
import statistics
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent

def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

vox = load(ROOT / "summary_voxcpm.json")
nv = load(ROOT / "summary_nanovllm.json")
short = load(ROOT / "summary.json")  # original short-sample baseline
perf = load(ROOT / "perf_report.json")
perf_nv = load(ROOT / "perf_report_nanovllm.json")

if vox is None and nv is None and short is None:
    raise SystemExit("no summary files found")

ref_rel = "../examples/reference_speaker_male.wav"

def mean_rtf(items, skip_first=True):
    if not items: return None
    vals = [it["rtf"] for it in items if it.get("rtf") is not None]
    if skip_first and len(vals) > 1:
        vals = vals[1:]
    return round(statistics.mean(vals), 3) if vals else None

def total_dur_gen(items):
    if not items: return (0, 0)
    d = sum((it["duration_s"] or 0) for it in items)
    g = sum((it["gen_time_s"] or 0) for it in items)
    return round(d, 1), round(g, 1)

# Merge VoxCPM and Nano-vLLM summaries by language code
codes = []
seen = set()
for src in (vox or []) + (nv or []):
    if src["code"] not in seen:
        codes.append(src["code"])
        seen.add(src["code"])

def row_for(code, items, out_dir):
    if not items: return ("", "—")
    by_code = {it["code"]: it for it in items}
    it = by_code.get(code)
    if not it or not it.get("file") or it["status"] != "ok":
        return (f'<span class="err">{escape((it or {}).get("status", "missing"))}</span>', "—")
    audio = f'<audio controls preload="none" src="{out_dir}/{it["file"]}"></audio>'
    meta = f'{it["duration_s"]}s · RTF {it["rtf"]}'
    return (audio, meta)

rows = []
text_by_code = {it["code"]: it for it in (vox or nv or [])}
name_by_code = {it["code"]: it["name"] for it in (vox or nv or [])}
text_field = text_by_code

for code in codes:
    txt = escape(text_field.get(code, {}).get("text", ""))
    name = escape(name_by_code.get(code, code))
    vox_audio, vox_meta = row_for(code, vox, "outputs_voxcpm")
    nv_audio, nv_meta = row_for(code, nv, "outputs_nanovllm")
    rows.append(f"""<tr>
<td class="code">{code}</td>
<td class="name">{name}</td>
<td class="text" lang="{code}">{txt}</td>
<td>{vox_audio}<div class="meta">{vox_meta}</div></td>
<td>{nv_audio}<div class="meta">{nv_meta}</div></td>
</tr>""")

vox_d, vox_g = total_dur_gen(vox)
nv_d, nv_g = total_dur_gen(nv)
vox_mean = mean_rtf(vox)
nv_mean = mean_rtf(nv)

perf_table = ""
if perf or perf_nv:
    def cell(p, key, sub, fmt="{:.0f}"):
        if not p: return "—"
        try: return fmt.format(p[key][sub])
        except Exception: return "—"
    perf_table = f"""
<h2>Resource use during steady-state inference</h2>
<table class="perf">
<thead><tr><th>metric</th><th>VoxCPM (PyTorch)</th><th>Nano-vLLM-VoxCPM</th></tr></thead>
<tbody>
<tr><td>GPU util avg / peak</td><td>{cell(perf,'batch','gpu_util_pct.avg') if False else (str(perf['batch']['gpu_util_pct']['avg'])+' % / '+str(perf['batch']['gpu_util_pct']['peak'])+' %' if perf else '—')}</td><td>{(str(perf_nv['batch']['gpu_util_pct']['avg'])+' % / '+str(perf_nv['batch']['gpu_util_pct']['peak'])+' %' if perf_nv else '—')}</td></tr>
<tr><td>VRAM peak</td><td>{(str(perf['batch']['vram_mib']['peak'])+' MiB' if perf else '—')}</td><td>{(str(perf_nv['batch']['vram_mib']['peak'])+' MiB' if perf_nv else '—')}</td></tr>
<tr><td>RSS RAM peak</td><td>{(str(perf['batch']['rss_mib']['peak'])+' MiB' if perf else '—')}</td><td>{(str(perf_nv['batch']['rss_mib']['peak'])+' MiB' if perf_nv else '—')}</td></tr>
<tr><td>CPU avg / peak (sum across cores)</td><td>{(str(perf['batch']['cpu_pct']['avg'])+' % / '+str(perf['batch']['cpu_pct']['peak'])+' %' if perf else '—')}</td><td>{(str(perf_nv['batch']['cpu_pct']['avg'])+' % / '+str(perf_nv['batch']['cpu_pct']['peak'])+' %' if perf_nv else '—')}</td></tr>
<tr><td>Cold model load (s)</td><td>{(str(perf['model_load']['seconds']) if perf else '—')}</td><td>{(str(perf_nv['model_load']['seconds']) if perf_nv else '—')}</td></tr>
</tbody>
</table>
"""

html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VoxCPM2 multilingual voice-cloning — engine comparison</title>
<style>
  :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
  body {{ max-width: 1400px; margin: 2rem auto; padding: 0 1rem; line-height: 1.4; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .sub {{ color: #666; margin-bottom: 1.5rem; }}
  .ref {{ background: #f3f3f3; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1.5rem; }}
  @media (prefers-color-scheme: dark) {{ .ref {{ background: #222; }} }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.6rem; border-bottom: 1px solid #ddd; vertical-align: top; }}
  @media (prefers-color-scheme: dark) {{ th, td {{ border-bottom-color: #333; }} }}
  th {{ font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em; }}
  td.code {{ font-family: ui-monospace, monospace; color: #888; }}
  td.name {{ font-weight: 600; white-space: nowrap; }}
  td.text {{ max-width: 32ch; font-size: 0.9rem; }}
  .meta {{ font-family: ui-monospace, monospace; font-size: 0.75rem; color: #777; margin-top: 0.2rem; }}
  audio {{ width: 240px; height: 36px; }}
  .err {{ color: #c33; font-family: ui-monospace, monospace; font-size: 0.85rem; }}
  table.summary td, table.summary th {{ text-align: right; }}
  table.summary td:first-child, table.summary th:first-child {{ text-align: left; }}
  table.perf td:nth-child(2), table.perf td:nth-child(3),
  table.perf th:nth-child(2), table.perf th:nth-child(3) {{ text-align: right; font-family: ui-monospace, monospace; }}
</style>
</head>
<body>
<h1>VoxCPM2 — Multilingual voice cloning, engine comparison</h1>
<div class="sub">Same male reference clip, same 30-language paragraph test set, two inference engines.</div>
<div class="ref">
  <strong>Reference speaker:</strong>
  <audio controls preload="none" src="{ref_rel}"></audio>
  <div style="margin-top:.25rem; color:#666; font-size:.9rem">examples/reference_speaker_male.wav</div>
</div>

<h2>Aggregate timings (30 languages × ~28s audio each)</h2>
<table class="summary">
<thead><tr><th>engine</th><th>total audio</th><th>total wall</th><th>mean RTF (warm)</th></tr></thead>
<tbody>
<tr><td>VoxCPM (PyTorch)</td><td>{vox_d} s</td><td>{vox_g} s</td><td>{vox_mean}</td></tr>
<tr><td>Nano-vLLM-VoxCPM</td><td>{nv_d} s</td><td>{nv_g} s</td><td>{nv_mean}</td></tr>
</tbody>
</table>
{perf_table}

<h2>Per-language outputs</h2>
<table>
<thead><tr><th>code</th><th>language</th><th>text</th><th>VoxCPM (PyTorch)</th><th>Nano-vLLM-VoxCPM</th></tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody>
</table>
</body>
</html>
"""

out = ROOT / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}")
