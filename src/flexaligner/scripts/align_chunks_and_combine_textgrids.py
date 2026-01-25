#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Align existing chunk WAVs with align_ce_w2v2.py (optionally parallel),
merge per-chunk TextGrids into one full TextGrid with NULL gaps,
and optionally export one or more tiers to TSV.

Chunk WAV naming example:
  HeMY_15F_di_target.chunk001_21.241-33.041.wav
i.e., contains start/end times in seconds in the filename.

This script resolves the correct chunk wav by:
- globbing candidates with --chunk_wav_glob (default: "{chunk_id}_*.wav")
- parsing *_<start>-<end>.wav from filenames
- choosing the best match to TSV start_s/end_s within --time_match_tol_s

TSV export schema:
  utt_id    tier    start    end    dur    label
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import wave
import shutil
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

EPS = 1e-6


# ============================
# TextGrid structures + I/O
# ============================
@dataclass
class Interval:
    xmin: float
    xmax: float
    text: str


@dataclass
class IntervalTier:
    name: str
    xmin: float
    xmax: float
    intervals: List[Interval]


@dataclass
class TextGrid:
    xmin: float
    xmax: float
    tiers: List[IntervalTier]


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s


def parse_textgrid_long(path: str) -> TextGrid:
    """Minimal parser for Praat TextGrid long text format (IntervalTier only)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    def find_value(pattern: str) -> Optional[str]:
        for ln in lines:
            m = re.match(pattern, ln.strip())
            if m:
                return m.group(1)
        return None

    xmin_s = find_value(r'xmin\s*=\s*([0-9.eE+-]+)\s*$')
    xmax_s = find_value(r'xmax\s*=\s*([0-9.eE+-]+)\s*$')
    if xmin_s is None or xmax_s is None:
        raise ValueError(f"Failed to parse global xmin/xmax in {path}")
    xmin = float(xmin_s)
    xmax = float(xmax_s)

    tiers: List[IntervalTier] = []
    i = 0
    n = len(lines)

    while i < n:
        if re.match(r'item\s*\[\d+\]\s*:', lines[i].strip()):
            cls = None
            name = None
            txmin = None
            txmax = None
            intervals: List[Interval] = []

            i += 1
            while i < n and not re.match(r'item\s*\[\d+\]\s*:', lines[i].strip()):
                s = lines[i].strip()
                if s.startswith("class"):
                    cls = _strip_quotes(s.split("=", 1)[1])
                elif s.startswith("name"):
                    name = _strip_quotes(s.split("=", 1)[1])
                elif s.startswith("xmin") and txmin is None:
                    txmin = float(s.split("=", 1)[1])
                elif s.startswith("xmax") and txmax is None:
                    txmax = float(s.split("=", 1)[1])
                elif re.match(r'intervals\s*\[\d+\]\s*:', s):
                    ixmin = ixmax = None
                    itext = ""
                    i += 1
                    while i < n:
                        ss = lines[i].strip()
                        if ss.startswith("xmin"):
                            ixmin = float(ss.split("=", 1)[1])
                        elif ss.startswith("xmax"):
                            ixmax = float(ss.split("=", 1)[1])
                        elif ss.startswith("text"):
                            itext = _strip_quotes(ss.split("=", 1)[1])
                        elif re.match(r'(intervals|item)\s*\[', ss):
                            i -= 1
                            break
                        i += 1
                    if ixmin is None or ixmax is None:
                        raise ValueError(f"Bad interval in {path} near line {i}")
                    intervals.append(Interval(ixmin, ixmax, itext))
                i += 1

            if cls == "IntervalTier":
                if name is None or txmin is None or txmax is None:
                    raise ValueError(f"Incomplete tier header in {path}")
                tiers.append(IntervalTier(name=name, xmin=txmin, xmax=txmax, intervals=intervals))
        else:
            i += 1

    return TextGrid(xmin=xmin, xmax=xmax, tiers=tiers)


def write_textgrid_long(tg: TextGrid, out_path: str) -> None:
    """Write Praat TextGrid long text format (IntervalTier only)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('File type = "ooTextFile"\n')
        f.write('Object class = "TextGrid"\n\n')
        f.write(f"xmin = {tg.xmin:.6f}\n")
        f.write(f"xmax = {tg.xmax:.6f}\n")
        f.write("tiers? <exists>\n")
        f.write(f"size = {len(tg.tiers)}\n")
        f.write("item []:\n")
        for ti, tier in enumerate(tg.tiers, start=1):
            f.write(f"    item [{ti}]:\n")
            f.write('        class = "IntervalTier"\n')
            f.write(f'        name = "{tier.name}"\n')
            f.write(f"        xmin = {tier.xmin:.6f}\n")
            f.write(f"        xmax = {tier.xmax:.6f}\n")
            f.write(f"        intervals: size = {len(tier.intervals)}\n")
            for ii, itv in enumerate(tier.intervals, start=1):
                f.write(f"        intervals [{ii}]:\n")
                f.write(f"            xmin = {itv.xmin:.6f}\n")
                f.write(f"            xmax = {itv.xmax:.6f}\n")
                f.write(f'            text = "{itv.text}"\n')


def merge_adjacent(intervals: List[Interval], eps: float = 1e-5) -> List[Interval]:
    """Merge adjacent intervals if same label and boundaries touch."""
    out: List[Interval] = []
    for itv in sorted(intervals, key=lambda x: (x.xmin, x.xmax)):
        if not out:
            out.append(Interval(itv.xmin, itv.xmax, itv.text))
            continue
        last = out[-1]
        if last.text == itv.text and abs(last.xmax - itv.xmin) <= eps:
            last.xmax = max(last.xmax, itv.xmax)
        else:
            out.append(Interval(itv.xmin, itv.xmax, itv.text))
    return out


# ============================
# TSV export
# ============================
def textgrid_to_tsv(
    tg: TextGrid,
    tier_name: str,
    utt_id: str,
    out_path: str,
    include_null: bool,
    null_label: str,
) -> None:
    tier = next((t for t in tg.tiers if t.name == tier_name), None)
    if tier is None:
        raise ValueError(f'TSV tier "{tier_name}" not found. Available: {[t.name for t in tg.tiers]}')

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("utt_id\ttier\tstart\tend\tdur\tlabel\n")
        for itv in tier.intervals:
            lab = (itv.text or "").strip()
            if not include_null and lab == null_label:
                continue
            if lab == "":
                continue
            start = float(itv.xmin)
            end = float(itv.xmax)
            dur = max(0.0, end - start)
            if dur <= EPS:
                continue
            f.write(f"{utt_id}\t{tier_name}\t{start:.6f}\t{end:.6f}\t{dur:.6f}\t{lab}\n")


# ============================
# Chunk TSV + WAV utilities
# ============================
def read_chunks_tsv(tsv: str, transcript_col: str) -> List[Tuple[str, float, float, str]]:
    rows: List[Tuple[str, float, float, str]] = []
    with open(tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        cols = reader.fieldnames or []
        required = {"chunk_id", "start_s", "end_s"}
        missing = [c for c in sorted(required) if c not in cols]
        if missing:
            raise ValueError(f"Missing required columns {missing}. Found columns: {cols}")
        if transcript_col not in cols:
            raise ValueError(f'Column "{transcript_col}" not found. Available columns: {cols}')
        for r in reader:
            rows.append((
                r["chunk_id"],
                float(r["start_s"]),
                float(r["end_s"]),
                (r.get(transcript_col) or "").strip(),
            ))
    return sorted(rows, key=lambda x: x[1])


def wav_duration_seconds(wav_path: str) -> float:
    with wave.open(wav_path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# Robust filename time parser:
# matches "..._21.241-33.041.wav" or "..._21.241-33.041.WAV"
_TIME_RE = re.compile(r"_([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)\.wav$", re.IGNORECASE)


def _extract_times_from_chunk_wav(path: str) -> Optional[Tuple[float, float]]:
    base = os.path.basename(path)
    m = _TIME_RE.search(base)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def find_chunk_wav_by_times(
    chunk_id: str,
    start_s: float,
    end_s: float,
    wav_dir: str,
    glob_pat: str,
    tol_s: float,
) -> str:
    """
    Find the chunk wav whose filename-embedded times best match start_s/end_s.

    glob_pat is relative to wav_dir, may include {chunk_id}.
      default: "{chunk_id}_*.wav"
    """
    pattern = os.path.join(wav_dir, glob_pat.format(chunk_id=chunk_id))
    cands = glob.glob(pattern)
    if not cands:
        raise FileNotFoundError(f"{chunk_id}: no wav candidates matched glob: {pattern}")

    scored: List[Tuple[float, str, float, float]] = []
    for p in cands:
        times = _extract_times_from_chunk_wav(p)
        if times is None:
            continue
        s2, e2 = times
        err = abs(s2 - start_s) + abs(e2 - end_s)
        scored.append((err, p, s2, e2))

    if not scored:
        raise FileNotFoundError(
            f"{chunk_id}: candidates found but none matched *_<start>-<end>.wav pattern. "
            f"Example expected suffix: _21.241-33.041.wav. Candidates: {cands[:10]}"
        )

    scored.sort(key=lambda x: x[0])
    best_err, best_path, bs, be = scored[0]

    # Require each boundary within tol OR total error within 2*tol
    if (abs(bs - start_s) > tol_s) or (abs(be - end_s) > tol_s):
        # show a helpful shortlist
        top = "\n".join([f"  err={e:.6f}  {os.path.basename(p)}  (parsed {s:.3f}-{t:.3f})"
                         for e, p, s, t in scored[:8]])
        raise FileNotFoundError(
            f"{chunk_id}: best wav time mismatch exceeds tol={tol_s}s.\n"
            f"TSV: {start_s:.3f}-{end_s:.3f}\n"
            f"Best: {os.path.basename(best_path)} parsed {bs:.3f}-{be:.3f} (err={best_err:.6f})\n"
            f"Top candidates:\n{top}"
        )

    return best_path


# ============================
# Aligner runner (parallel-safe)
# ============================
def _align_one_chunk_worker(
    python_bin: str,
    align_script: str,
    ckpt: str,
    lexicon: str,
    wav: str,
    txt: str,
    out_tg: str,
    out_json: str,
    optional_sil: bool,
    sil_at_ends: bool,
    sil_phone: str,
    sil_cost: float,
    beam: int,
    p_stay: float,
    frame_hop_s: float,
    capture_logs: bool,
) -> Tuple[str, int, str, str]:
    cmd = [
        python_bin, align_script,
        "--ckpt", ckpt,
        "--wav", wav,
        "--lexicon", lexicon,
        "--transcript", txt,
        "--out_textgrid", out_tg,
        "--out_json", out_json,
        "--sil_phone", sil_phone,
        "--sil_cost", str(sil_cost),
        "--beam", str(beam),
        "--p_stay", str(p_stay),
        "--frame_hop_s", str(frame_hop_s),
    ]
    if optional_sil:
        cmd.append("--optional_sil")
    if sil_at_ends:
        cmd.append("--sil_at_ends")

    os.makedirs(os.path.dirname(out_tg) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)

    if capture_logs:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return out_tg, proc.returncode, proc.stdout, proc.stderr
    else:
        proc = subprocess.run(cmd)
        return out_tg, proc.returncode, "", ""


# ============================
# Main
# ============================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--cleanup_work_dir", action="store_true",
                help="Delete --work_dir after successful completion.")
    ap.add_argument("--chunks_tsv", required=True)
    ap.add_argument("--transcript_col", default="words",
                    help='Which TSV column to use as transcript (default: "words").')

    ap.add_argument("--chunk_wav_dir", required=True,
                    help="Directory containing existing chunk wavs.")
    ap.add_argument("--chunk_wav_glob", default="{chunk_id}_*.wav",
                    help='Glob under chunk_wav_dir to find candidates. Default: "{chunk_id}_*.wav"')
    ap.add_argument("--time_match_tol_s", type=float, default=0.02,
                    help="Tolerance (seconds) to match filename times to TSV times. Default: 0.02")

    ap.add_argument("--full_wav", required=True,
                    help="Full wav path used only to get total duration for NULL gaps.")

    ap.add_argument("--work_dir", required=True,
                    help="Work dir for per-chunk txt/tg/json outputs.")

    # aligner args
    ap.add_argument("--align_script", default="align_ce_w2v2.py")
    ap.add_argument("--python_bin", default="python")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lexicon", required=True)

    ap.add_argument("--optional_sil", action="store_true")
    ap.add_argument("--sil_at_ends", action="store_true")
    ap.add_argument("--sil_phone", default="sil")
    ap.add_argument("--sil_cost", type=float, default=-0.5)
    ap.add_argument("--beam", type=int, default=400)
    ap.add_argument("--p_stay", type=float, default=0.92)
    ap.add_argument("--frame_hop_s", type=float, default=0.005)

    # parallel alignment
    ap.add_argument("--nj", type=int, default=1,
                    help="Number of parallel alignment jobs (default: 1).")
    ap.add_argument("--capture_aligner_logs", action="store_true",
                    help="Capture aligner stdout/stderr and show on failure (slower).")

    # merge / output
    ap.add_argument("--null_label", default="NULL")
    ap.add_argument("--out", required=True, help="Output merged TextGrid path.")

    # TSV export
    ap.add_argument("--tsv_out_prefix", default=None,
                    help="If set, write TSV(s) as PREFIX.<tier>.tsv")
    ap.add_argument("--tsv_tiers", nargs="+", default=[],
                    help="One or more tier names to export, e.g. words phones")
    ap.add_argument("--tsv_include_null", action="store_true",
                    help="Include NULL segments in TSV (default: skip NULL).")
    ap.add_argument("--tsv_utt_id", default=None,
                    help="utt_id in TSV. Default: basename of --full_wav without extension.")

    args = ap.parse_args()

    chunks = read_chunks_tsv(args.chunks_tsv, args.transcript_col)
    if not chunks:
        raise RuntimeError("No chunks found in TSV.")
    if args.nj < 1:
        raise ValueError("--nj must be >= 1")

    total_dur = wav_duration_seconds(args.full_wav)

    txt_dir = os.path.join(args.work_dir, "txt")
    tg_dir = os.path.join(args.work_dir, "tg")
    js_dir = os.path.join(args.work_dir, "json")
    os.makedirs(txt_dir, exist_ok=True)
    os.makedirs(tg_dir, exist_ok=True)
    os.makedirs(js_dir, exist_ok=True)

    # ----------------------------
    # 1) Prepare transcripts + build alignment tasks
    # ----------------------------
    tasks = []
    for cid, start, end, transcript in chunks:
        chunk_wav = find_chunk_wav_by_times(
            chunk_id=cid,
            start_s=start,
            end_s=end,
            wav_dir=args.chunk_wav_dir,
            glob_pat=args.chunk_wav_glob,
            tol_s=args.time_match_tol_s,
        )

        chunk_txt = os.path.join(txt_dir, f"{cid}.txt")
        out_tg = os.path.join(tg_dir, f"{cid}.TextGrid")
        out_json = os.path.join(js_dir, f"{cid}.json")

        with open(chunk_txt, "w", encoding="utf-8") as f:
            f.write(transcript + "\n")

        if not os.path.exists(out_tg):
            tasks.append((cid, chunk_wav, chunk_txt, out_tg, out_json))

    # ----------------------------
    # 2) Run alignment (parallel)
    # ----------------------------
    if tasks:
        if args.nj == 1:
            for cid, wavp, txtp, tgp, jsp in tasks:
                out_tg, rc, stdout, stderr = _align_one_chunk_worker(
                    python_bin=args.python_bin,
                    align_script=args.align_script,
                    ckpt=args.ckpt,
                    lexicon=args.lexicon,
                    wav=wavp,
                    txt=txtp,
                    out_tg=tgp,
                    out_json=jsp,
                    optional_sil=args.optional_sil,
                    sil_at_ends=args.sil_at_ends,
                    sil_phone=args.sil_phone,
                    sil_cost=args.sil_cost,
                    beam=args.beam,
                    p_stay=args.p_stay,
                    frame_hop_s=args.frame_hop_s,
                    capture_logs=args.capture_aligner_logs,
                )
                if rc != 0:
                    msg = f"Alignment failed for chunk_id={cid}\nTextGrid: {out_tg}\nWAV: {wavp}\n"
                    if args.capture_aligner_logs:
                        msg += f"\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n"
                    raise RuntimeError(msg)
        else:
            failures: List[Tuple[str, str, str]] = []
            with ProcessPoolExecutor(max_workers=args.nj) as ex:
                futs = []
                for cid, wavp, txtp, tgp, jsp in tasks:
                    futs.append(ex.submit(
                        _align_one_chunk_worker,
                        args.python_bin,
                        args.align_script,
                        args.ckpt,
                        args.lexicon,
                        wavp,
                        txtp,
                        tgp,
                        jsp,
                        args.optional_sil,
                        args.sil_at_ends,
                        args.sil_phone,
                        args.sil_cost,
                        args.beam,
                        args.p_stay,
                        args.frame_hop_s,
                        args.capture_aligner_logs,
                    ))

                for fut in as_completed(futs):
                    out_tg, rc, stdout, stderr = fut.result()
                    if rc != 0:
                        failures.append((out_tg, stdout, stderr))

            if failures:
                lines = ["One or more alignments failed:"]
                for out_tg, stdout, stderr in failures[:10]:
                    lines.append(f"- {out_tg}")
                    if args.capture_aligner_logs:
                        lines.append("  STDOUT:")
                        lines.append(stdout.strip()[:2000])
                        lines.append("  STDERR:")
                        lines.append(stderr.strip()[:2000])
                raise RuntimeError("\n".join(lines))

    # ----------------------------
    # 3) Merge chunk TextGrids -> full TextGrid with NULL gaps
    # ----------------------------
    first_cid = chunks[0][0]
    first_tg_path = os.path.join(tg_dir, f"{first_cid}.TextGrid")
    if not os.path.exists(first_tg_path):
        raise FileNotFoundError(f"Missing first chunk TextGrid: {first_tg_path}")

    first_tg = parse_textgrid_long(first_tg_path)
    if not first_tg.tiers:
        raise RuntimeError(f"No tiers found in {first_tg_path}")
    tier_names = [t.name for t in first_tg.tiers]

    acc: Dict[str, List[Interval]] = {name: [] for name in tier_names}

    def add_gap(gap_start: float, gap_end: float):
        if gap_end - gap_start <= EPS:
            return
        for name in tier_names:
            acc[name].append(Interval(gap_start, gap_end, args.null_label))

    prev_end = 0.0
    for cid, start, end, _ in chunks:
        if start > prev_end + 1e-6:
            add_gap(prev_end, start)

        tg_path = os.path.join(tg_dir, f"{cid}.TextGrid")
        tg = parse_textgrid_long(tg_path)
        tier_map = {t.name: t for t in tg.tiers}

        for name in tier_names:
            if name not in tier_map:
                acc[name].append(Interval(start, end, args.null_label))
                continue
            for itv in tier_map[name].intervals:
                acc[name].append(Interval(itv.xmin + start, itv.xmax + start, itv.text))

        prev_end = max(prev_end, end)

    if prev_end < total_dur - 1e-6:
        add_gap(prev_end, total_dur)

    final_tiers: List[IntervalTier] = []
    for name in tier_names:
        final_tiers.append(IntervalTier(
            name=name,
            xmin=0.0,
            xmax=total_dur,
            intervals=merge_adjacent(acc[name]),
        ))

    combined = TextGrid(xmin=0.0, xmax=total_dur, tiers=final_tiers)
    write_textgrid_long(combined, args.out)
    print(f"[OK] Combined TextGrid written: {args.out}")

    # ----------------------------
    # 4) Optional: TSV export for one or more tiers
    # ----------------------------
    if args.tsv_out_prefix and args.tsv_tiers:
        utt_id = args.tsv_utt_id
        if utt_id is None:
            utt_id = os.path.splitext(os.path.basename(args.full_wav))[0]

        available = {t.name for t in combined.tiers}
        for tier_name in args.tsv_tiers:
            if tier_name not in available:
                raise ValueError(f'TSV tier "{tier_name}" not found. Available: {sorted(available)}')

            out_tsv = f"{args.tsv_out_prefix}.{tier_name}.tsv"
            textgrid_to_tsv(
                tg=combined,
                tier_name=tier_name,
                utt_id=utt_id,
                out_path=out_tsv,
                include_null=args.tsv_include_null,
                null_label=args.null_label,
            )
            print(f"[OK] TSV written: {out_tsv} (tier={tier_name})")
    # ----------------------------
    # 5) Optional: cleanup
    # ----------------------------
    if args.cleanup_work_dir:
        # Safety: refuse to delete suspicious paths
        wd = os.path.abspath(args.work_dir)
        if wd in ("/", os.path.expanduser("~"), os.path.abspath(".")):
            raise ValueError(f"Refusing to delete unsafe work_dir: {wd}")
        shutil.rmtree(wd)
        print(f"[OK] Deleted work_dir: {wd}")

if __name__ == "__main__":
    main()

