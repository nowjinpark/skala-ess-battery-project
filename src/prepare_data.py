"""Read only the summaries and cycle 10/100 curves needed for DAY 1.

The original MAT files are never modified. No model is trained here.
Run: .venv/bin/python src/prepare_data.py
"""
from pathlib import Path
import argparse
import json
import re
import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
NOMINAL_AH = 1.1
EOL_AH = .8 * NOMINAL_AH
SUMMARY_FIELDS = ["cycle", "QDischarge", "QCharge", "IR", "Tavg", "Tmin", "Tmax", "chargetime"]


def numeric(f, node):
    """Unwrap numeric HDF5 references without materializing entire cells."""
    if isinstance(node, h5py.Reference):
        if not node:
            return np.array([], dtype=float)
        node = f[node]
    if not isinstance(node, h5py.Dataset):
        return np.array([], dtype=float)
    if node.attrs.get("MATLAB_empty", 0):
        return np.array([], dtype=float)
    arr = np.asarray(node[()])
    if h5py.check_dtype(ref=arr.dtype):
        parts = [numeric(f, ref) for ref in arr.ravel() if ref]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(float).ravel()
    return np.array([], dtype=float)


def text_field(f, node):
    if isinstance(node, h5py.Reference):
        node = f[node] if node else None
    if not isinstance(node, h5py.Dataset):
        return ""
    arr = np.asarray(node[()])
    if h5py.check_dtype(ref=arr.dtype) and arr.size == 1:
        return text_field(f, arr.ravel()[0])
    matlab_class = node.attrs.get("MATLAB_class", b"")
    if matlab_class == b"char":
        return "".join(chr(int(x)) for x in arr.ravel() if x).strip()
    if arr.dtype.kind in "SU":
        return "".join(x.decode() if isinstance(x, bytes) else str(x) for x in arr.ravel()).strip()
    return ""  # MATLAB string objects may need a separate documented decoder.


def safe_stat(values, fn=np.mean):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if len(a) else np.nan


def extract_batch(path, batch_name):
    rows, summaries, curves, audits = [], [], {}, []
    with h5py.File(path, "r") as f:
        b = f["batch"]
        n_cells = b["summary"].size
        metadata = {"file": path.name, "batch": batch_name, "n_cells": n_cells,
                    "batch_date": text_field(f, f.get("batch_date")),
                    "top_level_fields": list(f.keys()), "cell_fields": list(b.keys())}
        for i, ref in enumerate(b["summary"][()].ravel()):
            cell_id = f"b{batch_name.split()[-1]}c{i}"
            s = f[ref]
            values = {k: numeric(f, s[k]) if k in s else np.array([]) for k in SUMMARY_FIELDS}
            lengths = {k: len(v) for k, v in values.items()}
            n = lengths["cycle"]
            if not n or any(length != n for length in lengths.values()):
                raise ValueError(f"{cell_id}: inconsistent summary lengths {lengths}")
            df = pd.DataFrame(values)
            df.insert(0, "cell_id", cell_id)
            df.insert(0, "batch", batch_name)
            qd = df.QDischarge.to_numpy()
            cy = df.cycle.to_numpy()
            # Keep all raw values; these flags affect plots/features only.
            df["capacity_plausible"] = np.isfinite(qd) & (qd > .2) & (qd < 1.5)
            df["early_window"] = df.cycle.between(2, 100)
            summaries.append(df)
            def field(k):
                return f[b[k][()].ravel()[i]] if k in b else None
            life = safe_stat(numeric(f, field("cycle_life")))
            policy = text_field(f, field("policy_readable"))
            barcode = text_field(f, field("barcode"))
            channel_text = text_field(f, field("channel_id"))
            channel_node = field("channel_id")
            channel_numeric = (np.array([]) if isinstance(channel_node, h5py.Dataset) and channel_node.attrs.get("MATLAB_class") == b"string" else numeric(f, channel_node))
            channel = channel_text or (str(channel_numeric.tolist()) if len(channel_numeric) else "")
            early = df.loc[df.early_window].copy()
            good_q = early.loc[early.capacity_plausible]
            early_q = good_q.set_index("cycle").QDischarge
            def q_at(c):
                return float(early_q.loc[c]) if c in early_q.index and np.isscalar(early_q.loc[c]) else np.nan
            features = {
                "qd_cycle2": q_at(2), "qd_mean_2_100": safe_stat(good_q.QDischarge),
                "qd_std_2_100": safe_stat(good_q.QDischarge, np.std),
                "qd_change_100_10": q_at(100) - q_at(10),
                "qd_slope_10_100": np.nan,
                "ir_mean_2_100": safe_stat(early.IR.where(early.IR > 0)),
                "ir_change_early": np.nan,
                "tavg_mean_2_100": safe_stat(early.Tavg.where(early.Tavg.between(0, 80))),
                "tmax_mean_2_100": safe_stat(early.Tmax.where(early.Tmax.between(0, 80))),
                "chargetime_mean_2_100": safe_stat(early.chargetime.where(early.chargetime.between(0, 120, inclusive="neither"))),
                "log10_deltaq_var": np.nan, "deltaq_mean": np.nan, "deltaq_min": np.nan,
            }
            fitq = good_q.loc[good_q.cycle.between(10, 100)]
            if len(fitq) >= 70:
                features["qd_slope_10_100"] = float(np.polyfit(fitq.cycle, fitq.QDischarge, 1)[0])
            ir_good = early.loc[early.IR > 0]
            features["ir_change_early"] = (safe_stat(ir_good.loc[ir_good.cycle >= 91, "IR"])
                                            - safe_stat(ir_good.loc[ir_good.cycle <= 10, "IR"]))
            cg = field("cycles")
            voltage = numeric(f, field("Vdlin"))
            curve_reason = ""
            slot10 = slot100 = -1
            if isinstance(cg, h5py.Group) and "Qdlin" in cg:
                refs = cg["Qdlin"][()].ravel()
                # Use recorded cycle values, not a guessed +1 index.
                match10, match100 = np.flatnonzero(cy == 10), np.flatnonzero(cy == 100)
                if len(refs) == n and len(match10) == len(match100) == 1:
                    slot10, slot100 = int(match10[0]), int(match100[0])
                    q10, q100 = numeric(f, refs[slot10]), numeric(f, refs[slot100])
                    if len(q10) == len(q100) == len(voltage) and len(voltage) > 10:
                        delta = q100 - q10
                        if np.isfinite(delta).all() and np.isfinite(voltage).all():
                            variance = float(np.var(delta, ddof=1))
                            if variance > 0:
                                features["log10_deltaq_var"] = float(np.log10(variance))
                            features["deltaq_mean"] = float(np.mean(delta))
                            features["deltaq_min"] = float(np.min(delta))
                            curves[cell_id + "_voltage"] = voltage
                            curves[cell_id + "_q10"] = q10
                            curves[cell_id + "_q100"] = q100
                            curves[cell_id + "_deltaq"] = delta
                        else:
                            curve_reason = "nonfinite_curve"
                    else:
                        curve_reason = "curve_length_mismatch"
                else:
                    curve_reason = "cycle_alignment_unverified"
            else:
                curve_reason = "Qdlin_missing"
            valid_q = df.loc[df.capacity_plausible & (df.cycle >= 2)]
            under = valid_q.loc[valid_q.QDischarge < EOL_AH, "cycle"]
            cvals = re.findall(r"([0-9]+(?:\.[0-9]+)?)C", policy)
            row = {"batch": batch_name, "cell_id": cell_id, "source_index": i,
                   "barcode": barcode, "channel_id": channel, "policy": policy,
                   "cycle_life_raw": life, "n_summary": n,
                   "first_cycle": float(cy[0]), "last_cycle": float(cy[-1]),
                   "last_qd": float(qd[-1]), "tail5_qd": safe_stat(valid_q.QDischarge.tail(5), np.median),
                   "first_below_80": float(under.iloc[0]) if len(under) else np.nan,
                   "n_below_80": len(under), "curve_issue": curve_reason,
                   "curve_slot10_zero_based": slot10, "curve_slot100_zero_based": slot100,
                   "c_rate_stage1": float(cvals[0]) if cvals else np.nan,
                   "c_rate_stage2": float(cvals[1]) if len(cvals) > 1 else np.nan,
                   "n_early_valid_qd": len(good_q), **features}
            rows.append(row)
            audits.append({"batch": batch_name, "cell_id": cell_id,
                           "invalid_capacity_count": int((~df.capacity_plausible).sum()),
                           "zero_capacity_count": int((df.QDischarge == 0).sum()),
                           "missing_values": int(df[SUMMARY_FIELDS].isna().sum().sum()),
                           "duplicate_cycle_count": int(df.cycle.duplicated().sum()),
                           "curve_issue": curve_reason})
        print(f"{batch_name}: {n_cells} cells, {sum(len(x) for x in summaries):,} cycle records", flush=True)
    return pd.DataFrame(rows), pd.concat(summaries, ignore_index=True), curves, pd.DataFrame(audits), metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", choices=["1", "2", "3"], help="Extract one completed download first")
    args = ap.parse_args()
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, filename in FILES.items():
        if args.batch and not name.endswith(args.batch):
            continue
        path = ROOT / "data/raw" / filename
        if not path.exists():
            raise FileNotFoundError(f"Download required: {path}")
        cells, summary, curves, audit, meta = extract_batch(path, name)
        prefix = name.lower().replace(" ", "")
        cells.to_csv(out / f"{prefix}_cells.csv", index=False)
        summary.to_csv(out / f"{prefix}_summary.csv.gz", index=False, compression="gzip")
        audit.to_csv(out / f"{prefix}_quality.csv", index=False)
        np.savez_compressed(out / f"{prefix}_curves.npz", **curves)
        (out / f"{prefix}_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
