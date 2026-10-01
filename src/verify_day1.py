"""Check data identity, observation window, splits and generated artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = ROOT / "data/processed"
    cells = pd.read_csv(p / "cells_analysis.csv")
    assignment = pd.read_csv(ROOT / "results/split_assignment.csv")
    assert cells.cell_id.is_unique, "One row per battery is required"
    assert cells.cell_id.tolist() == assignment.cell_id.tolist()
    assert set(cells.batch) == {"Batch 1", "Batch 2", "Batch 3"}, "DAY1 EDA requires all three batches"
    eligible = cells.loc[cells.analysis_eligible]
    assert eligible.cycle_life.gt(100).all()
    assert eligible.cycle_life.le(eligible.n_summary+1).all()
    observed = eligible.loc[eligible.label_status.eq("observed_80pct_crossing")]
    proxy = eligible.loc[eligible.label_status.eq("provided_near80_endpoint_proxy")]
    assert len(observed)+len(proxy) == len(eligible)
    assert observed.cycle_life.sub(observed.first_below_80).abs().le(1).all()
    assert proxy.last_qd.between(.88,.885).all()
    assert proxy.cycle_life.eq(proxy.n_summary+1).all()
    assert eligible.curve_issue.isna().all()
    roles = {r: set(g.cell_id) for r,g in cells.groupby("split_role")}
    names = list(roles)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            assert not roles[a] & roles[b], f"Overlapping cells between {a} and {b}"
    assert cells.loc[cells.split_role.str.startswith("Batch1"), "batch"].eq("Batch 1").all()
    assert cells.loc[cells.split_role.eq("Batch2_final_test"), "batch"].eq("Batch 2").all()
    assert cells.loc[cells.split_role.eq("Batch3_EDA_only"), "batch"].eq("Batch 3").all()
    assert eligible.loc[eligible.batch.eq("Batch 3"), "split_role"].eq("Batch3_EDA_only").all()
    baseline = pd.read_csv(ROOT / "references/baseline_batch12_splits.csv")
    current_b12 = assignment.loc[assignment.batch.isin(["Batch 1", "Batch 2"])]
    fixed_columns = ["batch", "analysis_eligible", "split_role"]
    pd.testing.assert_frame_equal(
        baseline.set_index("cell_id")[fixed_columns].sort_index(),
        current_b12.set_index("cell_id")[fixed_columns].sort_index())
    assert len(roles["Batch1_CV_development"]) == 28
    assert len(roles["Batch1_holdout"]) == 8
    assert len(roles["Batch2_final_test"]) == 39
    recomputed = []
    for batch in (1, 2, 3):
        summary = pd.read_csv(p / f"batch{batch}_summary.csv.gz")
        early = summary.loc[summary.early_window]
        assert early.cycle.between(2, 100).all(), "Future cycles cannot enter features"
        with np.load(p / f"batch{batch}_curves.npz", allow_pickle=False) as curves:
            for r in eligible.loc[eligible.batch.eq(f"Batch {batch}")].itertuples():
                q10, q100 = curves[r.cell_id+"_q10"], curves[r.cell_id+"_q100"]
                delta = curves[r.cell_id+"_deltaq"]
                np.testing.assert_allclose(delta, q100-q10, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(r.log10_deltaq_var, np.log10(np.var(delta, ddof=1)))
                s = summary.loc[summary.cell_id.eq(r.cell_id)].reset_index(drop=True)
                assert s.loc[int(r.curve_slot10_zero_based), "cycle"] == 10
                assert s.loc[int(r.curve_slot100_zero_based), "cycle"] == 100
                recomputed.append(r.cell_id)
    analysis = json.loads((ROOT / "results/analysis.json").read_text())
    assert analysis["model_plan"]["development_n"] == len(roles["Batch1_CV_development"])
    assert analysis["model_plan"]["holdout_n"] == len(roles["Batch1_holdout"])
    assert analysis["model_plan"]["test_n"] == len(roles["Batch2_final_test"])
    assert analysis["model_plan"]["eda_only_n"] == len(roles["Batch3_EDA_only"])
    for name, path in analysis["figures"].items():
        assert (ROOT/path).stat().st_size > 1000, f"Missing figure: {name}"
    output = {"status":"passed", "scope":"DAY1 Batch1+Batch2+Batch3 EDA", "raw_cells":len(cells),
              "eligible_cells":len(eligible),"deltaq_recomputed_cells":len(recomputed),
              "checks":["cell-level uniqueness", "disjoint split IDs", "original Batch1/2 cohort and split unchanged", "Batch3 EDA-only role", "initial cycles <=100",
                        "actual cycle number and HDF5 slot mapping", "Delta Q difference and variance",
                        "separate observed EOL and supplied endpoint-proxy labels", "report counts and plots"],
              "limits":["Cell ID disjointness is not proof of unique physical cells when barcode is unavailable.",
                        "Descriptive EDA includes Batch2 and Batch3 labels; no prediction model was fitted."]}
    (ROOT/"results/validation.json").write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
