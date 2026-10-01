"""Package verified DAY 1 outputs without multi-GB raw data or local runtime."""
from pathlib import Path
import json
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    validation = json.loads((ROOT / "results/validation.json").read_text())
    assert validation["status"] == "passed", "Run DAY1 validation first"
    notebook = json.loads((ROOT / "notebooks/01_DAY1_EDA.ipynb").read_text())
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert code_cells and all(c["execution_count"] is not None for c in code_cells)
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c.get("outputs", []))
    pdfs = list((ROOT / "output/pdf").glob("*.pdf"))
    assert len(pdfs) == 1, "Exactly one final design PDF is expected"
    report_qa = json.loads((ROOT / "results/report_validation.json").read_text())
    assert report_qa["status"] == "passed"
    assert report_qa["sha256"] == hashlib.sha256(pdfs[0].read_bytes()).hexdigest(), "PDF QA must match the latest report"
    files = [ROOT / name for name in ["README.md", "requirements.txt", ".gitignore", "data/downloads/download_manifest.json"]]
    for folder in ["src", "notebooks", "data/processed", "results", "docs", "references", "assets", "output/pdf"]:
        files.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and ".ipynb_checkpoints" not in p.parts)
    target = ROOT / "output/DAY1_분석자료.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(n.startswith(("data/raw/", ".venv/", "tmp/")) for n in archive.namelist())
        print(f"Packaged {len(archive.namelist())} files; {target.stat().st_size:,} bytes: {target}")


if __name__ == "__main__":
    main()
