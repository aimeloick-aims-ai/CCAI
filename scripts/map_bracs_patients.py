"""Inspect source BRACS metadata, then build a completely matched patient mapping.

Column names must be explicitly selected after inspection; no guessed patient IDs.
"""
import argparse
from pathlib import Path
import re

import pandas as pd


def slide_key(value):
    value = str(value).strip()
    match = re.fullmatch(r"(?:BRACS_)?(\d+)(?:\.0|\.svs)?", value, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot interpret slide identifier: {value!r}")
    return str(int(match.group(1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--sheet", default="0", help="Sheet name or zero-based index")
    ap.add_argument("--key-column")
    ap.add_argument("--patient-column")
    ap.add_argument("--key-level", choices=["slide", "sample"], default="slide")
    ap.add_argument("--metadata", default="data/graph-bracs/metadata.csv")
    ap.add_argument("--output", default="processed/graph-bracs/patient_mapping.csv")
    args = ap.parse_args()
    source = Path(args.input)
    if source.suffix.lower() in (".xlsx", ".xls"):
        book = pd.ExcelFile(source)
        if args.inspect:
            for name in book.sheet_names:
                frame = pd.read_excel(book, sheet_name=name, dtype=str)
                print(f"Sheet: {name!r}, rows: {len(frame)}, columns: {list(frame.columns)}")
            return
        sheet = int(args.sheet) if args.sheet.isdigit() else args.sheet
        mapping = pd.read_excel(book, sheet_name=sheet, dtype=str)
    else:
        mapping = pd.read_csv(source, dtype=str)
        if args.inspect:
            print(f"Rows: {len(mapping)}, columns: {list(mapping.columns)}")
            return
    if not args.key_column or not args.patient_column:
        ap.error("Inspect first, then supply --key-column and --patient-column explicitly")
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    mapping = mapping[[args.key_column, args.patient_column]].dropna().copy()
    mapping.columns = ["key", "patient_id"]
    if args.key_level == "slide":
        mapping["key"] = mapping.key.map(slide_key)
    else:
        mapping["key"] = mapping.key.str.replace(r"\.(?:png|pt)$", "", regex=True)
    mapping["patient_id"] = mapping.patient_id.str.strip()
    if (mapping.patient_id == "").any() or (mapping.groupby("key").patient_id.nunique() > 1).any():
        raise ValueError("Ambiguous or empty patient assignments")
    mapping = mapping.drop_duplicates("key")
    meta = pd.read_csv(args.metadata)
    meta["key"] = (meta.sample_id.str.extract(r"^BRACS_(\d+)_")[0].map(slide_key)
                   if args.key_level == "slide" else meta.sample_id)
    joined = meta.merge(mapping, on="key", how="left", validate="many_to_one")
    missing = joined.patient_id.isna()
    if missing.any():
        raise ValueError(f"{missing.sum()}/{len(joined)} graph IDs do not match source metadata. "
                         f"Examples: {joined.loc[missing, 'sample_id'].head().tolist()}. No mapping written.")
    out.parent.mkdir(parents=True, exist_ok=True)
    joined[["sample_id", "patient_id"]].to_csv(out, index=False)
    print(f"Complete mapping: {len(joined)} graphs, {joined.patient_id.nunique()} patients -> {out}")


if __name__ == "__main__":
    main()
