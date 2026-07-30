"""
VetXRay Dataset — Validation & summary script
=============================================
Verifies the integrity of a downloaded copy of the VetXRay dataset, prints the
basic statistics, and writes a set of summary figures.

The script assumes **only the released artefacts** are available:

    1. a folder of DICOM images  (*.dcm)
    2. the annotation spreadsheet (*.xlsx)

It performs the following groups of checks:

    A. Spreadsheet structure   — required columns, duplicates, missing values
    B. Controlled vocabularies — species, projection, quality, finding labels
    C. File cross-reference    — every annotated image exists on disk, and
                                 every image on disk is annotated
    D. DICOM headers           — modality, photometric interpretation, bit
                                 depth, image size, pixel spacing
    E. Pixel data              — a random subset is fully decoded and compared
                                 against the header geometry
    F. De-identification       — headers are scanned for identifying tags

Every check is reported as PASS / WARN / FAIL:

    FAIL  the downloaded copy is incomplete or corrupt — missing images,
          unreadable files, image data that contradicts its own header
    WARN  a noteworthy property of the release itself (case-inconsistent
          labels, images without a spreadsheet row, and so on); the data is
          usable, but the caveat should be understood before analysis
    INFO  contextual information, no action implied

The process exit status is 0 when no check FAILs and 1 otherwise, so the
script can be used in automated pipelines.

Dependencies
------------
    pip install pydicom pandas numpy matplotlib openpyxl

Usage
-----
Either edit DICOM_DIR / XLSX_PATH below and run

    python validate.py

or pass the paths on the command line

    python validate.py --dicom-dir /path/to/dicom --xlsx /path/to/annotations.xlsx

A faster, partial run (500 headers instead of the full set):

    python validate.py --dicom-dir ... --xlsx ... --quick
"""

import os
import sys
import glob
import random
import argparse
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:  # pragma: no cover - dependency guard
    sys.exit("pydicom is required:  pip install pydicom pandas numpy matplotlib openpyxl")


# ══════════════════════════════════════════════════════════════════════════════
# 0. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# Path to the folder containing all released .dcm files
DICOM_DIR = None    # TODO: set this to the path where your DICOM files are located

# Path to the released annotation spreadsheet. If left as None the script looks
# for a single .xlsx file sitting next to it.
XLSX_PATH = None    # TODO: set this to the path of the annotation spreadsheet

# Where the summary figures and the validation report are written
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_output")

MAX_FILES    = None   # limit the DICOM header scan (None = scan every file)
PIXEL_SAMPLE = 25     # how many images are fully decoded (pixel-level check)
RANDOM_SEED  = 42
FIG_DPI      = 200


# ══════════════════════════════════════════════════════════════════════════════
# 1. THE RELEASED DATA CONTRACT
# ══════════════════════════════════════════════════════════════════════════════
# Columns the annotation spreadsheet is expected to provide
EXPECTED_COLUMNS = ["FileName", "PatientName", "breed", "specie",
                    "Projection", "Quality", "TAG", "NOTE"]

# Controlled vocabularies (compared case-insensitively)
PROJECTIONS = {"LL", "DV", "VD"}

QUALITY_VALUES = {"correct", "overexposed", "underexposed", "positioning", "exclude"}

# Species carrying enough images for statistical analysis; the release also
# contains a small number of other (exotic) species.
PRIMARY_SPECIES = ("Dog", "Cat")

# TAG is a "|"-separated list drawn from this vocabulary
FINDING_LABELS = {
    "alveolar_pattern", "bronchial_pattern", "cardiomegaly", "costal_fracture",
    "diaphragmatic_hernia", "foreign_body", "fracture", "hernia", "mass",
    "megaesophagus", "pleural_effusion", "pleural_mineralization",
    "pneumoderma", "pneumomediastinum", "pneumothorax", "tracheal_collapse",
    "interstitial_pattern",
}
# Non-pathological TAG tokens
META_LABELS = {"no_finding", "exclude"}

# Expected DICOM header values
EXPECTED_MODALITY    = "CR"
EXPECTED_PHOTOMETRIC = {"MONOCHROME1", "MONOCHROME2"}
# Radiographs below this size on either axis are almost certainly not usable
# images (calibration strips, thumbnails, truncated acquisitions)
MIN_IMAGE_PX = 256

# ── De-identification ─────────────────────────────────────────────────────────
# Header fields naming a person (clinic staff, or the animal's owner). These
# must be empty in a public release; anything left here is reported.
PERSON_TAGS = [
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "PhysiciansOfRecord", "RequestingPhysician", "PatientAddress",
    "OtherPatientIDs", "IssuerOfPatientID",
]
# Site and equipment fields — not personal data, but they identify the clinic
SITE_TAGS = [
    "InstitutionName", "InstitutionAddress", "InstitutionalDepartmentName",
    "StationName", "DeviceSerialNumber",
]
# Animal-level identifiers, retained by design. PatientName holds the animal's
# given name and is part of the published spreadsheet, so it is not listed here.
SUBJECT_TAGS = ["PatientID", "PatientBirthDate", "AccessionNumber"]

IDENTIFYING_TAGS = PERSON_TAGS + SITE_TAGS + SUBJECT_TAGS

# Header fields collected during the scan
HEADER_FIELDS = [
    "Modality", "PhotometricInterpretation", "SamplesPerPixel",
    "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation",
    "Rows", "Columns", "Manufacturer",
]

# ── Plot style ────────────────────────────────────────────────────────────────
PALETTE = {"Dog": "#1F4E9C", "Cat": "#B03A1E", "Other": "#4E6472"}
QUAL_COLORS = {
    "correct":      "#2E7D32",
    "positioning":  "#8A5A08",
    "exclude":      "#B03A1E",
    "underexposed": "#6A3D9A",
    "overexposed":  "#1F4E9C",
}
# Deep tile fills — white text keeps a WCAG-AA contrast ratio of at least 5:1
TILE_COLORS = ["#1F4E9C", "#B03A1E", "#6A3D9A", "#A32E6E",
               "#8A5A08", "#12695E", "#2E7D32", "#4E6472"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
class CheckLog:
    """Collects the outcome of every validation check."""

    PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"

    def __init__(self):
        self.rows = []

    def add(self, status, group, check, detail=""):
        self.rows.append({"Status": status, "Group": group,
                          "Check": check, "Detail": detail})
        print(f"  [{status}] {check}" + (f" — {detail}" if detail else ""))

    def ok(self, group, check, detail=""):
        self.add(self.PASS, group, check, detail)

    def warn(self, group, check, detail=""):
        self.add(self.WARN, group, check, detail)

    def fail(self, group, check, detail=""):
        self.add(self.FAIL, group, check, detail)

    def info(self, group, check, detail=""):
        self.add(self.INFO, group, check, detail)

    def verdict(self, status, group, check, detail=""):
        """Add a PASS or a chosen failure status depending on `status`."""
        self.add(status, group, check, detail)

    @property
    def frame(self):
        return pd.DataFrame(self.rows, columns=["Status", "Group", "Check", "Detail"])

    def counts(self):
        return Counter(r["Status"] for r in self.rows)

    @property
    def n_failed(self):
        return self.counts()[self.FAIL]

    def print_summary(self):
        c = self.counts()
        print("\n" + "═" * 78)
        print("VALIDATION SUMMARY")
        print("═" * 78)
        print(f"  passed:   {c[self.PASS]:>3}")
        print(f"  warnings: {c[self.WARN]:>3}")
        print(f"  failed:   {c[self.FAIL]:>3}")
        for status, label in ((self.FAIL, "FAILED"), (self.WARN, "WARNINGS")):
            items = [r for r in self.rows if r["Status"] == status]
            if items:
                print(f"\n  {label}:")
                for r in items:
                    print(f"    · {r['Check']}" + (f" — {r['Detail']}" if r["Detail"] else ""))
        print()
        if c[self.FAIL]:
            print("  RESULT: dataset did NOT pass validation.")
        elif c[self.WARN]:
            print("  RESULT: dataset passed validation, with warnings (see above).")
        else:
            print("  RESULT: dataset passed validation.")
        print("═" * 78)


def split_tags(tag_str):
    """TAG is a '|'-separated list; returns the individual tokens."""
    if pd.isna(tag_str):
        return []
    return [t.strip() for t in str(tag_str).split("|") if t.strip()]


def print_counts(title, counts, total, top=None, indent="  "):
    """Print a value-count table with absolute counts and percentages."""
    print(f"\n{title}")
    print(f"{indent}{'-' * (len(title))}")
    shown = counts.head(top) if top else counts
    width = max((len(str(k)) for k in shown.index), default=10)
    for key, value in shown.items():
        pct = value / total * 100 if total else 0.0
        print(f"{indent}{str(key):<{width}}  {value:>7,}  ({pct:5.1f}%)")
    if top and len(counts) > top:
        rest = counts.iloc[top:].sum()
        print(f"{indent}{'… other':<{width}}  {rest:>7,}  ({rest / total * 100:5.1f}%)")


def save_fig(fig, name, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOADING
# ══════════════════════════════════════════════════════════════════════════════
def resolve_xlsx(xlsx_path):
    """Return the spreadsheet path, auto-detecting it when not configured."""
    if xlsx_path:
        return xlsx_path
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = sorted(glob.glob(os.path.join(here, "*.xlsx")))
    if len(candidates) == 1:
        print(f"  XLSX_PATH not set — using the spreadsheet found next to the script: "
              f"{os.path.basename(candidates[0])}")
        return candidates[0]
    return None


def load_metadata(xlsx_path):
    """Read the annotation spreadsheet and add normalised helper columns."""
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # Strip stray whitespace from every text column
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip()
    df.replace({"nan": np.nan, "": np.nan}, inplace=True)

    if "FileName" in df:
        df["FileName"] = df["FileName"].astype(str).str.strip()
    if "specie" in df:
        df["species_norm"] = df["specie"].astype(str).str.strip().str.title()
        df.loc[df["specie"].isna(), "species_norm"] = np.nan
    if "Projection" in df:
        df["projection_norm"] = df["Projection"].astype(str).str.strip().str.upper()
    if "Quality" in df:
        df["quality_norm"] = df["Quality"].astype(str).str.strip().str.lower()
    if "breed" in df:
        df["breed_norm"] = df["breed"].astype(str).str.strip().str.title()
    if "TAG" in df:
        df["tags_list"] = df["TAG"].apply(split_tags)
        df["n_findings"] = df["tags_list"].apply(
            lambda lst: sum(1 for t in lst if t not in META_LABELS))
    return df


def list_dicom_files(dicom_dir):
    """All .dcm files below `dicom_dir`, as paths relative to it."""
    found = []
    for root, _dirs, files in os.walk(dicom_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                found.append(os.path.relpath(os.path.join(root, f), dicom_dir))
    return sorted(found)


# ══════════════════════════════════════════════════════════════════════════════
# 4. CHECKS — A. spreadsheet structure
# ══════════════════════════════════════════════════════════════════════════════
def check_spreadsheet(df, log):
    print("\nA. Spreadsheet structure")
    group = "spreadsheet"

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns
                  if c not in EXPECTED_COLUMNS and not c.endswith(("_norm", "_list"))
                  and c != "n_findings"]
    if missing_cols:
        log.fail(group, "required columns present",
                 f"missing: {', '.join(missing_cols)}")
    else:
        log.ok(group, "required columns present", f"{len(EXPECTED_COLUMNS)} columns")
    if extra_cols:
        log.info(group, "additional columns", ", ".join(extra_cols))

    if len(df) == 0:
        log.fail(group, "spreadsheet is not empty", "0 rows")
        return
    log.ok(group, "spreadsheet is not empty", f"{len(df):,} rows")

    if "FileName" not in df.columns:
        return

    empty_names = int(df["FileName"].isna().sum())
    if empty_names:
        log.fail(group, "every row names a file", f"{empty_names} row(s) with no FileName")
    else:
        log.ok(group, "every row names a file")

    dup = df["FileName"].value_counts()
    dup = dup[dup > 1]
    if len(dup):
        log.warn(group, "FileName is unique",
                 f"{len(dup)} filename(s) appear more than once, e.g. {dup.index[0]}")
    else:
        log.ok(group, "FileName is unique")

    wrong_ext = int((~df["FileName"].astype(str).str.lower().str.endswith(".dcm")).sum())
    if wrong_ext:
        log.warn(group, "FileName has a .dcm extension", f"{wrong_ext} row(s) do not")
    else:
        log.ok(group, "FileName has a .dcm extension")

    # Missing values per column (NOTE is free-text and legitimately sparse)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns or col == "NOTE":
            continue
        n_missing = int(df[col].isna().sum())
        if not n_missing:
            continue
        pct = n_missing / len(df) * 100
        log.warn(group, f"'{col}' is fully populated",
                 f"{n_missing:,} missing value(s) ({pct:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CHECKS — B. controlled vocabularies
# ══════════════════════════════════════════════════════════════════════════════
def _case_variants(series):
    """Values that differ only by letter case / whitespace."""
    groups = {}
    for value in series.dropna().unique():
        groups.setdefault(str(value).strip().lower(), set()).add(value)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def check_vocabularies(df, log):
    print("\nB. Controlled vocabularies")
    group = "vocabulary"

    if "projection_norm" in df:
        unknown = sorted(set(df["projection_norm"].dropna()) - PROJECTIONS)
        if unknown:
            log.warn(group, "Projection uses the documented vocabulary",
                     f"unexpected: {', '.join(unknown)}")
        else:
            log.ok(group, "Projection uses the documented vocabulary",
                   f"{', '.join(sorted(PROJECTIONS))}")

    if "quality_norm" in df:
        unknown = sorted(set(df["quality_norm"].dropna()) - QUALITY_VALUES)
        if unknown:
            log.warn(group, "Quality uses the documented vocabulary",
                     f"unexpected: {', '.join(unknown)}")
        else:
            log.ok(group, "Quality uses the documented vocabulary",
                   f"{len(QUALITY_VALUES)} categories")
        variants = _case_variants(df["Quality"])
        if variants:
            example = next(iter(variants.values()))
            log.warn(group, "Quality spelling is consistent",
                     f"case-only duplicates, e.g. {' / '.join(example)}")

    if "TAG" in df:
        tokens = Counter(t for lst in df["tags_list"] for t in lst)
        unknown = sorted(set(tokens) - FINDING_LABELS - META_LABELS)
        if unknown:
            log.warn(group, "TAG uses the documented vocabulary",
                     f"unexpected token(s): {', '.join(unknown[:5])}")
        else:
            log.ok(group, "TAG uses the documented vocabulary",
                   f"{len(tokens)} distinct token(s)")

        empty_tags = int((df["tags_list"].apply(len) == 0).sum())
        if empty_tags:
            log.warn(group, "every row carries at least one TAG",
                     f"{empty_tags} row(s) with an empty TAG")
        else:
            log.ok(group, "every row carries at least one TAG")

        # 'no_finding' is mutually exclusive with any pathological label
        contradictory = df["tags_list"].apply(
            lambda lst: "no_finding" in lst and any(t in FINDING_LABELS for t in lst))
        n_contra = int(contradictory.sum())
        if n_contra:
            combos = Counter("|".join(sorted(lst)) for lst in
                             df.loc[contradictory, "tags_list"])
            log.warn(group, "'no_finding' is never combined with a finding",
                     f"{n_contra} row(s), most often {combos.most_common(1)[0][0]}")
        else:
            log.ok(group, "'no_finding' is never combined with a finding")

    if "species_norm" in df:
        known = set(PRIMARY_SPECIES)
        others = sorted(set(df["species_norm"].dropna()) - known)
        n_other = int(df["species_norm"].isin(others).sum())
        if others:
            log.info(group, "species beyond Dog/Cat are present",
                     f"{n_other} image(s) across {len(others)} label(s)")
        n_primary = int(df["species_norm"].isin(PRIMARY_SPECIES).sum())
        log.ok(group, "Dog/Cat images present", f"{n_primary:,} image(s)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. CHECKS — C. spreadsheet ↔ disk cross-reference
# ══════════════════════════════════════════════════════════════════════════════
def check_file_crossref(df, disk_files, dicom_dir, log):
    print("\nC. File cross-reference")
    group = "files"

    if not disk_files:
        log.fail(group, "DICOM folder contains .dcm files", "none found")
        return {}

    log.ok(group, "DICOM folder contains .dcm files", f"{len(disk_files):,} file(s)")

    disk_basenames = {os.path.basename(p): p for p in disk_files}
    sheet_names = set(df["FileName"].dropna().astype(str))

    missing = sorted(sheet_names - set(disk_basenames))
    if missing:
        log.fail(group, "every annotated image exists on disk",
                 f"{len(missing):,} missing, e.g. {', '.join(missing[:3])}")
    else:
        log.ok(group, "every annotated image exists on disk",
               f"{len(sheet_names):,} file(s) resolved")

    unreferenced = sorted(set(disk_basenames) - sheet_names)
    if unreferenced:
        log.warn(group, "every image on disk is annotated",
                 f"{len(unreferenced):,} file(s) carry no spreadsheet row, "
                 f"e.g. {', '.join(unreferenced[:3])}")
    else:
        log.ok(group, "every image on disk is annotated")

    # Case-only mismatches would silently break on case-sensitive filesystems
    lower_disk = {n.lower(): n for n in disk_basenames}
    case_only = [n for n in missing if n.lower() in lower_disk]
    if case_only:
        log.warn(group, "filename capitalisation matches the spreadsheet",
                 f"{len(case_only)} name(s) differ only by case")

    empty = [n for n, p in disk_basenames.items()
             if os.path.getsize(os.path.join(dicom_dir, p)) == 0]
    if empty:
        log.fail(group, "no zero-byte image files", f"{len(empty)} empty file(s)")
    else:
        log.ok(group, "no zero-byte image files")

    return {"missing": missing, "unreferenced": unreferenced,
            "basenames": disk_basenames}


# ══════════════════════════════════════════════════════════════════════════════
# 7. CHECKS — D. DICOM headers
# ══════════════════════════════════════════════════════════════════════════════
def scan_dicom_headers(dicom_dir, relpaths, log, max_files=None):
    """Read every header once; returns a DataFrame with one row per file."""
    print("\nD. DICOM headers")
    group = "dicom"

    paths = list(relpaths)
    if max_files is not None and len(paths) > max_files:
        random.seed(RANDOM_SEED)
        paths = sorted(random.sample(paths, max_files))
        log.info(group, "header scan is partial",
                 f"{len(paths):,} of {len(relpaths):,} file(s) — pass --max-files 0 for all")

    print(f"  Reading {len(paths):,} DICOM header(s) …")
    records, unreadable = [], []
    for i, rel in enumerate(paths, 1):
        if i % 1000 == 0:
            print(f"    {i:,}/{len(paths):,}")
        full = os.path.join(dicom_dir, rel)
        try:
            ds = pydicom.dcmread(full, stop_before_pixels=True)
        except (InvalidDicomError, Exception) as exc:      # noqa: B014
            unreadable.append((os.path.basename(rel), str(exc)[:60]))
            continue

        rec = {"FileName": os.path.basename(rel),
               "FileSizeMB": os.path.getsize(full) / 1024 ** 2}
        for field in HEADER_FIELDS:
            rec[field] = getattr(ds, field, None)
        try:
            rec["TransferSyntax"] = str(ds.file_meta.TransferSyntaxUID.name)
        except Exception:
            rec["TransferSyntax"] = None

        spacing = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        try:
            rec["PixelSpacing"] = float(spacing[0]) if spacing is not None else None
        except Exception:
            rec["PixelSpacing"] = None

        # De-identification: record which identifying tags still carry a value.
        # For person-naming tags the value itself is kept so the report can show
        # exactly what would be published.
        present, person_values = [], []
        for tag in IDENTIFYING_TAGS:
            value = getattr(ds, tag, None)
            if value in (None, "", " "):
                continue
            present.append(tag)
            if tag in PERSON_TAGS:
                person_values.append(f"{tag}={str(value).strip()}")
        rec["IdentifyingTags"] = "|".join(present)
        rec["PersonTagValues"] = "|".join(person_values)
        records.append(rec)

    if unreadable:
        log.fail(group, "every file is a readable DICOM object",
                 f"{len(unreadable)} unreadable, e.g. {unreadable[0][0]} ({unreadable[0][1]})")
    else:
        log.ok(group, "every file is a readable DICOM object", f"{len(records):,} file(s)")

    dcm = pd.DataFrame(records)
    for col in ("Rows", "Columns", "BitsAllocated", "BitsStored",
                "SamplesPerPixel", "PixelSpacing"):
        if col in dcm:
            dcm[col] = pd.to_numeric(dcm[col], errors="coerce")
    return dcm


def check_dicom_consistency(dcm, log):
    group = "dicom"
    if dcm.empty:
        log.fail(group, "DICOM headers were read", "no headers available")
        return

    modalities = dcm["Modality"].dropna().unique()
    if set(modalities) == {EXPECTED_MODALITY}:
        log.ok(group, "modality is uniform", EXPECTED_MODALITY)
    else:
        log.warn(group, "modality is uniform",
                 f"found: {', '.join(map(str, modalities))}")

    photo = dcm["PhotometricInterpretation"].value_counts()
    unexpected = set(photo.index) - EXPECTED_PHOTOMETRIC
    if unexpected:
        log.warn(group, "photometric interpretation is monochrome",
                 f"unexpected: {', '.join(map(str, unexpected))}")
    else:
        detail = ", ".join(f"{k}: {v:,}" for k, v in photo.items())
        log.ok(group, "photometric interpretation is monochrome", detail)
        if len(photo) > 1:
            log.info(group, "MONOCHROME1 images need inversion when displayed",
                     f"{int(photo.get('MONOCHROME1', 0)):,} image(s) affected")

    spp = dcm["SamplesPerPixel"].dropna().unique()
    if set(spp) <= {1}:
        log.ok(group, "images are single-channel")
    else:
        log.fail(group, "images are single-channel",
                 f"SamplesPerPixel: {', '.join(map(str, spp))}")

    bad_depth = dcm[dcm["BitsStored"] > dcm["BitsAllocated"]]
    if len(bad_depth):
        log.fail(group, "BitsStored fits in BitsAllocated", f"{len(bad_depth)} file(s)")
    else:
        combos = (dcm.groupby(["BitsAllocated", "BitsStored"]).size()
                  .sort_values(ascending=False))
        detail = ", ".join(f"{int(a)}/{int(s)} bit: {n:,}" for (a, s), n in combos.items())
        log.ok(group, "BitsStored fits in BitsAllocated", detail)

    bad_dims = dcm[(dcm["Rows"].isna()) | (dcm["Columns"].isna()) |
                   (dcm["Rows"] <= 0) | (dcm["Columns"] <= 0)]
    if len(bad_dims):
        log.fail(group, "every image declares a valid size", f"{len(bad_dims)} file(s)")
    else:
        log.ok(group, "every image declares a valid size",
               f"{int(dcm['Columns'].min())}×{int(dcm['Rows'].min())} to "
               f"{int(dcm['Columns'].max())}×{int(dcm['Rows'].max())} px")

    tiny = dcm[(dcm["Rows"] < MIN_IMAGE_PX) | (dcm["Columns"] < MIN_IMAGE_PX)]
    if len(tiny):
        example = tiny.iloc[0]
        log.warn(group, f"every image is at least {MIN_IMAGE_PX} px on both axes",
                 f"{len(tiny)} file(s) below it, e.g. {example['FileName']} "
                 f"({int(example['Columns'])}×{int(example['Rows'])} px)")
    else:
        log.ok(group, f"every image is at least {MIN_IMAGE_PX} px on both axes")

    n_no_spacing = int(dcm["PixelSpacing"].isna().sum())
    if n_no_spacing:
        log.warn(group, "pixel spacing is available",
                 f"{n_no_spacing:,} file(s) without PixelSpacing")
    else:
        log.ok(group, "pixel spacing is available",
               f"median {dcm['PixelSpacing'].median():.4f} mm")

    syntaxes = dcm["TransferSyntax"].value_counts()
    log.info(group, "transfer syntaxes",
             ", ".join(f"{k}: {v:,}" for k, v in syntaxes.items()))


# ══════════════════════════════════════════════════════════════════════════════
# 8. CHECKS — E. pixel data
# ══════════════════════════════════════════════════════════════════════════════
def check_pixel_data(dicom_dir, dcm, log, n_sample=PIXEL_SAMPLE):
    print("\nE. Pixel data")
    group = "pixels"
    if dcm.empty:
        log.fail(group, "pixel data is decodable", "no files to sample")
        return pd.DataFrame()

    sample = dcm.sample(min(n_sample, len(dcm)), random_state=RANDOM_SEED)
    print(f"  Decoding {len(sample)} image(s) …")

    failures, mismatches, constant, rows = [], [], [], []
    for rec in sample.itertuples():
        path = os.path.join(dicom_dir, rec.FileName)
        try:
            ds = pydicom.dcmread(path)
            arr = ds.pixel_array
        except Exception as exc:
            failures.append((rec.FileName, str(exc)[:60]))
            continue
        if arr.shape != (int(rec.Rows), int(rec.Columns)):
            mismatches.append(rec.FileName)
        if arr.min() == arr.max():
            constant.append(rec.FileName)
        rows.append({"FileName": rec.FileName, "shape": arr.shape,
                     "dtype": str(arr.dtype), "min": int(arr.min()),
                     "max": int(arr.max())})

    if failures:
        log.fail(group, "pixel data is decodable",
                 f"{len(failures)} of {len(sample)} failed, e.g. {failures[0][0]} ({failures[0][1]})")
    else:
        log.ok(group, "pixel data is decodable", f"{len(sample)} sampled image(s)")

    if mismatches:
        log.fail(group, "pixel array matches the header geometry",
                 f"{len(mismatches)} mismatch(es), e.g. {mismatches[0]}")
    elif rows:
        log.ok(group, "pixel array matches the header geometry")

    if constant:
        log.fail(group, "images are not blank",
                 f"{len(constant)} constant-valued image(s), e.g. {constant[0]}")
    elif rows:
        log.ok(group, "images are not blank")

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 9. CHECKS — F. de-identification
# ══════════════════════════════════════════════════════════════════════════════
def check_anonymisation(dcm, log):
    print("\nF. De-identification")
    group = "privacy"
    if dcm.empty or "IdentifyingTags" not in dcm:
        log.warn(group, "identifying tags were inspected", "no headers available")
        return

    tag_counts = Counter()
    for entry in dcm["IdentifyingTags"].fillna(""):
        for tag in filter(None, entry.split("|")):
            tag_counts[tag] += 1

    # Person-naming tags: these should be empty in a public release
    person = {t: n for t, n in tag_counts.items() if t in PERSON_TAGS}
    if person:
        values = Counter()
        for entry in dcm["PersonTagValues"].fillna(""):
            for item in filter(None, entry.split("|")):
                values[item] += 1
        examples = ", ".join(v for v, _ in values.most_common(4))
        log.warn(group, "no person-naming header tag carries a value",
                 f"{', '.join(f'{t}: {n:,}' for t, n in sorted(person.items()))}"
                 f"  |  {len(values)} distinct value(s), e.g. {examples}")
    else:
        log.ok(group, "no person-naming header tag carries a value",
               f"{len(PERSON_TAGS)} tag(s) inspected")

    site = {t: n for t, n in tag_counts.items() if t in SITE_TAGS}
    if site:
        log.info(group, "site/equipment tags are populated",
                 ", ".join(f"{t}: {n:,}" for t, n in sorted(site.items())))

    subject = {t: n for t, n in tag_counts.items() if t in SUBJECT_TAGS}
    if subject:
        log.info(group, "animal-level identifiers are retained by design",
                 ", ".join(f"{t}: {n:,}" for t, n in sorted(subject.items())))


# ══════════════════════════════════════════════════════════════════════════════
# 10. STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
def print_statistics(df, dcm):
    print("\n" + "═" * 78)
    print("BASIC STATISTICS")
    print("═" * 78)

    total = len(df)
    primary = df[df["species_norm"].isin(PRIMARY_SPECIES)] if "species_norm" in df else df

    print(f"\nImages (all rows):            {total:>8,}")
    print(f"Images (Dog & Cat):           {len(primary):>8,}")
    if "PatientName" in df:
        print(f"Distinct patient names:       {df['PatientName'].nunique():>8,}"
              "   (names are not unique identifiers)")
    if "breed_norm" in df:
        print(f"Distinct breeds:              {df['breed_norm'].nunique():>8,}")
    if "tags_list" in df:
        tokens = Counter(t for lst in df["tags_list"] for t in lst)
        findings = {k: v for k, v in tokens.items() if k in FINDING_LABELS}
        print(f"Distinct finding labels:      {len(findings):>8,}")
        n_path = int((df["n_findings"] > 0).sum())
        print(f"Images with ≥1 finding:       {n_path:>8,}   ({n_path / total * 100:.1f}%)")
        n_healthy = int(df["tags_list"].apply(lambda l: "no_finding" in l).sum())
        print(f"Images labelled 'no_finding': {n_healthy:>8,}   ({n_healthy / total * 100:.1f}%)")

    if "species_norm" in df:
        print_counts("Species", df["species_norm"].value_counts(dropna=False), total, top=6)
    if "projection_norm" in df:
        print_counts("Projection", df["projection_norm"].value_counts(dropna=False), total)
    if "quality_norm" in df:
        print_counts("Quality", df["quality_norm"].value_counts(dropna=False), total)
    if "breed_norm" in df:
        print_counts("Top breeds", df["breed_norm"].value_counts(), total, top=10)
    if "tags_list" in df:
        finding_counts = pd.Series(
            {k: v for k, v in Counter(
                t for lst in df["tags_list"] for t in lst).items() if k in FINDING_LABELS}
        ).sort_values(ascending=False)
        print_counts("Pathological findings", finding_counts, total)
        per_image = df["n_findings"].value_counts().sort_index()
        print_counts("Findings per image", per_image, total)

    if not dcm.empty:
        print("\nDICOM technical properties")
        print("  " + "-" * 26)
        print(f"  Files inspected      : {len(dcm):,}")
        print(f"  Image width  (px)    : min {int(dcm['Columns'].min()):,}  "
              f"median {int(dcm['Columns'].median()):,}  max {int(dcm['Columns'].max()):,}")
        print(f"  Image height (px)    : min {int(dcm['Rows'].min()):,}  "
              f"median {int(dcm['Rows'].median()):,}  max {int(dcm['Rows'].max()):,}")
        if dcm["PixelSpacing"].notna().any():
            print(f"  Pixel spacing (mm)   : median {dcm['PixelSpacing'].median():.4f}")
        print(f"  Total size on disk   : {dcm['FileSizeMB'].sum() / 1024:.1f} GB "
              f"({len(dcm):,} files)")
        for field in ("PhotometricInterpretation", "BitsStored", "Manufacturer"):
            counts = dcm[field].value_counts(dropna=False)
            print_counts(f"  {field}", counts, len(dcm), top=5, indent="    ")


# ══════════════════════════════════════════════════════════════════════════════
# 11. FIGURES
# ══════════════════════════════════════════════════════════════════════════════
def figure_overview(df, dcm):
    """At-a-glance tiles summarising the release."""
    total = len(df)
    n_primary = int(df["species_norm"].isin(PRIMARY_SPECIES).sum())
    n_path = int((df["n_findings"] > 0).sum())
    n_correct = int((df["quality_norm"] == "correct").sum())
    findings = {t for lst in df["tags_list"] for t in lst} & FINDING_LABELS
    size_gb = f"{dcm['FileSizeMB'].sum() / 1024:.0f} GB" if not dcm.empty else "n/a"

    tiles = [
        ("Total Images",      f"{total:,}",                                  ""),
        ("Dog & Cat Images",  f"{n_primary:,}",       f"{n_primary / total * 100:.1f}% of images"),
        ("Unique Breeds",     f"{df['breed_norm'].nunique():,}",             ""),
        ("Projections",       f"{df['projection_norm'].nunique():,}",        ""),
        ("Finding Labels",    f"{len(findings):,}",                          ""),
        ("Images w/ Findings", f"{n_path:,}",         f"{n_path / total * 100:.1f}% of images"),
        ("Correct Quality",   f"{n_correct:,}",       f"{n_correct / total * 100:.1f}% of images"),
        ("Dataset Size",      size_gb,                                       "DICOM (.dcm)"),
    ]

    fig = plt.figure(figsize=(14, 6.2))
    fig.patch.set_facecolor("#FFFFFF")
    gs = GridSpec(2, 4, figure=fig, hspace=0.18, wspace=0.14)

    for i, (title, value, subtitle) in enumerate(tiles):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.set_facecolor(TILE_COLORS[i])
        ax.text(0.5, 0.64, value, ha="center", va="center", transform=ax.transAxes,
                fontsize=32 if len(value) <= 7 else 26, fontweight="bold", color="white")
        ax.text(0.5, 0.32, title, ha="center", va="center", transform=ax.transAxes,
                fontsize=15, color="white")
        if subtitle:
            ax.text(0.5, 0.14, subtitle, ha="center", va="center", transform=ax.transAxes,
                    fontsize=14, fontweight="bold", color="white")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("VetXRay Dataset — Validation Summary", fontsize=20,
                 fontweight="bold", y=0.98)
    return fig


def figure_species_breeds(df, top_n=20):
    fig, (ax_sp, ax_br) = plt.subplots(1, 2, figsize=(14, 7),
                                       gridspec_kw={"width_ratios": [1, 1.4]})
    fig.suptitle("Species and Breed Composition", fontsize=15, fontweight="bold")

    species = df["species_norm"].fillna("Unspecified").value_counts()
    labels = [s if s in PRIMARY_SPECIES or s == "Unspecified" else "Other / exotic"
              for s in species.index]
    merged = pd.Series(species.values, index=labels).groupby(level=0).sum()
    merged = merged.sort_values(ascending=False)
    colors = [PALETTE.get(s, PALETTE["Other"]) for s in merged.index]

    bars = ax_sp.bar(merged.index, merged.values, color=colors)
    ax_sp.set_ylabel("Number of images")
    ax_sp.set_title("Species", fontsize=12)
    ax_sp.set_ylim(0, merged.max() * 1.10)
    for bar, val in zip(bars, merged.values):
        ax_sp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + merged.max() * 0.01,
                   f"{val:,}\n({val / len(df) * 100:.1f}%)",
                   ha="center", va="bottom", fontsize=10)

    breeds = df["breed_norm"].value_counts().head(top_n)
    bars = ax_br.barh(range(len(breeds)), breeds.values, color="#1F4E9C")
    ax_br.set_yticks(range(len(breeds)))
    ax_br.set_yticklabels(breeds.index, fontsize=9)
    ax_br.invert_yaxis()
    ax_br.set_xlabel("Number of images")
    ax_br.set_title(f"Top {top_n} breeds", fontsize=12)
    ax_br.set_xlim(0, breeds.max() * 1.12)
    for bar, val in zip(bars, breeds.values):
        ax_br.text(bar.get_width() + breeds.max() * 0.01,
                   bar.get_y() + bar.get_height() / 2,
                   f"{val:,}", va="center", fontsize=8)

    plt.tight_layout()
    return fig


def figure_quality_projection(df):
    fig, (ax_q, ax_p) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Acquisition Quality and Projection", fontsize=15, fontweight="bold")

    quality = df["quality_norm"].value_counts()
    colors = [QUAL_COLORS.get(q, "#4E6472") for q in quality.index]
    bars = ax_q.bar(quality.index, quality.values, color=colors)
    ax_q.set_ylabel("Number of images")
    ax_q.set_title("Quality category", fontsize=12)
    ax_q.set_ylim(0, quality.max() * 1.10)
    plt.setp(ax_q.get_xticklabels(), rotation=20, ha="right")
    for bar, val in zip(bars, quality.values):
        ax_q.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + quality.max() * 0.01,
                  f"{val:,}", ha="center", va="bottom", fontsize=10)

    proj = df["projection_norm"].value_counts()
    bars = ax_p.bar(proj.index, proj.values, color="#12695E")
    ax_p.set_ylabel("Number of images")
    ax_p.set_title("Projection", fontsize=12)
    ax_p.set_ylim(0, proj.max() * 1.10)
    for bar, val in zip(bars, proj.values):
        ax_p.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + proj.max() * 0.01,
                  f"{val:,}\n({val / len(df) * 100:.1f}%)",
                  ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    return fig


def figure_findings(df):
    counts = pd.Series(
        {k: v for k, v in Counter(t for lst in df["tags_list"] for t in lst).items()
         if k in FINDING_LABELS}
    ).sort_values(ascending=False)

    fig, (ax_f, ax_n) = plt.subplots(1, 2, figsize=(14, 7),
                                     gridspec_kw={"width_ratios": [1.5, 1]})
    fig.suptitle("Pathological Findings", fontsize=15, fontweight="bold")

    labels = [f.replace("_", " ").title() for f in counts.index]
    bars = ax_f.barh(range(len(counts)), counts.values, color="#B03A1E")
    ax_f.set_yticks(range(len(counts)))
    ax_f.set_yticklabels(labels, fontsize=9)
    ax_f.invert_yaxis()
    ax_f.set_xlabel("Number of images")
    ax_f.set_title("Label frequency (an image may carry several labels)", fontsize=11)
    ax_f.set_xlim(0, counts.max() * 1.15)
    for bar, val in zip(bars, counts.values):
        ax_f.text(bar.get_width() + counts.max() * 0.01,
                  bar.get_y() + bar.get_height() / 2,
                  f"{val:,} ({val / len(df) * 100:.1f}%)", va="center", fontsize=8)

    per_image = df["n_findings"].value_counts().sort_index()
    bars = ax_n.bar(per_image.index, per_image.values, color="#6A3D9A")
    ax_n.set_xlabel("Pathological findings per image")
    ax_n.set_ylabel("Number of images")
    ax_n.set_title("Findings per image", fontsize=11)
    ax_n.set_xticks(list(per_image.index))
    ax_n.set_ylim(0, per_image.max() * 1.10)
    for bar, val in zip(bars, per_image.values):
        ax_n.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + per_image.max() * 0.01,
                  f"{val:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


def figure_image_properties(dcm):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("DICOM Image Properties", fontsize=15, fontweight="bold")

    ax = axes[0][0]
    ax.scatter(dcm["Columns"], dcm["Rows"], s=12, alpha=0.35, color="#1F4E9C",
               edgecolors="none")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image dimensions", fontsize=11)

    ax = axes[0][1]
    ax.hist(dcm["FileSizeMB"], bins=40, color="#12695E", edgecolor="white")
    ax.set_xlabel("File size (MB)")
    ax.set_ylabel("Number of files")
    ax.set_title(f"File size — total {dcm['FileSizeMB'].sum() / 1024:.1f} GB", fontsize=11)

    ax = axes[1][0]
    photo = dcm["PhotometricInterpretation"].value_counts()
    bars = ax.bar([str(i) for i in photo.index], photo.values, color="#8A5A08")
    ax.set_ylabel("Number of files")
    ax.set_title("Photometric interpretation", fontsize=11)
    ax.set_ylim(0, photo.max() * 1.12)
    for bar, val in zip(bars, photo.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + photo.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)

    ax = axes[1][1]
    depth = dcm.groupby(["BitsAllocated", "BitsStored"]).size().sort_values(ascending=False)
    labels = [f"{int(a)} alloc / {int(s)} stored" for a, s in depth.index]
    bars = ax.bar(labels, depth.values, color="#A32E6E")
    ax.set_ylabel("Number of files")
    ax.set_title("Bit depth", fontsize=11)
    ax.set_ylim(0, depth.max() * 1.12)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
    for bar, val in zip(bars, depth.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + depth.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def build_figures(df, dcm, out_dir):
    """Create every summary figure; returns {filename: Figure}."""
    print("\nGenerating summary figures …")
    needed = {"species_norm", "breed_norm", "projection_norm", "quality_norm",
              "tags_list", "n_findings"}
    if not needed.issubset(df.columns):
        print("  Skipped — the spreadsheet is missing columns the figures rely on: "
              f"{', '.join(sorted(needed - set(df.columns)))}")
        return {}

    figures = {
        "validation_overview.png":          figure_overview(df, dcm),
        "validation_species_breeds.png":    figure_species_breeds(df),
        "validation_quality_projection.png": figure_quality_projection(df),
        "validation_findings.png":          figure_findings(df),
    }
    if not dcm.empty:
        figures["validation_image_properties.png"] = figure_image_properties(dcm)

    if out_dir:
        for name, fig in figures.items():
            save_fig(fig, name, out_dir)
    return figures


# ══════════════════════════════════════════════════════════════════════════════
# 12. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Validate a downloaded copy of the VetXRay dataset.")
    p.add_argument("--dicom-dir", default=DICOM_DIR,
                   help="folder containing the released .dcm files")
    p.add_argument("--xlsx", default=XLSX_PATH,
                   help="released annotation spreadsheet (.xlsx)")
    p.add_argument("--output-dir", default=OUTPUT_DIR,
                   help="where figures and the validation report are written")
    p.add_argument("--max-files", type=int, default=MAX_FILES,
                   help="limit the DICOM header scan (0 or omitted = every file)")
    p.add_argument("--pixel-sample", type=int, default=PIXEL_SAMPLE,
                   help="how many images are fully decoded")
    p.add_argument("--quick", action="store_true",
                   help="shortcut for --max-files 500 --pixel-sample 10")
    p.add_argument("--no-figures", action="store_true",
                   help="run the checks and statistics only")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.max_files = 500
        args.pixel_sample = 10
    max_files = None if not args.max_files else args.max_files

    print("═" * 78)
    print("VetXRay Dataset — validation")
    print("═" * 78)

    xlsx_path = resolve_xlsx(args.xlsx)
    if not xlsx_path or not os.path.isfile(xlsx_path):
        sys.exit("Annotation spreadsheet not found. Set XLSX_PATH at the top of this "
                 "script or pass --xlsx /path/to/annotations.xlsx")
    if not args.dicom_dir or not os.path.isdir(args.dicom_dir):
        sys.exit("DICOM folder not found. Set DICOM_DIR at the top of this script or "
                 "pass --dicom-dir /path/to/dicom")

    print(f"  DICOM folder : {args.dicom_dir}")
    print(f"  Spreadsheet  : {xlsx_path}")
    print(f"  Output       : {args.output_dir}")

    log = CheckLog()
    df = load_metadata(xlsx_path)
    disk_files = list_dicom_files(args.dicom_dir)

    check_spreadsheet(df, log)
    check_vocabularies(df, log)
    check_file_crossref(df, disk_files, args.dicom_dir, log)
    dcm = scan_dicom_headers(args.dicom_dir, disk_files, log, max_files=max_files)
    check_dicom_consistency(dcm, log)
    check_pixel_data(args.dicom_dir, dcm, log, n_sample=args.pixel_sample)
    check_anonymisation(dcm, log)

    print_statistics(df, dcm)

    if not args.no_figures:
        figures = build_figures(df, dcm, args.output_dir)
        for fig in figures.values():
            plt.close(fig)

    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, "validation_report.csv")
    log.frame.to_csv(report_path, index=False)
    print(f"\n  Saved: {report_path}")

    log.print_summary()
    return 1 if log.n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
