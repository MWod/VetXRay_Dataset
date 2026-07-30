# VetXRay: A dataset of 9,882 manually annotated canine and feline thoracic radiographs with lesion and image quality annotations

Loading example and validation utilities for the **VetXRay** dataset of canine and
feline thoracic radiographs.

The dataset itself is distributed separately and consists of two artefacts: a folder
of DICOM images (`*.dcm`) and an annotation spreadsheet (`*.xlsx`). The scripts here
assume nothing else.

**Dataset:** `<DOI / repository link>` · **Paper:** `<citation>`

## Contents

| File | Purpose |
|---|---|
| `example.py` / `example.ipynb` | Load one radiograph, look up its annotation, preprocess and display it |
| `validate.py` / `validate.ipynb` | Verify a downloaded copy, print the summary statistics, and write the summary figures |

The `.py` and `.ipynb` forms are equivalent; use whichever you prefer.
`validate.ipynb` imports `validate.py`, so keep the two in the same folder.

## Requirements

```bash
pip install pydicom pandas numpy matplotlib openpyxl
```

## Usage

Set `DICOM_DIR` and `XLSX_PATH` at the top of either script, then:

```bash
python example.py
python validate.py
```

`validate.py` also takes the paths on the command line:

```bash
python validate.py --dicom-dir /path/to/dicom --xlsx /path/to/annotations.xlsx
python validate.py --dicom-dir ... --xlsx ... --quick   # partial, fast pass
```

## What is validated

Spreadsheet structure · controlled vocabularies · spreadsheet ↔ disk
cross-reference · DICOM headers · pixel data · de-identification.

Each check is reported as PASS / WARN / FAIL, where **FAIL** means the downloaded
copy is incomplete or corrupt and **WARN** flags a property of the release worth
knowing before analysis. The exit status is non-zero only on FAIL. Results are
written to `validation_output/` as five figures and `validation_report.csv`.

## License

`<license>`
