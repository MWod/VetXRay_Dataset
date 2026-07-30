"""
VetXRay Dataset — Minimal loading example
==========================================
Shows how to load a single chest X-ray from the VetXRay dataset,
retrieve its annotation from the metadata spreadsheet, apply basic
image preprocessing, and display the result.

Dependencies
------------
    pip install pydicom pandas numpy matplotlib openpyxl

Usage
-----
Set DICOM_DIR and XLSX_PATH to match your local installation,
then run:  python example.py
"""

import os
import pydicom
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────
# Path to the folder containing all .dcm files
DICOM_DIR = None # TODO: set this to the path where your DICOM files are located

# Path to the annotation spreadsheet
XLSX_PATH = None # TODO: set this to the path of the annotation spreadsheet (e.g., "annotations.xlsx")

# Which file to load (must exist in DICOM_DIR and in the spreadsheet)
SAMPLE_FILE = "IM-0015-0001-0001.dcm" # Example filename, change as needed

# ── 1. Load the DICOM file ─────────────────────────────────────────────────────
dicom_path = os.path.join(DICOM_DIR, SAMPLE_FILE)
ds = pydicom.dcmread(dicom_path)

# Raw pixel data as a 2-D NumPy array (uint16)
pixels = ds.pixel_array

print(f"Loaded:  {SAMPLE_FILE}")
print(f"Shape:   {pixels.shape}  |  dtype: {pixels.dtype}")
print(f"Modality: {getattr(ds, 'Modality', 'N/A')}")

# ── 2. Look up the annotation row ─────────────────────────────────────────────
annotations = pd.read_excel(XLSX_PATH, engine="openpyxl")
row = annotations.loc[annotations["FileName"] == SAMPLE_FILE].iloc[0]

species    = row["specie"]
breed      = row["breed"]
projection = row["Projection"]
quality    = row["Quality"]
# TAG is pipe-separated; "no_finding" means a healthy image
findings   = [t.strip() for t in str(row["TAG"]).split("|") if t.strip()]

print(f"\nAnnotation:")
print(f"  Species / Breed : {species} / {breed}")
print(f"  Projection      : {projection}")
print(f"  Quality         : {quality}")
print(f"  Findings        : {', '.join(findings)}")

# ── 3. Preprocess the image ────────────────────────────────────────────────────
# Convert to float and apply percentile-based contrast stretch
img = pixels.astype(np.float32)
p_low, p_high = np.percentile(img, [2, 98])
img = np.clip(img, p_low, p_high)
img = (img - p_low) / (p_high - p_low)   # now in [0, 1]

# MONOCHROME1: high pixel value = dark on film → invert so lungs appear dark
if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
    img = 1.0 - img

# ── 4. Display ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 7))

ax.imshow(img, cmap="gray", aspect="equal")
ax.axis("off")

ax.set_title(
    f"{species}  ·  {breed}  ·  Projection: {projection}",
    fontsize=12, fontweight="bold", pad=10,
)

# Annotation box below the image
finding_text = ", ".join(findings) if findings else "—"
info = f"Quality: {quality}\nFindings: {finding_text}"
ax.text(
    0.5, -0.02, info,
    transform=ax.transAxes,
    ha="center", va="top",
    fontsize=9, color="#333333",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", edgecolor="#cccccc"),
)

plt.tight_layout()
plt.show()
