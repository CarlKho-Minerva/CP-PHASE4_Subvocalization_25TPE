#!/usr/bin/env python3
"""
CS156 Assignment Generator - Phase 4: Subvocalization
Merges all 10 section markdown files into one final submission.

Run this whenever you update any individual section file.
"""

from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).parent

# Files in order of appearance (matching assignment rubric)
SECTION_FILES = [
    ("01_data_explanation.md", "Section 1: Data Explanation"),
    ("02_data_loading.md", "Section 2: Data Loading & Python Conversion"),
    ("03_preprocessing.md", "Section 3: Preprocessing, Cleaning & EDA"),
    ("04_analysis_plan.md", "Section 4: Analysis Plan & Data Splits"),
    ("05_model_selection.md", "Section 5: Model Selection & Mathematical Foundations"),
    ("06_training.md", "Section 6: Model Training"),
    ("07_predictions.md", "Section 7: Predictions & Performance Metrics"),
    ("08_visualization.md", "Section 8: Visualization & Conclusions"),
    ("09_executive_summary.md", "Section 9: Executive Summary"),
    ("10_references.md", "Section 10: References"),
]

# Output file
OUTPUT_FILE = BASE_DIR / "[Kho]_CS156_Phase4_Subvocalization_MERGED.md"


def generate_merged_assignment():
    """Concatenate all section files into final submission."""

    output_lines = []

    # Header
    output_lines.append("# CS156 Pipeline - Final Draft\n")
    output_lines.append("## Phase 4: Subvocalization Detection with Low-Cost Hardware\n\n")
    output_lines.append(f"**Student:** Carl Vincent Kho\n")
    output_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_lines.append("**Course:** CS156 - Machine Learning Pipeline\n")
    output_lines.append("\n---\n\n")

    # Table of Contents
    output_lines.append("## Table of Contents\n\n")
    for i, (filename, title) in enumerate(SECTION_FILES, 1):
        # Create anchor from title
        anchor = title.lower().replace(" ", "-").replace(":", "").replace("&", "and")
        output_lines.append(f"{i}. [{title}](#{anchor})\n")
    output_lines.append("\n---\n\n")

    # Append each section file
    for filename, title in SECTION_FILES:
        filepath = BASE_DIR / filename

        if not filepath.exists():
            print(f"⚠️  WARNING: {filename} not found, skipping...")
            output_lines.append(f"## {title}\n\n")
            output_lines.append(f"> ⚠️ File `{filename}` not found.\n\n")
            output_lines.append("---\n\n")
            continue

        print(f"✅ Adding: {filename}")

        # Read file content
        content = filepath.read_text(encoding="utf-8")

        # Add page break before each section (for PDF/DOCX)
        output_lines.append(f"\n\n<div style='page-break-before: always;'></div>\n\n")

        # Append content
        output_lines.append(content)
        output_lines.append("\n\n---\n\n")

    # Write output
    OUTPUT_FILE.write_text("".join(output_lines), encoding="utf-8")
    print(f"\n🎉 Merged assignment generated: {OUTPUT_FILE.name}")
    print(f"   Total size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")

    return OUTPUT_FILE


if __name__ == "__main__":
    generate_merged_assignment()
