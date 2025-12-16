#!/usr/bin/env python3
"""
CS156 Assignment MD to DOCX Converter
Converts merged markdown to Word document with professional styling.

Usage:
    python convert_to_docx.py

Requirements:
    - Pandoc: brew install pandoc
    - python-docx: pip install python-docx
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def check_pandoc():
    """Check if pandoc is installed."""
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version = result.stdout.split("\n")[0]
        print(f"✅ Found: {version}")
        return True
    except FileNotFoundError:
        print("❌ Pandoc not found. Install with: brew install pandoc")
        return False


def convert_md_to_docx(input_path: Path, output_path: Path, reference_doc: Path = None):
    """Convert markdown to docx using pandoc."""

    print(f"📄 Input:  {input_path}")
    print(f"📄 Output: {output_path}")

    cmd = [
        "pandoc",
        str(input_path),
        "-o", str(output_path),
        "--from=markdown",
        "--to=docx",
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "--highlight-style=tango",
        "-M", "title=CS156 Pipeline - Phase 4: Subvocalization",
        "-M", "author=Carl Vincent Kho",
        "-M", f"date={datetime.now().strftime('%B %d, %Y')}",
    ]

    # Add reference doc if available
    if reference_doc and reference_doc.exists():
        cmd.extend(["--reference-doc", str(reference_doc)])
        print(f"📎 Using reference: {reference_doc.name}")

    print(f"\n🔄 Converting...")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success! Generated: {output_path.name}")
        print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e.stderr}")
        return False


def main():
    print("=" * 60)
    print("📘 CS156 ASSIGNMENT → DOCX CONVERTER")
    print("=" * 60)
    print()

    if not check_pandoc():
        sys.exit(1)

    script_dir = Path(__file__).parent

    # First, generate merged markdown
    print("\n📝 Step 1: Generating merged markdown...")
    from generate_merged_assignment import generate_merged_assignment
    merged_md = generate_merged_assignment()

    if not merged_md.exists():
        print(f"❌ Merged file not found: {merged_md}")
        sys.exit(1)

    # Output with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = script_dir / f"[Kho]_CS156_Phase4_{timestamp}.docx"

    # Check for reference.docx
    reference_doc = script_dir / "reference.docx"

    print("\n📝 Step 2: Converting to DOCX...")
    if convert_md_to_docx(merged_md, output_path, reference_doc):
        # Post-process tables
        print("\n📝 Step 3: Styling tables...")
        try:
            from style_tables import style_docx_tables
            style_docx_tables(output_path)
        except ImportError:
            print("⚠️  style_tables.py not found, skipping table styling")
        except Exception as e:
            print(f"⚠️  Table styling error: {e}")

        print()
        print("=" * 60)
        print(f"🎉 Done! Open with:")
        print(f'   open "{output_path}"')
        print("=" * 60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
