#!/bin/bash
# Build script for CSDP paper

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "CSDP Paper Build"
echo "=================================="

# Step 1: Extract data from experiment reports
echo ""
echo "Step 1: Extracting data from experiment reports..."
python3 scripts/extract_data.py

# Step 2: Generate figures
echo ""
echo "Step 2: Generating figures..."
python3 scripts/generate_figures.py

# Step 3: Compile LaTeX (if main.tex exists)
if [ -f "main.tex" ]; then
    echo ""
    echo "Step 3: Compiling LaTeX..."
    if command -v tectonic &> /dev/null; then
        tectonic main.tex
    elif command -v pdflatex &> /dev/null; then
        pdflatex -interaction=nonstopmode main.tex
        pdflatex -interaction=nonstopmode main.tex  # Run twice for references
    else
        echo "Warning: No LaTeX compiler found (tectonic or pdflatex)"
        echo "Skipping LaTeX compilation"
    fi
else
    echo ""
    echo "Step 3: Skipping LaTeX (main.tex not found)"
fi

echo ""
echo "=================================="
echo "Build complete!"
echo "=================================="
echo ""
echo "Generated files:"
ls -la figures/*.pdf 2>/dev/null || echo "  No figures yet"
ls -la scripts/extracted_data.json 2>/dev/null || echo "  No extracted data yet"
[ -f main.pdf ] && echo "  main.pdf"
