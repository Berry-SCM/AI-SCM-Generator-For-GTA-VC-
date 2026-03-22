#!/bin/bash
# Setup script

echo "Creating directory structure..."
mkdir -p data/raw/main_scm_segments
mkdir -p data/raw/map/maps
mkdir -p data/processed
mkdir -p models
mkdir -p output

echo ""
echo "=== COPYING YOUR GTA VC DATA FILES ==="
echo ""
echo "OPTION A (preferred) — single full SCM file:"
echo "  Copy your full decompiled main.scm text to:"
echo "    data/raw/main.txt"
echo ""
echo "OPTION B (fallback) — segmented SCM files:"
echo "  Copy main[0]_1.txt, main[0]_2.txt, etc. to:"
echo "    data/raw/main_scm_segments/"
echo "  (The pipeline uses this if data/raw/main.txt is absent)"
echo ""
echo "MAP FILES — copy all of these:"
echo "  info.zon, gta_vc.dat, default.ide  →  data/raw/map/"
echo "  All *.IPL files                    →  data/raw/map/maps/<area>/"
echo ""

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Done! Run pipeline:"
echo "  python main.py --mode parse"
echo "  python main.py --mode train       # requires GPU, ~4-8 hours"
echo "  python main.py --mode generate --missions 5"
echo "  python main.py --mode validate --input output/new_main.scm"
