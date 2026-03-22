#!/bin/bash
# Setup script

echo "Creating directory structure..."
mkdir -p data/raw/main_scm_segments
mkdir -p data/raw/map/maps
mkdir -p data/processed
mkdir -p models
mkdir -p output

echo "Copying your GTA VC data files..."
# Copy your main[0]_1.txt, main[0]_2.txt, etc. to:
#   data/raw/main_scm_segments/
# Copy your map files (info.zon, gta_vc.dat, *.IPL, *.IDE) to:
#   data/raw/map/

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Done! Run pipeline:"
echo "  python main.py --mode parse"
echo "  python main.py --mode train       # requires GPU, ~4-8 hours"
echo "  python main.py --mode generate --missions 5"
echo "  python main.py --mode validate --input output/new_main.scm"