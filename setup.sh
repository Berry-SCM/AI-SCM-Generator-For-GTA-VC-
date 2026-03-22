#!/bin/bash
# Setup script for GTA VC SCM AI Pipeline

echo "Creating directory structure..."
mkdir -p data/raw/main_scm_segments
mkdir -p data/raw/map/maps
mkdir -p data/raw/paths
mkdir -p data/processed
mkdir -p models
mkdir -p output

echo "Creating Python package __init__.py files..."
touch parsers/__init__.py
touch spatial/__init__.py
touch training/__init__.py
touch generator/__init__.py

echo ""
echo "Placing your GTA VC data files:"
echo "  Option A (preferred) — single full SCM file:"
echo "    Copy your full decompiled main.scm text to:  data/raw/main.txt"
echo "    Copy your stripped SCM (bare minimum) to:    data/raw/stripped.txt"
echo ""
echo "  Option B (fallback) — segmented files:"
echo "    Copy main[0]_1.txt, main[0]_2.txt, etc. to: data/raw/main_scm_segments/"
echo ""
echo "  Map files (copy from your GTA VC DATA directory):"
echo "    data/raw/map/info.zon"
echo "    data/raw/map/default.ide"
echo "    data/raw/map/maps/<area>/<area>.IPL  (all IPL files)"
echo "    data/raw/map/maps/<area>/<area>.IDE  (all area IDE files)"
echo ""
echo "  Path/nav nodes — IMPORTANT for coordinate quality:"
echo "    Copy entire DATA\\paths\\ folder contents to: data/raw/paths/"
echo "    This gives the AI real road + footpath node coordinates."
echo "    (ROADBLOCKS.DAT and similar text DAT files from that folder)"
echo ""
echo "  Game location (Steam default):"
echo "    C:\\Program Files (x86)\\Steam\\steamapps\\common\\Grand Theft Auto Vice City\\"
echo "    DATA\\  -> contains info.zon, default.ide, gta_vc.dat"
echo "    DATA\\MAPS\\  -> contains all .IPL and .IDE area files"
echo "    DATA\\paths\\ -> contains path node DAT files"
echo ""
echo "  COL files: NOT needed. The game engine handles collision at runtime."
echo "  The AI uses zone bounds + IPL placement + path nodes for spatial awareness."

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Done! Run the pipeline in order:"
echo "  python main.py --mode parse          # Parse SCM + map + path node files"
echo "  python main.py --mode scrape         # Build opcode/vehicle/ped/weapon ID data"
echo "  python main.py --mode train          # Fine-tune LLM (requires GPU, ~4-8 hours)"
echo "  python main.py --mode generate --missions 5"
echo "  python main.py --mode validate --input output/new_main.scm"
echo ""
echo "  Or run everything at once:"
echo "  python main.py --mode all --missions 5"
