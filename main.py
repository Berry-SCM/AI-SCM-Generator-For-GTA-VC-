"""
GTA Vice City AI SCM Generator — Main Pipeline Entry Point
Corrected: run_parse() now also calls PathsParser for road/ped node extraction.
MapGraph now receives paths_json parameter.
"""
import argparse
import os
import glob


def run_parse():
    print("=" * 60)
    print("STEP 1: PARSING GTA VC DATA FILES")
    print("=" * 60)

    from parsers.zon_parser import ZonParser
    from parsers.ipl_parser import IPLParser
    from parsers.ide_parser import IDEParser
    from parsers.scm_parser import SCMParser
    from parsers.paths_parser import PathsParser
    import json
    from pathlib import Path

    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # ── Parse zone file ──────────────────────────────────────────────────────
    zon_path = "data/raw/map/info.zon"
    if os.path.exists(zon_path):
        zp = ZonParser(zon_path)
        zp.parse()
        zp.to_json("data/processed/zones_db.json")
        print(f"[Parse] Zones: {len(zp.zones)} zones parsed")
    else:
        print(f"[Parse] WARNING: {zon_path} not found — skipping zone parse")

    # ── Parse IDE files ───────────────────────────────────────────���──────────
    ide_files = ["data/raw/map/default.ide"] + \
                glob.glob("data/raw/map/maps/**/*.IDE", recursive=True) + \
                glob.glob("data/raw/map/maps/**/*.ide", recursive=True)
    all_ide_entries = []
    for ide_path in ide_files:
        if os.path.exists(ide_path):
            try:
                idep = IDEParser(ide_path)
                idep.parse()
                all_ide_entries.extend([
                    {"id": e.id, "model_name": e.model_name, "txd_name": e.txd_name,
                     "entry_type": e.entry_type, "extra": e.extra}
                    for e in idep.entries
                ])
            except Exception as ex:
                print(f"[Parse] Warning: could not parse {ide_path}: {ex}")
    if all_ide_entries:
        with open("data/processed/ide_entries.json", "w") as f:
            json.dump(all_ide_entries, f, indent=2)
        print(f"[Parse] IDE: {len(all_ide_entries)} entries saved")

    # ── Parse IPL files ──────────────────────────────────────────────────────
    ipl_files = glob.glob("data/raw/map/maps/**/*.IPL", recursive=True) + \
                glob.glob("data/raw/map/maps/**/*.ipl", recursive=True)
    all_instances = []
    all_zones_ipl = []
    for ipl_path in ipl_files:
        if os.path.exists(ipl_path):
            try:
                ip = IPLParser(ipl_path)
                ip.parse()
                all_instances.extend([
                    {"obj_id": inst.obj_id, "model_name": inst.model_name,
                     "interior": inst.interior, "x": inst.x, "y": inst.y, "z": inst.z,
                     "rx": inst.rx, "ry": inst.ry, "rz": inst.rz, "rw": inst.rw,
                     "lod": inst.lod}
                    for inst in ip.instances
                ])
                all_zones_ipl.extend([
                    {"name": z.name, "type": z.type,
                     "x1": z.x1, "y1": z.y1, "z1": z.z1,
                     "x2": z.x2, "y2": z.y2, "z2": z.z2,
                     "island": z.island}
                    for z in ip.zones
                ])
            except Exception as ex:
                print(f"[Parse] Warning: could not parse {ipl_path}: {ex}")
    if all_instances:
        with open("data/processed/ipl_instances.json", "w") as f:
            json.dump({"instances": all_instances, "zones": all_zones_ipl}, f)
        print(f"[Parse] IPL: {len(all_instances)} instances, {len(all_zones_ipl)} zones saved")

    # ── Parse path nodes (NEW) ───────────────────────────────────────────────
    # Path nodes are the most accurate source of drivable/walkable coordinates.
    # They come from two sources:
    #   1. 'path' sections inside IPL files
    #   2. DATA/paths/*.DAT text files (e.g. ROADBLOCKS.DAT)
    pp = PathsParser()

    maps_dir = "data/raw/map/maps"
    if os.path.isdir(maps_dir):
        pp.parse_all_ipls(maps_dir)

    paths_dir = "data/raw/paths"
    if os.path.isdir(paths_dir):
        pp.parse_paths_dat_dir(paths_dir)
    else:
        print(f"[Parse] Info: {paths_dir} not found — "
              f"copy DATA\\paths\\ folder to data/raw/paths/ for better coord quality")

    pp.to_json("data/processed/path_nodes.json")

    # ── Parse SCM ────────────────────────────────────────────────────────────
    scm_parsed = False
    for scm_path in ["data/raw/main.txt", "data/raw/stripped.txt"]:
        if os.path.exists(scm_path):
            try:
                sp = SCMParser(scm_path)
                scm_file = sp.parse()
                sp.to_json("data/processed/scm_parsed.json")
                print(f"[Parse] SCM: {len(scm_file.scripts)} scripts, "
                      f"{len(scm_file.missions)} missions, "
                      f"{len(scm_file.objects)} objects parsed from {scm_path}")
                scm_parsed = True
                break
            except Exception as ex:
                print(f"[Parse] Warning: could not parse {scm_path}: {ex}")

    if not scm_parsed:
        seg_files = sorted(glob.glob("data/raw/main_scm_segments/*.txt"))
        if seg_files:
            combined = ""
            for sf in seg_files:
                with open(sf, encoding="utf-8", errors="replace") as f:
                    combined += f.read() + "\n"
            combined_path = "data/raw/main_combined.txt"
            with open(combined_path, "w") as f:
                f.write(combined)
            try:
                sp = SCMParser(combined_path)
                scm_file = sp.parse()
                sp.to_json("data/processed/scm_parsed.json")
                print(f"[Parse] SCM (from {len(seg_files)} segments): "
                      f"{len(scm_file.scripts)} scripts parsed")
            except Exception as ex:
                print(f"[Parse] Error parsing combined segments: {ex}")
        else:
            print("[Parse] WARNING: No SCM files found. Place main.txt in data/raw/")

    print("[Parse] Done.")


def run_scrape():
    """Step 1b: Build opcode + vehicle/ped/weapon ID reference data."""
    print("=" * 60)
    print("STEP 1b: BUILDING ID & OPCODE REFERENCE DATA")
    print("=" * 60)
    from training.opcode_scraper import save_id_data
    save_id_data()
    print("[Scrape] Done.")


def run_train():
    print("=" * 60)
    print("STEP 2: FINE-TUNING LLM ON SCM DATA")
    print("=" * 60)

    scm_json = "data/processed/scm_parsed.json"
    if not os.path.exists(scm_json):
        print("[Train] ERROR: Run --mode parse first to generate scm_parsed.json")
        return

    from spatial.map_graph import MapGraph
    zones_path  = "data/processed/zones_db.json"
    ipl_path    = "data/processed/ipl_instances.json"
    paths_path  = "data/processed/path_nodes.json"
    mg = MapGraph(
        zones_json=zones_path  if os.path.exists(zones_path)  else None,
        ipl_json=ipl_path      if os.path.exists(ipl_path)    else None,
        paths_json=paths_path  if os.path.exists(paths_path)  else None,
    )

    from training.dataset_builder import DatasetBuilder
    db = DatasetBuilder(scm_json, map_graph=mg)
    db.build_all()
    db.save_jsonl("data/processed/training_pairs.jsonl")

    from training.finetune import train
    train()


def run_generate(num_missions: int, output_path: str):
    print("=" * 60)
    print("STEP 3: GENERATING NEW MAIN.SCM")
    print("=" * 60)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    zones_path  = "data/processed/zones_db.json"
    ipl_path    = "data/processed/ipl_instances.json"
    paths_path  = "data/processed/path_nodes.json"
    from spatial.map_graph import MapGraph
    mg = MapGraph(
        zones_json=zones_path  if os.path.exists(zones_path)  else None,
        ipl_json=ipl_path      if os.path.exists(ipl_path)    else None,
        paths_json=paths_path  if os.path.exists(paths_path)  else None,
    )

    model_path = "models/gtavc_scm_lora"
    if not os.path.exists(model_path):
        print(f"[WARNING] Fine-tuned model not found at {model_path}")
        print("[INFO] Using template-based generation (no LLM) as fallback")
        _generate_template_based(num_missions, output_path, mg)
        return

    from generator.mission_generator import MissionGenerator
    gen = MissionGenerator(model_path=model_path, map_graph=mg)
    scm_text = gen.generate_full_mod(num_missions=num_missions)

    with open(output_path, "w") as f:
        f.write(scm_text)
    print(f"[Generate] Wrote {output_path}")
    print("[Generate] Done. Run --mode validate to check it.")


def _generate_template_based(num_missions: int, output_path: str, map_graph):
    """Fallback generator using templates without LLM."""
    import random
    from generator.mission_generator import MISSION_CONCEPTS, SCMAssembler

    concepts = random.sample(MISSION_CONCEPTS, min(num_missions, len(MISSION_CONCEPTS)))
    missions = []
    trigger_scripts = []
    mission_bodies = []

    for idx, concept in enumerate(concepts):
        name = f"MOD{idx:02d}"
        loc = map_graph.get_random_outdoor_coord(concept.get("zone"))
        x, y, z = round(loc[0], 3), round(loc[1], 3), round(loc[2], 3)

        missions.append({"name": name, "label": name, "title": concept["title"]})

        trigger_scripts.append(f"""
:{name}_LOOP
wait 0
if
  Player.Defined($player_char)
goto_if_false @{name}_LOOP
if
  locate_player_on_foot_3d $player_char 0 {x} {y} {z} radius 2.0 2.0 2.0
goto_if_false @{name}_LOOP
if
  $onmission == 0
goto_if_false @{name}_LOOP
if
  Player.Controllable($player_char)
goto_if_false @{name}_LOOP
$onmission = 1
Player.MakeSafe($player_char)
set_fading_colour 0 0 0
do_fade 0 1500
print_big '{name[:8]}' 15000 ms 2
load_and_launch_mission_internal {idx}
goto @{name}_LOOP
""")

        tx, ty, tz = round(x + 10, 3), round(y + 10, 3), z
        mission_bodies.append((name, f"""
:{name}
script_name '{name[:8]}'
$onmission = 1
set_fading_colour 0 0 0
do_fade 0 2000
wait 2000
do_fade 1 1000

// Objective: {concept['description']}
$obj_marker = create_marker 4 at {tx} {ty} {tz}

:{name}_WAIT
wait 0
if
  not Player.Defined($player_char)
goto_if_false @{name}_FAIL
if
  Actor.Dead($player_actor)
goto_if_false @{name}_FAIL
if
  locate_player_on_foot_3d $player_char 0 {tx} {ty} {tz} radius 3.0 3.0 3.0
goto_if_false @{name}_WAIT
goto @{name}_PASS

:{name}_FAIL
Blip.Remove($obj_marker)
$onmission = 0
do_fade 0 1000
wait 1000
do_fade 1 500
terminate_this_script

:{name}_PASS
Blip.Remove($obj_marker)
$onmission = 0
do_fade 0 1000
wait 1000
do_fade 1 500
terminate_this_script
"""))

    assembler = SCMAssembler(missions, trigger_scripts, mission_bodies, map_graph)
    scm_text = assembler.assemble()
    with open(output_path, "w") as f:
        f.write(scm_text)
    print(f"[Generate] Template-based SCM written to {output_path}")


def run_validate(input_path: str):
    print("=" * 60)
    print("STEP 4: VALIDATING SCM")
    print("=" * 60)

    if not os.path.exists(input_path):
        print(f"[Validate] ERROR: File not found: {input_path}")
        return

    from generator.validator import SCMValidator
    from spatial.map_graph import MapGraph

    zones_path = "data/processed/zones_db.json"
    paths_path = "data/processed/path_nodes.json"
    mg = MapGraph(
        zones_json=zones_path if os.path.exists(zones_path) else None,
        paths_json=paths_path if os.path.exists(paths_path) else None,
    )
    validator = SCMValidator(map_graph=mg)

    with open(input_path) as f:
        scm_text = f.read()

    valid, errors = validator.validate(scm_text)
    if valid:
        print("[Validate] ✅ SCM is valid!")
    else:
        print(f"[Validate] ❌ Found {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        print("\n[Validate] Attempting auto-fix...")
        fixed = validator.auto_fix_coords(scm_text)
        fixed_path = input_path.replace(".scm", "_fixed.scm").replace(".txt", "_fixed.txt")
        with open(fixed_path, "w") as f:
            f.write(fixed)
        print(f"[Validate] Fixed SCM written to {fixed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTA VC AI SCM Generator")
    parser.add_argument("--mode",
                        choices=["parse", "scrape", "train", "generate", "validate", "all"],
                        required=True, help="Pipeline mode")
    parser.add_argument("--missions", type=int, default=5,
                        help="Number of missions to generate")
    parser.add_argument("--output", type=str, default="output/new_main.scm",
                        help="Output path for generated SCM")
    parser.add_argument("--input", type=str, default="output/new_main.scm",
                        help="Input path for validate mode")
    args = parser.parse_args()

    if args.mode == "parse":
        run_parse()
    elif args.mode == "scrape":
        run_scrape()
    elif args.mode == "train":
        run_train()
    elif args.mode == "generate":
        run_generate(args.missions, args.output)
    elif args.mode == "validate":
        run_validate(args.input)
    elif args.mode == "all":
        run_parse()
        run_scrape()
        run_train()
        run_generate(args.missions, args.output)
        run_validate(args.output)
