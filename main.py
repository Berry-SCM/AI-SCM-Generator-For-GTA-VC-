"""
Main entry point for the GTA VC SCM AI pipeline.

Usage:
  # Step 1: Parse and build dataset
  python main.py --mode parse

  # Step 2: Fine-tune the LLM
  python main.py --mode train

  # Step 3: Generate a new mod
  python main.py --mode generate --missions 5 --output output/new_main.scm

  # Step 4: Validate a SCM file
  python main.py --mode validate --input output/new_main.scm

Input files:
  Single file:   data/raw/main.txt        (preferred — full decompiled main.scm)
  Segmented:     data/raw/main_scm_segments/*.txt  (fallback if main.txt absent)
  Map files:     data/raw/map/info.zon, data/raw/map/default.ide, data/raw/map/maps/**/*.IPL
"""

import argparse
import os
import glob
import json


def run_parse():
    print("=" * 60)
    print("STEP 1: PARSING GTA VC DATA FILES")
    print("=" * 60)
    os.makedirs("data/processed", exist_ok=True)

    from parsers.scm_parser import SCMParser
    all_scripts = []
    all_missions = []
    all_objects = []

    # Prefer single full file, fall back to segments
    single_file = "data/raw/main.txt"
    segments = sorted(glob.glob("data/raw/main_scm_segments/*.txt"))

    if os.path.exists(single_file):
        print(f"  Using single SCM file: {single_file}")
        scm_files = [single_file]
    elif segments:
        print(f"  Using {len(segments)} segmented SCM files from data/raw/main_scm_segments/")
        scm_files = segments
    else:
        print("  [WARNING] No SCM source files found.")
        print("    Place data/raw/main.txt  OR  data/raw/main_scm_segments/*.txt")
        scm_files = []

    for seg in scm_files:
        print(f"  Parsing: {seg}")
        parser = SCMParser(seg)
        scm = parser.parse()
        all_scripts.extend([{
            'name': s.name, 'label': s.label,
            'coords': s.coords_used,
            'variables': s.variables_used,
            'raw': [i.raw for i in s.instructions]
        } for s in scm.scripts])
        all_missions.extend(scm.missions)
        all_objects.extend(scm.objects)

    combined = {
        'objects': list(set(all_objects)),
        'missions': all_missions,
        'scripts': all_scripts
    }
    with open("data/processed/scm_parsed.json", 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"  ✓ Parsed {len(all_scripts)} scripts, {len(all_missions)} missions")

    # Parse zone files
    from parsers.zon_parser import ZonParser
    if os.path.exists("data/raw/map/info.zon"):
        zon = ZonParser("data/raw/map/info.zon")
        zon.parse()
        zon.to_json("data/processed/zones_db.json")
        print(f"  ✓ Parsed {len(zon.zones)} zones")

    # Parse IPL files
    from parsers.ipl_parser import IPLParser
    ipl_files = glob.glob("data/raw/map/maps/**/*.IPL", recursive=True)
    all_instances = []
    for ipl_path in ipl_files:
        ipl = IPLParser(ipl_path)
        ipl.parse()
        all_instances.extend([vars(i) for i in ipl.instances])
    with open("data/processed/ipl_instances.json", 'w') as f:
        json.dump({'instances': all_instances}, f)
    print(f"  ✓ Parsed {len(all_instances)} IPL instances from {len(ipl_files)} files")

    # Parse IDE for car/ped/weapon IDs
    from parsers.ide_parser import IDEParser
    if os.path.exists("data/raw/map/default.ide"):
        ide = IDEParser("data/raw/map/default.ide")
        ide.parse()
        ide.to_json("data/processed/ide_entries.json")
        print(f"  ✓ Parsed {len(ide.entries)} IDE entries")

    # Build map graph
    from spatial.map_graph import MapGraph
    zones_json = "data/processed/zones_db.json" if os.path.exists("data/processed/zones_db.json") else None
    ipl_json = "data/processed/ipl_instances.json" if os.path.exists("data/processed/ipl_instances.json") else None
    mg = MapGraph(zones_json=zones_json, ipl_json=ipl_json)
    mg.export_locations_json("data/processed/coords_db.json")
    print(f"  ✓ Built map graph with {len(mg.locations)} known locations")

    # Build training dataset
    from training.dataset_builder import DatasetBuilder
    builder = DatasetBuilder("data/processed/scm_parsed.json", mg)
    builder.save_jsonl("data/processed/training_pairs.jsonl")
    print(f"  ✓ Training dataset: {len(builder.pairs)} pairs")


def run_train():
    print("=" * 60)
    print("STEP 2: FINE-TUNING LLM")
    print("=" * 60)
    from training.finetune import train
    train()


def run_generate(num_missions: int, output_path: str):
    print("=" * 60)
    print("STEP 3: GENERATING NEW MAIN.SCM")
    print("=" * 60)
    os.makedirs(os.path.dirname(output_path) or "output", exist_ok=True)

    zones_json = "data/processed/zones_db.json" if os.path.exists("data/processed/zones_db.json") else None
    from spatial.map_graph import MapGraph
    mg = MapGraph(zones_json=zones_json)

    model_path = "models/gtavc_scm_lora"
    if not os.path.exists(model_path):
        print(f"[WARNING] Fine-tuned model not found at {model_path}")
        print("[INFO] Using template-based generation (no LLM) as fallback")
        _generate_template_based(num_missions, output_path, mg)
        return

    from generator.mission_generator import MissionGenerator
    gen = MissionGenerator(model_path=model_path, map_graph=mg)
    gen.generate_full_mod(num_missions=num_missions, output_path=output_path)


def _generate_template_based(num_missions: int, output_path: str, map_graph):
    """Fallback generator using templates without LLM"""
    from generator.mission_generator import SCMAssembler, MISSION_CONCEPTS
    import random

    concepts = random.sample(MISSION_CONCEPTS, min(num_missions, len(MISSION_CONCEPTS)))

    from spatial.map_graph import KNOWN_LOCATIONS
    trigger_scripts = []
    for i, concept in enumerate(concepts):
        loc = KNOWN_LOCATIONS.get(concept['start_location'], (83.0, -849.8, 9.3, ''))
        x, y, z = loc[0], loc[1], loc[2]
        idx = i + 2
        name = concept['name']
        trigger_scripts.append(f"""
:{name}
script_name '{name}'

:{name}_LOOP
wait $default_wait_time
if
  $passed_{name} == 1
goto_if_false @{name}_CHECK
terminate_this_script

:{name}_CHECK
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
HELP_2932()
load_and_launch_mission_internal {idx}
goto @{name}_LOOP
""")

    mission_bodies = []
    for concept in concepts:
        loc = KNOWN_LOCATIONS.get(concept['start_location'], (83.0, -849.8, 9.3, ''))
        tx, ty, tz = loc[0] + 10, loc[1] + 10, loc[2]
        name = concept['name']
        body = f"""
:M{name}
script_name 'M{name}'
$onmission = 1
set_fading_colour 0 0 0
do_fade 0 2000
wait 2000
do_fade 1 1000

// Objective: {concept['description']}
$obj_marker = create_marker 4 at {tx} {ty} {tz}

:M{name}_WAIT
wait 0
if
  not Player.Defined($player_char)
goto_if_false @M{name}_FAIL
if
  Actor.Dead($player_actor)
goto_if_false @M{name}_FAIL
if
  locate_player_on_foot_3d $player_char 0 {tx} {ty} {tz} radius 3.0 3.0 3.0
goto_if_false @M{name}_WAIT
goto @M{name}_PASS

:M{name}_FAIL
Blip.Remove($obj_marker)
$onmission = 0
print_now 'M_FAIL' time 3000 1
end_thread

:M{name}_PASS
Blip.Remove($obj_marker)
$passed_{name} = 1
$onmission = 0
add_score $player_char score 1000
print_now 'M_PASS' time 5000 1
end_thread
"""
        mission_bodies.append((name, body))

    assembler = SCMAssembler(
        missions=concepts,
        trigger_scripts=trigger_scripts,
        mission_bodies=mission_bodies,
        map_graph=map_graph
    )
    scm = assembler.assemble()
    with open(output_path, 'w') as f:
        f.write(scm)
    print(f"[Generator] Template-based SCM written to {output_path}")


def run_validate(input_path: str):
    print("=" * 60)
    print("STEP 4: VALIDATING SCM")
    print("=" * 60)
    from generator.validator import SCMValidator
    from spatial.map_graph import MapGraph

    mg = MapGraph()
    validator = SCMValidator(mg)

    with open(input_path) as f:
        scm_text = f.read()

    is_valid, issues = validator.validate(scm_text)
    if is_valid:
        print("  ✓ SCM is valid!")
    else:
        print("  ✗ SCM has issues:")
    for issue in issues:
        print(f"    - {issue}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTA VC SCM AI Pipeline")
    parser.add_argument('--mode', choices=['parse', 'train', 'generate', 'validate'],
                        required=True)
    parser.add_argument('--missions', type=int, default=5)
    parser.add_argument('--output', default='output/new_main.scm')
    parser.add_argument('--input', default='output/new_main.scm')
    args = parser.parse_args()

    if args.mode == 'parse':
        run_parse()
    elif args.mode == 'train':
        run_train()
    elif args.mode == 'generate':
        run_generate(args.missions, args.output)
    elif args.mode == 'validate':
        run_validate(args.input)
