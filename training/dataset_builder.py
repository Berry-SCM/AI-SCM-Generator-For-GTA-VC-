"""
Builds fine-tuning dataset from parsed SCM JSON + map data.
Corrected: loads opcode/vehicle/ped/weapon ID data from processed JSON files
produced by training/opcode_scraper.py.
"""
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Optional

MISSION_STRUCTURE_TEMPLATE = '''You are an expert GTA Vice City SCM script writer.
Given a mission concept, write a complete, valid SCM mission script.
Use correct opcodes, real Vice City coordinates, and proper label/goto structure.
Always end missions with terminate_this_script.
Always reset $onmission = 0 on both pass and fail paths.
'''

# Fallback hardcoded examples (used if opcodes.json not found)
OPCODE_EXAMPLES = {
    "wait": {
        "description": "Wait N milliseconds before continuing",
        "example": "wait 2000  // wait 2 seconds"
    },
    "goto": {
        "description": "Jump unconditionally to a label",
        "example": "goto @MISSION_LOOP"
    },
    "goto_if_false": {
        "description": "Jump to label if the last condition was false",
        "example": "if\n  Player.Defined($player_char)\ngoto_if_false @FAIL"
    },
    "terminate_this_script": {
        "description": "End the current script thread",
        "example": "terminate_this_script"
    },
    "load_and_launch_mission_internal": {
        "description": "Load and start a mission by index",
        "example": "load_and_launch_mission_internal 5  // start mission 5"
    },
    "Actor.Create": {
        "description": "Spawn an actor (ped) at coordinates",
        "example": "$enemy = Actor.Create(4, #WMYCR, 200.0, -800.0, 10.5)"
    },
    "Car.Create": {
        "description": "Spawn a vehicle at coordinates",
        "example": "$getaway = Car.Create(#SENTINEL, 210.0, -810.0, 10.5)"
    },
    "locate_player_on_foot_3d": {
        "description": "Check if player is near a 3D coordinate on foot",
        "example": "locate_player_on_foot_3d $player_char 0 200.0 -800.0 10.5 radius 3.0 3.0 3.0"
    },
    "create_marker": {
        "description": "Create a visible marker sphere at coordinates",
        "example": "$marker = create_marker 4 at 200.0 -800.0 10.5"
    },
    "give_player_weapon": {
        "description": "Give the player a weapon with ammo",
        "example": "give_player_weapon $player_char 274 100  // colt45, 100 ammo"
    },
}


class DatasetBuilder:
    def __init__(self, scm_json_path: str, map_graph=None):
        self.scm_json_path = scm_json_path
        self.map_graph = map_graph
        self.scm_data: Dict = {}
        self.pairs: List[Dict] = []

        # Load parsed SCM data
        if os.path.exists(scm_json_path):
            with open(scm_json_path) as f:
                self.scm_data = json.load(f)
            print(f"[DatasetBuilder] Loaded SCM data: {len(self.scm_data.get('scripts', []))} scripts")
        else:
            print(f"[DatasetBuilder] Warning: SCM JSON not found at {scm_json_path}")

        # Load ID reference data (produced by opcode_scraper.py)
        processed_dir = "data/processed"
        self.vehicle_ids: Dict = {}
        self.ped_ids: Dict = {}
        self.weapon_ids: Dict = {}
        self.opcodes: Dict = {}

        for fname, attr in [
            ("vehicle_ids.json", "vehicle_ids"),
            ("ped_ids.json", "ped_ids"),
            ("weapon_ids.json", "weapon_ids"),
            ("opcodes.json", "opcodes"),
        ]:
            fpath = os.path.join(processed_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    setattr(self, attr, json.load(f))
                print(f"[DatasetBuilder] Loaded {fpath}")
            else:
                print(f"[DatasetBuilder] Warning: {fpath} not found — run training/opcode_scraper.py first")

    def build_opcode_pairs(self) -> List[Dict]:
        """Build Q&A pairs for opcode usage."""
        pairs = []

        # From hardcoded OPCODE_EXAMPLES
        for opcode_name, data in OPCODE_EXAMPLES.items():
            pairs.append({
                "messages": [
                    {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                    {"role": "user", "content": f"How do I use the '{opcode_name}' opcode in GTA VC SCM?"},
                    {"role": "assistant", "content": f"{data['description']}\n\nExample:\n{data['example']}"}
                ]
            })

        # From scraped opcodes.json (if available)
        for opcode_hex, data in self.opcodes.items():
            name = data.get("name", opcode_hex)
            desc = data.get("desc", "")
            args = data.get("args", "")
            pairs.append({
                "messages": [
                    {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                    {"role": "user", "content": f"What does opcode {opcode_hex} ({name}) do in GTA VC SCM?"},
                    {"role": "assistant", "content": f"Opcode {opcode_hex}: {name}\nArguments: {args}\n{desc}"}
                ]
            })

        # Vehicle ID pairs (from vehicle_ids.json)
        for vid, vname in self.vehicle_ids.items():
            pairs.append({
                "messages": [
                    {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                    {"role": "user", "content": f"How do I spawn a {vname} in a VC SCM mission?"},
                    {"role": "assistant", "content": f"$car = Car.Create(#{vname.upper()}, x, y, z)  // vehicle ID {vid}"}
                ]
            })

        # Weapon ID pairs (from weapon_ids.json)
        for wid, wname in self.weapon_ids.items():
            pairs.append({
                "messages": [
                    {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                    {"role": "user", "content": f"Give the player a {wname} in SCM."},
                    {"role": "assistant", "content": f"give_player_weapon $player_char {wid} 100  // {wname}"}
                ]
            })

        self.pairs.extend(pairs)
        return pairs

    def build_script_pairs(self) -> List[Dict]:
        """Build training pairs from actual parsed SCM scripts."""
        pairs = []
        scripts = self.scm_data.get("scripts", [])

        for script in scripts:
            name = script.get("name", "UNKNOWN")
            label = script.get("label", "")
            instructions = script.get("instructions", [])
            mission_idx = script.get("mission_index")

            if len(instructions) < 3:
                continue

            # Build raw script text
            script_lines = []
            for instr in instructions[:60]:  # cap to avoid token overflow
                raw = instr.get("raw", "").strip()
                if raw:
                    script_lines.append(raw)

            if not script_lines:
                continue

            script_text = "\n".join(script_lines)

            # Training pair: explain what the script does
            pairs.append({
                "messages": [
                    {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                    {"role": "user", "content": f"Explain this GTA VC SCM script section '{name}':"},
                    {"role": "assistant", "content": f"This is the '{name}' script (label @{label}).\n\n```\n{script_text}\n```"}
                ]
            })

            # If it's a mission, ask to generate similar
            if mission_idx is not None:
                coords = script.get("coords_used", [])
                coord_str = ""
                if coords:
                    cx, cy, cz = coords[0]
                    coord_str = f" near ({cx}, {cy}, {cz})"
                pairs.append({
                    "messages": [
                        {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                        {"role": "user", "content": f"Write a GTA VC SCM mission similar to mission {mission_idx} ('{name}'){coord_str}."},
                        {"role": "assistant", "content": f":{name}\nscript_name '{name[:8]}'\n$onmission = 1\n\n{script_text}\n\nterminate_this_script"}
                    ]
                })

        self.pairs.extend(pairs)
        return pairs

    def build_coord_pairs(self) -> List[Dict]:
        """Build training pairs for coordinate-aware placement."""
        pairs = []

        if self.map_graph is None:
            return pairs

        # Zone-based pairs
        zone_examples = [
            ("OCEAN_BEACH", 400.0, -800.0, 10.5),
            ("WASHINGTON", 100.0, -600.0, 10.5),
            ("LITTLE_HAITI", -900.0, 200.0, 10.5),
            ("AIRPORT", -1700.0, -100.0, 14.0),
            ("DOCKS", -1100.0, -1400.0, 11.5),
            ("VICE_POINT", 500.0, 200.0, 10.5),
            ("DOWNTOWN", 200.0, 900.0, 10.5),
        ]

        for zone_name, x, y, z in zone_examples:
            pairs.append({
                "messages": [
                    {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                    {"role": "user", "content": f"Give me a valid outdoor spawn coordinate in {zone_name} for a mission trigger."},
                    {"role": "assistant", "content": f"In {zone_name}, use approximately ({x}, {y}, {z}).\nSCM: $marker = create_marker 4 at {x} {y} {z}"}
                ]
            })

        # Known location pairs
        try:
            locations = self.map_graph.get_locations_by_type('mission_trigger')
            for loc in locations[:20]:
                pairs.append({
                    "messages": [
                        {"role": "system", "content": MISSION_STRUCTURE_TEMPLATE},
                        {"role": "user", "content": f"Where should I place a mission trigger near {loc.description}?"},
                        {"role": "assistant", "content": f"Place the trigger at ({loc.x}, {loc.y}, {loc.z}) in zone {loc.zone}.\nSCM:\n$marker = create_marker 4 at {loc.x} {loc.y} {loc.z}"}
                    ]
                })
        except Exception:
            pass

        self.pairs.extend(pairs)
        return pairs

    def build_all(self) -> List[Dict]:
        """Build all training pairs."""
        print("[DatasetBuilder] Building opcode pairs...")
        self.build_opcode_pairs()
        print("[DatasetBuilder] Building script pairs...")
        self.build_script_pairs()
        print("[DatasetBuilder] Building coord pairs...")
        self.build_coord_pairs()

        # Also load pre-built ID pairs from opcode_scraper if they exist
        id_pairs_path = "data/processed/id_training_pairs.jsonl"
        if os.path.exists(id_pairs_path):
            with open(id_pairs_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.pairs.append(json.loads(line))
            print(f"[DatasetBuilder] Loaded ID training pairs from {id_pairs_path}")

        print(f"[DatasetBuilder] Total pairs: {len(self.pairs)}")
        return self.pairs

    def save_jsonl(self, out_path: str):
        """Save training pairs to JSONL file."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for pair in self.pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"[DatasetBuilder] Saved {len(self.pairs)} pairs → {out_path}")
