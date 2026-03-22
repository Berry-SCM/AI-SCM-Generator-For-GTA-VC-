"""
Builds fine-tuning dataset from parsed SCM scripts.
Creates (prompt, completion) pairs for LLM training.

Training pair formats:
1. Opcode teaching: "What opcode creates a pickup at coordinates?" → example
2. Mission structure: "Write a mission that [desc]" → full mission script
3. Coordinate usage: "Place a car at the docks" → SCM with correct coords
4. Script patterns: Fill-in-the-blank SCM completion
"""

import json
import random
from typing import List, Dict, Tuple
from parsers.scm_parser import SCMParser, SCMScript

MISSION_STRUCTURE_TEMPLATE = '''You are an expert GTA Vice City SCM script writer.
Write a complete mission script using proper GTA VC opcodes and syntax.

Rules:
- Use locate_player_on_foot_3d for trigger zones
- Always check Player.Defined() and Player.Controllable() before triggering
- Use $onmission flag to prevent double-triggers  
- Use print_big for mission name, load_and_launch_mission_internal for loading
- Coordinates must be valid Vice City world coords
- Use wait $default_wait_time in loops
- End scripts with terminate_this_script when passed flag is set
'''

OPCODE_EXAMPLES = {
    'create_actor': {
        'prompt': 'How do I create an NPC actor in GTA VC SCM?',
        'completion': '''// Create an actor at position
$actor = Actor.Create(4, #MALE01, -378.5, -551.3, 18.2)
// Set actor angle
Actor.Angle($actor) = 180.0
// Give weapon (weapon_id, ammo)
give_weapon_to_char $actor weapon 17 ammo 100
// Set health
Actor.Health($actor) = 100'''
    },
    'create_car': {
        'prompt': 'How do I spawn a vehicle at a specific location in SCM?',
        'completion': '''// Spawn car at position (x, y, z, angle)
$car = Car.Create(#TAXI, 200.0, -800.0, 10.5)
Car.Angle($car) = 90.0
// Or use car generator for persistent spawns
create_car_generator $gen = init_car_generator #TAXI -1 -1 force_spawn 1 alarm 0 door_lock 0 min_delay 0 max_delay 10000 at 200.0 -800.0 10.5 angle 90.0
switch_car_generator $gen cars_to_generate_to -1'''
    },
    'mission_trigger': {
        'prompt': 'How do I create a mission trigger zone in SCM?',
        'completion': ''':MYMISSION_LOOP
wait $default_wait_time
if
  $passed_MYMISSION == 1
goto_if_false @MYMISSION_CHECK
terminate_this_script

:MYMISSION_CHECK
if
  Player.Defined($player_char)
goto_if_false @MYMISSION_LOOP
if
  locate_player_on_foot_3d $player_char 0 200.0 -800.0 10.5 radius 2.0 2.0 2.0
goto_if_false @MYMISSION_LOOP
if
  $onmission == 0
goto_if_false @MYMISSION_LOOP
if
  Player.Controllable($player_char)
goto_if_false @MYMISSION_LOOP
HELP_2883()
print_big 'MY_MISS' 15000 ms 2
HELP_2932()
load_and_launch_mission_internal 5
goto @MYMISSION_LOOP'''
    },
    'pickup': {
        'prompt': 'How do I create a pickup (weapon, health, etc) in SCM?',
        'completion': '''// Create a weapon pickup (type 8 = on-street, respawns)
$pickup = Pickup.Create(#COLT45, 8, 200.0, -800.0, 10.5)
// Create a health pickup
$hp_pickup = Pickup.Create(#HEALTH, 3, 200.0, -810.0, 10.5)
// Check if player picked it up
if
  Pickup.Picked_up($pickup)
goto_if_false @NOT_PICKED
// do something
:NOT_PICKED'''
    },
    'marker': {
        'prompt': 'How do I add a blip/marker on the radar in SCM?',
        'completion': '''// Add blip for a coordinate
$blip = Blip.AddForCoord(200.0, -800.0, 10.5)
// Or add sprite blip
add_sprite_blip_for_coord $blip2 = create_marker 27 at 200.0 -800.0 10.5
// Disable/hide marker
Marker.Disable($blip)
// Remove marker
Blip.Remove($blip)'''
    },
    'timer': {
        'prompt': 'How do I use timers in GTA VC SCM?',
        'completion': '''// TIMERA and TIMERB are built-in millisecond timers
TIMERA = 0

:TIMER_LOOP
if
  TIMERA <= 30000
goto_if_false @TIMER_EXPIRED
wait 0
goto @TIMER_LOOP

:TIMER_EXPIRED
// 30 seconds have passed
print_now 'TIME_UP' time 3000 1'''
    },
    'camera': {
        'prompt': 'How do I control the camera in a GTA VC mission cutscene?',
        'completion': '''// Fade out
set_fading_colour 0 0 0
do_fade 0 1500
// Set camera position  
Camera.SetPosition(-378.5, -551.3, 25.0, 0.0, 0.0, 0.0)
// Point camera at target
Camera.PointAt(-378.5, -560.0, 20.0, 2)
// Enable widescreen
switch_widescreen 1
// Restore player camera
Camera.Restore()
// Fade back in
do_fade 1 1000'''
    },
}

class DatasetBuilder:
    def __init__(self, scm_json_path: str, map_graph=None):
        with open(scm_json_path) as f:
            self.scm_data = json.load(f)
        self.map_graph = map_graph
        self.pairs: List[Dict] = []

    def build_opcode_pairs(self):
        """Add opcode teaching examples"""
        for key, example in OPCODE_EXAMPLES.items():
            self.pairs.append({
                'type': 'opcode_teaching',
                'messages': [
                    {'role': 'system', 'content': MISSION_STRUCTURE_TEMPLATE},
                    {'role': 'user', 'content': example['prompt']},
                    {'role': 'assistant', 'content': example['completion']}
                ]
            })
        print(f"[DatasetBuilder] Added {len(OPCODE_EXAMPLES)} opcode pairs")

    def build_script_pairs(self):
        """Convert parsed SCM scripts into training pairs"""
        scripts = self.scm_data.get('scripts', [])
        missions = {m['label']: m for m in self.scm_data.get('missions', [])}

        for script in scripts:
            raw_lines = script.get('raw', [])
            if len(raw_lines) < 5:
                continue

            script_text = '\n'.join(raw_lines)
            mission_name = script.get('name', script.get('label', 'UNKNOWN'))

            # Find matching mission info
            mission_info = missions.get(mission_name, {})
            display_name = mission_info.get('name', mission_name)

            # Pair 1: Full script reproduction
            prompt = f"Write the SCM script for the '{display_name}' script (label: {mission_name}) in GTA Vice City. Include proper trigger detection, mission flag checking, and mission launch."
            self.pairs.append({
                'type': 'script_reproduction',
                'messages': [
                    {'role': 'system', 'content': MISSION_STRUCTURE_TEMPLATE},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': script_text}
                ]
            })

            # Pair 2: Script completion (give first half, complete second)
            if len(raw_lines) > 20:
                half = len(raw_lines) // 2
                first_half = '\n'.join(raw_lines[:half])
                second_half = '\n'.join(raw_lines[half:])
                self.pairs.append({
                    'type': 'script_completion',
                    'messages': [
                        {'role': 'system', 'content': MISSION_STRUCTURE_TEMPLATE},
                        {'role': 'user', 'content': f"Complete this GTA VC SCM script:\n\n{first_half}"},
                        {'role': 'assistant', 'content': second_half}
                    ]
                })

        print(f"[DatasetBuilder] Added {len(scripts)} script pairs")

    def build_coord_pairs(self):
        """Build coordinate-to-location pairs"""
        for name, data in {**{k: v for k,v in {
            'hotel': (83.0, -849.8, 9.3),
            'mansion': (-378.5, -551.3, 18.2),
            'docks': (-685.8, -1495.6, 12.5),
            'airport': (-1720.3, -239.6, 14.8),
            'downtown': (-665.63, 1231.863, 10.1),
        }.items()}}.items():
            x, y, z = data
            self.pairs.append({
                'type': 'coord_knowledge',
                'messages': [
                    {'role': 'user', 'content': f'What are the coordinates for {name} in GTA Vice City SCM?'},
                    {'role': 'assistant', 'content': f'The {name} location in Vice City is at coordinates X={x}, Y={y}, Z={z}.\nIn SCM: locate_player_on_foot_3d $player_char 0 {x} {y} {z} radius 2.0 2.0 2.0'}
                ]
            })

    def build_all(self) -> List[Dict]:
        self.build_opcode_pairs()
        self.build_script_pairs()
        self.build_coord_pairs()
        return self.pairs

    def save_jsonl(self, out_path: str):
        self.build_all()
        with open(out_path, 'w') as f:
            for pair in self.pairs:
                f.write(json.dumps(pair) + '\n')
        print(f"[DatasetBuilder] Saved {len(self.pairs)} pairs to {out_path}")