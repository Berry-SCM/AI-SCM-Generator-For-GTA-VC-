"""
Builds fine-tuning dataset from parsed SCM scripts.
Creates (prompt, completion) pairs for LLM training.

Training pair formats:
1. Opcode teaching:    "What opcode creates a pickup at coordinates?" -> example
2. Mission structure:  "Write a mission that [desc]" -> full mission script
3. Script completion:  first-half -> second-half of real scripts
4. Coordinate pairs:   "Place something at [location]" -> SCM with correct coords
"""

import json
import random
from typing import List, Dict

MISSION_STRUCTURE_TEMPLATE = '''You are an expert GTA Vice City SCM script writer.
Write complete mission scripts using proper GTA VC opcodes and syntax.

Rules:
- Use locate_player_on_foot_3d for trigger zones
- Always check Player.Defined() and Player.Controllable() before triggering
- Use $onmission flag to prevent double-triggers
- Use print_big for mission name display
- Use load_and_launch_mission_internal to load a mission
- Coordinates must be valid Vice City world coords
- Use wait 500 in polling loops, wait 0 in tight active loops
- End trigger scripts with terminate_this_script when passed flag is set
- Handle mission failure (player death) with end_thread
- Use Blip.Remove() and cleanup before ending
'''

OPCODE_EXAMPLES = {
    'create_actor': {
        'prompt': 'How do I create an NPC actor in GTA VC SCM?',
        'completion': '''// Create an actor at position (pedtype 4=CIVMALE, model #MALE01)
$actor = Actor.Create(4, #MALE01, -378.5, -551.3, 18.2)
// Set actor facing angle
Actor.Angle($actor) = 180.0
// Give weapon (weapon_id 17 = AR15/M16, ammo 100)
give_weapon_to_char $actor weapon 17 ammo 100
// Set health
Actor.Health($actor) = 100'''
    },
    'create_car': {
        'prompt': 'How do I spawn a vehicle at a specific location in SCM?',
        'completion': '''// Spawn car at position (x, y, z) then set angle
$car = Car.Create(#TAXI, 200.0, -800.0, 10.5)
Car.Angle($car) = 90.0
// Or use car generator for persistent spawns that survive missions
create_car_generator $gen = init_car_generator #TAXI -1 -1 force_spawn 1 alarm 0 door_lock 0 min_delay 0 max_delay 10000 at 200.0 -800.0 10.5 angle 90.0
switch_car_generator $gen cars_to_generate_to -1'''
    },
    'mission_trigger': {
        'prompt': 'How do I create a mission trigger zone in SCM?',
        'completion': ''':MYMISSION_LOOP
wait 500
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
$onmission = 1
Player.MakeSafe($player_char)
print_big 'MY_MISS' 15000 ms 2
load_and_launch_mission_internal 5
goto @MYMISSION_LOOP'''
    },
    'pickup': {
        'prompt': 'How do I create a pickup (weapon, health, etc) in SCM?',
        'completion': '''// Create a weapon pickup (type 8 = on-street, respawns after pickup)
$pickup = Pickup.Create(#COLT45, 8, 200.0, -800.0, 10.5)
// Create a health pickup (type 3 = once only)
$hp_pickup = Pickup.Create(#HEALTH, 3, 200.0, -810.0, 10.5)
// Check if player picked it up (poll in loop)
if
  Pickup.Picked_up($pickup)
goto_if_false @NOT_PICKED
// do something on pickup
:NOT_PICKED'''
    },
    'marker': {
        'prompt': 'How do I add a blip/marker on the radar in SCM?',
        'completion': '''// Add blip for a coordinate
$blip = Blip.AddForCoord(200.0, -800.0, 10.5)
// Or add sprite blip (icon 27 = yellow star)
add_sprite_blip_for_coord $blip2 = create_marker 27 at 200.0 -800.0 10.5
// Hide marker until needed
Marker.Disable($blip)
// Remove marker when done
Blip.Remove($blip)'''
    },
    'timer': {
        'prompt': 'How do I use timers in GTA VC SCM?',
        'completion': '''// TIMERA and TIMERB are built-in millisecond counters
TIMERA = 0

:TIMER_LOOP
if
  TIMERA <= 30000
goto_if_false @TIMER_EXPIRED
wait 0
goto @TIMER_LOOP

:TIMER_EXPIRED
// 30 seconds have elapsed
print_now 'TIME_UP' time 3000 1'''
    },
    'camera': {
        'prompt': 'How do I control the camera in a GTA VC mission cutscene?',
        'completion': '''// Fade screen to black
set_fading_colour 0 0 0
do_fade 0 1500
wait 1500
// Position camera
Camera.SetPosition(-378.5, -551.3, 25.0, 0.0, 0.0, 0.0)
// Point camera at a target
Camera.PointAt(-378.5, -560.0, 20.0, 2)
// Enable widescreen bars
switch_widescreen 1
wait 3000
// Restore player camera
Camera.Restore()
switch_widescreen 0
// Fade back in
do_fade 1 1000'''
    },
    'save_system': {
        'prompt': 'How does the save/property system work in GTA VC SCM?',
        'completion': '''// Properties use asset pickups; $passed_MISSION flags track completion.
// Declare the mission flag at global scope:
declare_mission_flag $onmission

// Mark a mission as passed:
$passed_MYMISSION = 1
$onmission = 0

// Trigger scripts check the flag to know if already completed:
if
  $passed_MYMISSION == 1
goto_if_false @CHECK_TRIGGER
terminate_this_script  // already done, stop polling
:CHECK_TRIGGER

// Track progress:
player_made_progress 1'''
    },
}


class DatasetBuilder:
    def __init__(self, scm_json_path: str, map_graph=None):
        with open(scm_json_path) as f:
            self.scm_data = json.load(f)
        self.map_graph = map_graph
        self.pairs: List[Dict] = []

    def build_opcode_pairs(self):
        """Add opcode teaching examples to training set."""
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
        """Convert parsed SCM scripts into training pairs."""
        scripts = self.scm_data.get('scripts', [])
        missions = {m['label']: m for m in self.scm_data.get('missions', [])}

        count_before = len(self.pairs)
        for script in scripts:
            raw_lines = script.get('raw', [])
            if len(raw_lines) < 5:
                continue

            script_text = '\n'.join(raw_lines)
            mission_name = script.get('name', script.get('label', 'UNKNOWN'))
            mission_info = missions.get(mission_name, {})
            display_name = mission_info.get('name', mission_name)

            # Pair 1: Full script reproduction
            prompt = (f"Write the SCM script for '{display_name}' "
                      f"(script name: {mission_name}) in GTA Vice City. "
                      f"Include proper trigger detection, mission flag checking, "
                      f"and mission launch.")
            self.pairs.append({
                'type': 'script_reproduction',
                'messages': [
                    {'role': 'system', 'content': MISSION_STRUCTURE_TEMPLATE},
                    {'role': 'user', 'content': prompt},
                    {'role': 'assistant', 'content': script_text}
                ]
            })

            # Pair 2: Script completion (first half → second half)
            if len(raw_lines) > 20:
                half = len(raw_lines) // 2
                first_half = '\n'.join(raw_lines[:half])
                second_half = '\n'.join(raw_lines[half:])
                self.pairs.append({
                    'type': 'script_completion',
                    'messages': [
                        {'role': 'system', 'content': MISSION_STRUCTURE_TEMPLATE},
                        {'role': 'user',
                         'content': f"Complete this GTA VC SCM script:\n\n{first_half}"},
                        {'role': 'assistant', 'content': second_half}
                    ]
                })

        added = len(self.pairs) - count_before
        print(f"[DatasetBuilder] Added {added} script pairs from {len(scripts)} scripts")

    def build_coord_pairs(self):
        """
        Build coordinate-to-location training pairs.
        Uses live MapGraph locations if available (includes IPL-enriched data),
        otherwise falls back to the hardcoded KNOWN_LOCATIONS table.
        """
        from spatial.map_graph import KNOWN_LOCATIONS

        # Build location dict: prefer MapGraph (has zone + IPL data), else hardcoded
        if self.map_graph is not None:
            location_items = [
                (loc.name, loc.x, loc.y, loc.z, loc.description, loc.zone)
                for loc in self.map_graph.locations.values()
                if loc.location_type in ('mission_trigger', 'business', 'gang_turf')
            ]
        else:
            location_items = [
                (name, x, y, z, desc, '')
                for name, (x, y, z, desc) in KNOWN_LOCATIONS.items()
            ]

        count_before = len(self.pairs)
        for name, x, y, z, desc, zone in location_items:
            human_name = name.replace('_', ' ')
            self.pairs.append({
                'type': 'coord_knowledge',
                'messages': [
                    {'role': 'user',
                     'content': f'What are the coordinates for {human_name} in GTA Vice City SCM?'},
                    {'role': 'assistant',
                     'content': (
                         f'The {human_name} location ({desc}) is at '
                         f'X={x}, Y={y}, Z={z}'
                         + (f', zone: {zone}' if zone else '') + '.\n'
                         f'In SCM: locate_player_on_foot_3d $player_char 0 '
                         f'{x} {y} {z} radius 2.0 2.0 2.0'
                     )}
                ]
            })
        print(f"[DatasetBuilder] Added {len(self.pairs) - count_before} coord pairs")

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
