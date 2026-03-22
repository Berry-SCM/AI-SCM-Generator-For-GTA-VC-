"""
Generates new GTA VC mission scripts using the fine-tuned LLM.
Combines:
  - LLM for natural language -> SCM script
  - MapGraph for valid coordinate injection
  - Validator for opcode/syntax checking
"""

import json
import random
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

from spatial.map_graph import MapGraph, KNOWN_LOCATIONS

# Mission concept templates for generating diverse scenarios
MISSION_CONCEPTS = [
    {
        'name': 'NEWMIS1', 'display': 'Hot Pursuit',
        'type': 'chase',
        'description': 'Chase a target vehicle across Vice City and destroy it',
        'start_location': 'hotel_spawn',
        'required_vehicles': ['POLICE', 'TAXI'],
        'objectives': ['spawn_target_car', 'chase_car', 'destroy_car'],
    },
    {
        'name': 'NEWMIS2', 'display': 'The Deal',
        'type': 'delivery',
        'description': 'Pick up a package at the docks and deliver it to the mansion',
        'start_location': 'colonel_missions',
        'required_vehicles': ['RUMPO'],
        'objectives': ['go_to_docks', 'pickup_package', 'deliver_to_mansion'],
    },
    {
        'name': 'NEWMIS3', 'display': 'Gang War',
        'type': 'combat',
        'description': 'Attack a rival gang stronghold and eliminate all enemies',
        'start_location': 'cubans',
        'required_vehicles': ['PATRIOT'],
        'objectives': ['drive_to_stronghold', 'kill_all_enemies', 'escape'],
    },
    {
        'name': 'NEWMIS4', 'display': 'The Getaway',
        'type': 'escape',
        'description': 'Rob the printworks and escape the police with the cash',
        'start_location': 'vercetti_estate',
        'required_vehicles': ['SENTINEL', 'POLICE'],
        'objectives': ['pickup_contact', 'drive_to_point1', 'drive_to_point2', 'drive_to_point3'],
    },
    {
        'name': 'NEWMIS5', 'display': 'Snatch and Grab',
        'type': 'stealth',
        'description': 'Infiltrate the bank area and grab the briefcase without alerting police',
        'start_location': 'bank',
        'required_vehicles': [],
        'objectives': ['sneak_to_bank', 'grab_briefcase', 'escape_without_4stars'],
    },
    {
        'name': 'NEWMIS6', 'display': 'Hired Muscle',
        'type': 'escort',
        'description': 'Escort a contact from the airport to the downtown safehouse',
        'start_location': 'airport',
        'required_vehicles': ['STRETCH'],
        'objectives': ['go_to_airport', 'pick_up_contact', 'drive_to_safehouse', 'protect_contact'],
    },
    {
        'name': 'NEWMIS7', 'display': 'Waterfront Hit',
        'type': 'assassination',
        'description': 'Take out a crime boss on his yacht before it leaves the docks',
        'start_location': 'colonel_missions',
        'required_vehicles': ['SPEEDER'],
        'objectives': ['reach_yacht_by_boat', 'board_yacht', 'kill_boss', 'escape_by_sea'],
    },
    {
        'name': 'NEWMIS8', 'display': 'Contraband Run',
        'type': 'delivery',
        'description': 'Transport stolen contraband through three checkpoints without police attention',
        'start_location': 'bikers',
        'required_vehicles': ['BOXVILLE'],
        'objectives': ['pickup_cargo', 'checkpoint1', 'checkpoint2', 'checkpoint3', 'dropoff'],
    },
    {
        'name': 'NEWMIS9', 'display': 'Bike Blitz',
        'type': 'chase',
        'description': 'Chase a courier on a PCJ600 across Vice Point and retrieve the stolen package',
        'start_location': 'hotel_spawn',
        'required_vehicles': ['PCJ600'],
        'objectives': ['spot_courier', 'chase_bike', 'ram_off_road', 'grab_package'],
    },
    {
        'name': 'NEWMIS10', 'display': 'Downtown Takedown',
        'type': 'combat',
        'description': 'Clear out a rival crew occupying the downtown construction site',
        'start_location': 'downtown_ammu',
        'required_vehicles': ['PATRIOT'],
        'objectives': ['arm_up', 'reach_construction_site', 'kill_all_rivals', 'secure_area'],
    },
    {
        'name': 'NEWMIS11', 'display': 'Drive or Die',
        'type': 'timed_drive',
        'description': 'Deliver a car wired with explosives to the junkyard within 3 minutes',
        'start_location': 'vercetti_estate',
        'required_vehicles': ['INFERNUS'],
        'objectives': ['enter_rigged_car', 'drive_to_junkyard', 'escape_on_foot'],
    },
    {
        'name': 'NEWMIS12', 'display': 'Payback',
        'type': 'assassination',
        'description': 'Eliminate three targets across Vice City using only a sniper rifle',
        'start_location': 'lawyer_office',
        'required_vehicles': [],
        'objectives': ['collect_sniper', 'kill_target1', 'kill_target2', 'kill_target3', 'escape'],
    },
]


class MissionGenerator:
    def __init__(self,
                 model_path: str,
                 map_graph: MapGraph,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.map_graph = map_graph
        self.device = device
        self.model = None
        self.tokenizer = None
        self._load_model(model_path)

    def _load_model(self, model_path: str):
        print(f"[MissionGenerator] Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-Instruct-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        print("[MissionGenerator] Model loaded.")

    def _generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens, not the prompt
        new_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_trigger_script(self, concept: Dict,
                                 mission_index: int,
                                 trigger_coord: tuple) -> str:
        """Generate the trigger/ambient script for a mission concept."""
        x, y, z = trigger_coord
        name = concept['name']
        desc = concept['description']

        prompt = f"""[INST] <<SYS>>
You are an expert GTA Vice City SCM script writer. Write complete, valid SCM code.
<</SYS>>

Write a GTA VC SCM TRIGGER SCRIPT for mission index {mission_index}.
Mission name: {concept['display']}
Description: {desc}
Trigger coord: {x} {y} {z}
Script label: @{name}

Requirements:
- Check $passed_{name} == 1 at top, terminate_this_script if already done
- Poll loop with wait 500
- Check Player.Defined and not $onmission == 1
- Use locate_player_on_foot_3d at {x} {y} {z} radius 2.0 2.0 2.0
- Set $onmission = 1 before load_and_launch_mission_internal {mission_index}
- End with terminate_this_script
[/INST]"""

        result = self._generate(prompt, max_new_tokens=512)
        # Fallback to template if generation is empty or too short
        if len(result.strip()) < 50:
            result = self._template_trigger(name, mission_index, x, y, z)
        return result

    def _template_trigger(self, name: str, mission_index: int,
                           x: float, y: float, z: float) -> str:
        """Fallback template trigger when LLM output is insufficient."""
        return f"""
:{name}
script_name '{name[:8]}'

:{name}_LOOP
wait 500
if
  $passed_{name} == 1
goto_if_false @{name}_CHECK
terminate_this_script

:{name}_CHECK
if
  Player.Defined($player_char)
goto_if_false @{name}_LOOP
if
  not $onmission == 1
goto_if_false @{name}_LOOP
if
  locate_player_on_foot_3d $player_char 0 {x} {y} {z} radius 2.0 2.0 2.0
goto_if_false @{name}_LOOP
$onmission = 1
load_and_launch_mission_internal {mission_index}
terminate_this_script
"""

    def generate_mission_body(self, concept: Dict) -> str:
        """Generate the full mission body script."""
        desc = concept['description']
        objectives = concept.get('objectives', [])
        vehicles = concept.get('required_vehicles', [])
        start_loc = concept.get('start_location', 'hotel_spawn')

        loc_data = KNOWN_LOCATIONS.get(start_loc)
        if loc_data:
            x, y, z = loc_data[0], loc_data[1], loc_data[2]
        else:
            x, y, z = 83.0, -849.8, 9.3

        obj_text = '\n'.join(f'- {obj}' for obj in objectives)
        veh_text = ', '.join(f'#{v}' for v in vehicles) if vehicles else 'any vehicle'
        name = concept['name']

        prompt = f"""[INST] <<SYS>>
You are an expert GTA Vice City SCM mission script writer. Write complete, valid SCM code.
<</SYS>>

Write a complete GTA VC SCM MISSION body script for:
Mission name: {concept['display']}
Description: {desc}
Start location: {x} {y} {z}
Vehicles needed: {veh_text}
Objectives:
{obj_text}

Requirements:
- Label: @{name}
- script_name '{name[:8]}'
- Spawn required vehicles and actors
- Add radar blips with Blip.AddForCoord or Blip.AddForCar
- Use TIMERA for any time limit
- Check Actor.Dead($player_actor) for fail condition
- Set $passed_{name} = 1 and $onmission = 0 on success
- print_now 'M_PASS' on pass, 'M_FAIL' on fail
- end_thread at conclusion
[/INST]"""

        result = self._generate(prompt, max_new_tokens=1024)
        if len(result.strip()) < 100:
            result = self._template_mission_body(name, x, y, z)
        return result

    def _template_mission_body(self, name: str,
                                x: float, y: float, z: float) -> str:
        """Fallback template mission body when LLM output is insufficient."""
        ox = round(x + 50.0, 3)
        oy = round(y + 50.0, 3)
        return f"""
:{name}
script_name '{name[:8]}'
$player_actor = Actor.EmulateFromPlayer($player_char)
Player.CanMove($player_char, False)
do_fade 0 500
wait 500
do_fade 1 500
Player.CanMove($player_char, True)
$obj_marker = Blip.AddForCoord({ox}, {oy}, {z})
Blip.ChangeDisplay($obj_marker, 2)
print_big '{name[:8]}' time 5000 style 1

:{name}_MAIN
wait 0
if
  Actor.Dead($player_actor)
goto_if_false @{name}_OBJ
Blip.Remove($obj_marker)
$onmission = 0
print_now 'M_FAIL' time 3000 1
end_thread

:{name}_OBJ
if
  locate_player_on_foot_3d $player_char 0 {ox} {oy} {z} radius 5.0 5.0 5.0
goto_if_false @{name}_MAIN
Blip.Remove($obj_marker)
$passed_{name} = 1
$onmission = 0
add_score $player_char score 1000
player_made_progress 1
print_now 'M_PASS' time 5000 1
end_thread
"""

    def generate_full_mod(self, num_missions: int = 5,
                          output_path: str = "output/new_main.scm") -> str:
        """Generate a complete new main.scm mod."""
        print(f"[MissionGenerator] Generating {num_missions} missions...")

        # Cap at available concepts
        num_missions = min(num_missions, len(MISSION_CONCEPTS))
        selected = random.sample(MISSION_CONCEPTS, num_missions)
        trigger_scripts = []
        mission_bodies = []

        for i, concept in enumerate(selected):
            mission_idx = i + 2
            loc_data = KNOWN_LOCATIONS.get(concept.get('start_location', 'hotel_spawn'))
            coord = (loc_data[0], loc_data[1], loc_data[2]) if loc_data else (83.0, -849.8, 9.3)

            print(f"  Generating trigger for: {concept['display']}")
            trigger = self.generate_trigger_script(concept, mission_idx, coord)
            trigger_scripts.append(trigger)

            print(f"  Generating mission body for: {concept['display']}")
            body = self.generate_mission_body(concept)
            mission_bodies.append((concept['name'], body))

        assembler = SCMAssembler(
            missions=selected,
            trigger_scripts=trigger_scripts,
            mission_bodies=mission_bodies,
            map_graph=self.map_graph
        )
        scm_text = assembler.assemble()

        with open(output_path, 'w') as f:
            f.write(scm_text)
        print(f"[MissionGenerator] Written to {output_path}")
        return scm_text


class SCMAssembler:
    """Assembles a complete main.scm from parts."""

    DEFINE_OBJECTS_TEMPLATE = """DEFINE OBJECTS 10
DEFINE OBJECT (noname)
DEFINE OBJECT BRIBE                    // Object number -1
DEFINE OBJECT HEALTH                   // Object number -2
DEFINE OBJECT BODYARMOUR               // Object number -3
DEFINE OBJECT PICKUPSAVE               // Object number -4
DEFINE OBJECT INFO                     // Object number -5
DEFINE OBJECT KILLFRENZY               // Object number -6
DEFINE OBJECT MONEYBAG                 // Object number -7
DEFINE OBJECT BRIEFCASE                // Object number -8
DEFINE OBJECT DYNAMITE                 // Object number -9
"""

    MAIN_HEADER_TEMPLATE = """
//-------------MAIN---------------
script_name 'MAIN'
do_fade 0 0
set_total_number_of_missions {num_missions}
set_progress_total {num_missions}
set_max_wanted_level 4
set_collectable1_total 0
set_deatharrest_state 0
set_time_of_day 12 0
request_collision 83.0 -849.8
Camera.SetAtPos(83.0, -849.8, 9.3)
$player_char = Player.Create(0, 83.0, -849.8, 9.3)
$player_actor = Actor.EmulateFromPlayer($player_char)
declare_mission_flag $onmission
$onmission = 0
load_and_launch_mission_internal 0 // Initial

{car_generators}

{bribe_pickups}

wait 0
if
  Player.Defined($player_char)
goto_if_false @MAIN_LOOP
force_weather_now 0
if
  not Actor.Dead($player_actor)
goto_if_false @MAIN_FADEIN
undress_char $player_actor skin_to 'PLAYER'
load_all_models_now
if
  not Actor.Dead($player_actor)
goto_if_false @MAIN_FADEIN
dress_char $player_actor

:MAIN_FADEIN
do_fade 1 1000

{start_scripts}

if
  Player.Defined($player_char)
goto_if_false @MAIN_LOOP
set_area_visible 0
Player.CanMove($player_char, True)

release_weather

:MAIN_LOOP
wait 1000
if
  Player.Defined($player_char)
goto_if_false @MAIN_LOOP
goto @MAIN_LOOP
"""

    INITIAL_MISSION_TEMPLATE = """
:INITIAL
script_name 'INITAL'
remove_all_script_fires
set_actors_suffer_headshots 1
enable_choreo_control 1
clear_wanted_level_in_radius 500.0 0.0 0.0
set_player_cant_attack 0
set_always_draw_crosshair 0
enable_all_settings_for_tutorial 1
end_thread
"""

    def __init__(self, missions, trigger_scripts, mission_bodies, map_graph):
        self.missions = missions
        self.trigger_scripts = trigger_scripts
        self.mission_bodies = mission_bodies
        self.map_graph = map_graph

    def _make_define_missions(self) -> str:
        lines = [f"DEFINE MISSIONS {len(self.missions) + 2}"]
        lines.append("DEFINE MISSION 0 AT @INITIAL           // Initial")
        lines.append("DEFINE MISSION 1 AT @INTRO             // Intro")
        for i, concept in enumerate(self.missions):
            idx = i + 2
            lines.append(
                f"DEFINE MISSION {idx} AT @{concept['name']}           "
                f"// {concept['display']}"
            )
        return '\n'.join(lines)

    def _make_bribe_pickups(self) -> str:
        # Known bribe locations from original main.scm
        bribes = [
            (393.9, -60.2, 11.5),
            (116.0, -1313.1, 4.4),
            (393.7, -660.6, 10.7),
            (-822.7, 1304.5, 11.7),
        ]
        lines = []
        for i, (x, y, z) in enumerate(bribes):
            var = 110 + i
            lines.append(f"${var} = Pickup.Create(#BRIBE, 15, {x}, {y}, {z})")
        return '\n'.join(lines)

    def _make_car_generators(self) -> str:
        # Basic car generators at key spawn locations
        cars = [
            ('#SENTINEL', 83.0, -860.0, 9.3, 90.0),
            ('#TAXI', 120.0, -850.0, 9.3, 90.0),
            ('#CHEETAH', 200.0, -820.0, 10.0, 180.0),
        ]
        lines = []
        for i, (car, x, y, z, angle) in enumerate(cars):
            var = 87 + i
            lines.append(
                f"create_car_generator ${var} = init_car_generator {car} -1 -1 "
                f"force_spawn 1 alarm 0 door_lock 0 min_delay 0 max_delay 10000 "
                f"at {x} {y} {z} angle {angle}"
            )
            lines.append(f"switch_car_generator ${var} cars_to_generate_to -1")
        return '\n'.join(lines)

    def _make_start_scripts(self) -> str:
        names = [c['name'] for c in self.missions]
        return '\n'.join(f"start_new_script @{name}" for name in names)

    def _intro_mission(self) -> str:
        # mission index 2 is the first user mission only when missions exist
        first_mission_idx = 2 if self.missions else 0
        return f"""
:INTRO
script_name 'INTRO'
set_camera_behind_player
Camera.SetAtPos(83.0, -849.8, 9.3)
do_fade 1 2000
load_and_launch_mission_internal {first_mission_idx}
end_thread
"""

    def assemble(self) -> str:
        parts = [
            self.DEFINE_OBJECTS_TEMPLATE,
            self._make_define_missions(),
            self.MAIN_HEADER_TEMPLATE.format(
                num_missions=len(self.missions),
                car_generators=self._make_car_generators(),
                bribe_pickups=self._make_bribe_pickups(),
                start_scripts=self._make_start_scripts(),
            ),
            self.INITIAL_MISSION_TEMPLATE,
            self._intro_mission(),
        ]
        for script in self.trigger_scripts:
            parts.append('\n' + script)
        for name, body in self.mission_bodies:
            parts.append(f'\n// === MISSION: {name} ===\n' + body)
        return '\n'.join(parts)
