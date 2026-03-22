"""
Scrapes GTA VC opcode list from gtamods.com and saves to JSON for training.
Run once: python -m training.opcode_scraper
Output: data/processed/opcodes.json
"""
import json
import re
import urllib.request
from pathlib import Path

OPCODE_URL = "https://gtamods.com/wiki/List_of_opcodes"

# Hand-curated VC-specific opcodes extracted from main.scm analysis
# (Used as fallback if network unavailable)
VC_OPCODES = {
    # Player / Actor
    "0001": {"name": "wait", "args": "int ms", "desc": "Wait N milliseconds"},
    "0002": {"name": "goto", "args": "label", "desc": "Jump to label"},
    "0004": {"name": "goto_if_false", "args": "label", "desc": "Jump if condition false"},
    "0053": {"name": "load_and_launch_mission_internal", "args": "int mission_id", "desc": "Load and start a mission"},
    "0055": {"name": "load_and_launch_mission", "args": "int mission_id", "desc": "Load mission"},
    "0084": {"name": "Player.Create", "args": "int slot float x float y float z", "desc": "Create player"},
    "0086": {"name": "Player.Defined", "args": "Player handle", "desc": "Is player defined"},
    "0097": {"name": "Actor.Create", "args": "int type int model float x float y float z", "desc": "Create actor (ped)"},
    "009A": {"name": "Actor.Dead", "args": "Actor handle", "desc": "Is actor dead"},
    "00A1": {"name": "Actor.Remove", "args": "Actor handle", "desc": "Remove actor"},
    "00A5": {"name": "Car.Create", "args": "int model float x float y float z", "desc": "Create vehicle"},
    "00AA": {"name": "Car.Defined", "args": "Car handle", "desc": "Is car defined"},
    "00AB": {"name": "Car.Remove", "args": "Car handle", "desc": "Remove car"},
    # Mission flow
    "014B": {"name": "set_onmission", "args": "int 0or1", "desc": "Set mission flag"},
    "014D": {"name": "declare_mission_flag", "args": "var", "desc": "Declare the onmission variable"},
    "014E": {"name": "terminate_this_script", "args": "", "desc": "End current script"},
    # Markers / Blips
    "0164": {"name": "create_marker", "args": "int type float x float y float z", "desc": "Create a marker sphere"},
    "0165": {"name": "Blip.Remove", "args": "Blip handle", "desc": "Remove a blip/marker"},
    "019A": {"name": "Marker.Disable", "args": "Marker handle", "desc": "Disable marker"},
    "0199": {"name": "create_icon_marker_and_sphere", "args": "int icon float x float y float z", "desc": "Create icon marker"},
    # Screen / Camera
    "016A": {"name": "do_fade", "args": "int dir int time", "desc": "Fade screen"},
    "016B": {"name": "set_fading_colour", "args": "int r int g int b", "desc": "Set fade color"},
    "0170": {"name": "Camera.SetAtPos", "args": "float x float y float z", "desc": "Set camera position"},
    "016C": {"name": "print_big", "args": "str key int time int style", "desc": "Show big text message"},
    # Objects
    "0107": {"name": "Object.Init", "args": "int model float x float y float z", "desc": "Create object"},
    "0108": {"name": "Object.Remove", "args": "Object handle", "desc": "Remove object"},
    "0111": {"name": "Object.RemoveFromMissionCleanupList", "args": "Object handle", "desc": "Keep object after mission"},
    # Pickups
    "0213": {"name": "Pickup.Create", "args": "int model int type float x float y float z", "desc": "Create pickup"},
    "0214": {"name": "Pickup.Collected", "args": "Pickup handle", "desc": "Has pickup been collected"},
    # Cars
    "014C": {"name": "init_car_generator", "args": "...", "desc": "Set up a car generator"},
    "014D": {"name": "switch_car_generator", "args": "Generator handle int count", "desc": "Enable/disable car generator"},
    # Weapons
    "01B2": {"name": "give_actor_weapon", "args": "Actor handle int weapon int ammo", "desc": "Give actor a weapon"},
    "0227": {"name": "give_player_weapon", "args": "Player handle int weapon int ammo", "desc": "Give player a weapon"},
    # Location checks
    "00FF": {"name": "locate_player_on_foot_3d", "args": "Player handle int unused float x float y float z float rx float ry float rz", "desc": "Is player near coord on foot"},
    "0100": {"name": "locate_player_in_car_3d", "args": "Player handle int unused float x float y float z float rx float ry float rz", "desc": "Is player near coord in car"},
    # Progress / Stats
    "030C": {"name": "set_total_number_of_missions", "args": "int n", "desc": "Set mission total for stats"},
    "030D": {"name": "set_progress_total", "args": "int n", "desc": "Set progress total"},
    "0311": {"name": "set_max_wanted_level", "args": "int level", "desc": "Set max wanted stars"},
    # Save / Checkpoint
    "0169": {"name": "set_deatharrest_state", "args": "int 0or1", "desc": "Enable/disable death-arrest respawn"},
    "0395": {"name": "save_game_now", "args": "", "desc": "Force save (unused in VC main)"},
    # Time
    "00C0": {"name": "set_time_of_day", "args": "int hour int min", "desc": "Set game time"},
    # Phones
    "0214": {"name": "get_phone_at", "args": "float x float y", "desc": "Grab phone handle at coords"},
    # Collision preload
    "0213": {"name": "request_collision", "args": "float x float y", "desc": "Request collision data at coord"},
    # Script name
    "03A4": {"name": "script_name", "args": "str name", "desc": "Set script name for debugger"},
    # Wanted level
    "01F5": {"name": "Player.MakeSafe", "args": "Player handle", "desc": "Clear wanted level, freeze player"},
    "0211": {"name": "Player.Controllable", "args": "Player handle", "desc": "Is player controllable"},
}

# Vehicle IDs from default.ide (attached)
VC_VEHICLE_IDS = {
    130: "landstal", 131: "idaho", 132: "stinger", 133: "linerun", 134: "peren",
    135: "sentinel", 136: "rio", 137: "firetruk", 138: "trash", 139: "stretch",
    140: "manana", 141: "infernus", 142: "voodoo", 143: "pony", 144: "mule",
    145: "cheetah", 146: "ambulan", 147: "fbicar", 148: "moonbeam", 149: "esperant",
    150: "taxi", 151: "washing", 152: "bobcat", 153: "mrwhoop", 154: "bfinject",
    155: "hunter", 156: "police", 157: "enforcer", 158: "securica", 159: "banshee",
    160: "predator", 161: "bus", 162: "rhino", 163: "barracks", 164: "cuban",
    165: "chopper", 166: "angel", 167: "coach", 168: "cabbie", 169: "stallion",
    170: "rumpo", 171: "rcbandit", 172: "romero", 173: "packer", 174: "sentxs",
    175: "admiral", 176: "squalo", 177: "seaspar", 178: "pizzaboy", 179: "gangbur",
    180: "airtrain", 181: "deaddodo", 182: "speeder", 183: "reefer", 184: "tropic",
    185: "flatbed", 186: "yankee", 187: "caddy", 188: "zebra", 189: "topfun",
    190: "skimmer", 191: "pcj600", 192: "faggio", 193: "freeway", 194: "rcbaron",
    195: "rcraider", 196: "glendale", 197: "oceanic", 198: "sanchez", 199: "sparrow",
    200: "patriot", 201: "lovefist", 202: "coastg", 203: "dinghy", 204: "hermes",
    205: "sabre", 206: "sabretur", 207: "pheonix", 208: "walton", 209: "regina",
    210: "comet", 211: "deluxo", 212: "burrito", 213: "spand", 214: "marquis",
    215: "baggage", 216: "kaufman", 217: "maverick", 218: "vcnmav", 219: "rancher",
    220: "fbiranch", 221: "virgo", 222: "greenwoo", 223: "jetmax", 224: "hotring",
    225: "sandking", 226: "blistac", 227: "polmav", 228: "boxville", 229: "benson",
    230: "mesa", 231: "rcgoblin", 232: "hotrina", 233: "hotrinb", 234: "bloodra",
    235: "bloodrb", 236: "vicechee",
}

# Ped IDs from default.ide (attached)
VC_PED_IDS = {
    0: "null(player)", 1: "cop", 2: "swat", 3: "fbi", 4: "army",
    5: "medic", 6: "fireman", 7: "male01",
    9: "HFYST", 10: "HFOST", 11: "HMYST", 12: "HMOST", 13: "HFYRI",
    19: "HMYBE", 29: "BMODK", 30: "BMYCR", 33: "BMYST", 48: "WMYCR",
    83: "CBa(gang1)", 84: "CBb(gang1)", 85: "HNa(gang2)", 86: "HNb(gang2)",
    87: "SGa(gang3)", 88: "SGb(gang3)", 89: "CLa(gang4)", 90: "CLb(gang4)",
    91: "GDa(gang5)", 92: "GDb(gang5)", 93: "BKa(gang6)", 94: "BKb(gang6)",
    95: "PGa(gang7)", 96: "PGb(gang7)",
    109: "special01", 110: "special02", 111: "special03", 112: "special04",
    113: "special05", 114: "special06", 115: "special07", 116: "special08",
}

# Weapon IDs from default.ide (attached)
VC_WEAPON_IDS = {
    258: "cellphone", 259: "brassknuckle", 260: "screwdriver", 261: "golfclub",
    262: "nitestick", 263: "knifecur", 264: "bat", 265: "hammer",
    266: "cleaver", 267: "machete", 268: "katana", 269: "chnsaw",
    270: "grenade", 271: "teargas", 272: "molotov", 273: "missile",
    274: "colt45", 275: "python", 276: "ruger",
    277: "chromegun", 278: "shotgspa", 279: "buddyshot",
    280: "m4", 281: "tec9", 282: "uzi", 283: "ingramsl", 284: "mp5lng",
    285: "sniper", 286: "laser", 287: "rocketla", 288: "flame",
    289: "M60", 290: "minigun", 291: "bomb", 292: "camera",
    293: "fingers", 294: "minigun2",
}


def build_id_training_pairs() -> list:
    """Build training pairs for vehicle/ped/weapon IDs."""
    pairs = []

    # Vehicle ID pairs
    for vid, vname in VC_VEHICLE_IDS.items():
        pairs.append({
            "messages": [
                {"role": "system", "content": "You are a GTA Vice City SCM script expert."},
                {"role": "user", "content": f"What is the vehicle model name for ID {vid}?"},
                {"role": "assistant", "content": f"Vehicle ID {vid} is '{vname}'. Used in SCM as #{''.join(vname.upper().split()[:1])}."}
            ]
        })
        pairs.append({
            "messages": [
                {"role": "system", "content": "You are a GTA Vice City SCM script expert."},
                {"role": "user", "content": f"How do I spawn a {vname} in SCM?"},
                {"role": "assistant", "content": f"$car = Car.Create(#{vname.upper()}, x, y, z)  // model ID {vid}"}
            ]
        })

    # Weapon ID pairs
    for wid, wname in VC_WEAPON_IDS.items():
        pairs.append({
            "messages": [
                {"role": "system", "content": "You are a GTA Vice City SCM script expert."},
                {"role": "user", "content": f"How do I give the player a {wname} in SCM?"},
                {"role": "assistant", "content": f"give_player_weapon $player_char {wid} 100  // {wname}, 100 ammo"}
            ]
        })

    # Ped ID pairs
    for pid, pname in VC_PED_IDS.items():
        pairs.append({
            "messages": [
                {"role": "system", "content": "You are a GTA Vice City SCM script expert."},
                {"role": "user", "content": f"What ped skin ID is {pname}?"},
                {"role": "assistant", "content": f"Ped model '{pname}' has ID {pid}."}
            ]
        })

    return pairs


def save_id_data(out_dir: str = "data/processed"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Save opcode JSON
    with open(f"{out_dir}/opcodes.json", "w") as f:
        json.dump(VC_OPCODES, f, indent=2)
    print(f"[OpcodesScraper] Saved {len(VC_OPCODES)} opcodes → {out_dir}/opcodes.json")

    # Save vehicle IDs
    with open(f"{out_dir}/vehicle_ids.json", "w") as f:
        json.dump(VC_VEHICLE_IDS, f, indent=2)
    print(f"[OpcodesScraper] Saved {len(VC_VEHICLE_IDS)} vehicle IDs → {out_dir}/vehicle_ids.json")

    # Save ped IDs
    with open(f"{out_dir}/ped_ids.json", "w") as f:
        json.dump(VC_PED_IDS, f, indent=2)
    print(f"[OpcodesScraper] Saved {len(VC_PED_IDS)} ped IDs → {out_dir}/ped_ids.json")

    # Save weapon IDs
    with open(f"{out_dir}/weapon_ids.json", "w") as f:
        json.dump(VC_WEAPON_IDS, f, indent=2)
    print(f"[OpcodesScraper] Saved {len(VC_WEAPON_IDS)} weapon IDs → {out_dir}/weapon_ids.json")

    # Save training pairs
    pairs = build_id_training_pairs()
    with open(f"{out_dir}/id_training_pairs.jsonl", "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    print(f"[OpcodesScraper] Saved {len(pairs)} ID training pairs → {out_dir}/id_training_pairs.jsonl")


if __name__ == "__main__":
    save_id_data()