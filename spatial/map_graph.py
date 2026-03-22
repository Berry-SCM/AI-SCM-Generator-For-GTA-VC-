"""
Builds a spatial understanding of Vice City from parsed map data.
- Zones with bounding boxes (from info.zon via ZonParser)
- Known key locations (safe houses, mission triggers, businesses) from main.scm
- IPL-based object placement enrichment
- Coordinate validity checking
- Interior detection
- Ground Z estimation by area
"""

import json
import math
import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Known key locations extracted from main.scm coord analysis
# Format: name -> (x, y, z, description)
KNOWN_LOCATIONS = {
    'hotel_spawn':      (83.0,    -849.8,  9.3,  'Player spawn / hotel'),
    'lawyer_office':    (119.2,   -826.9,  9.7,  'Lawyer missions'),
    'vercetti_estate':  (-378.3,  -579.8,  24.5, 'Vercetti Estate / final missions'),
    'colonel_missions': (-246.6,  -1360.5, 7.3,  'Colonel Cortez yacht'),
    'diaz_mansion':     (-378.5,  -551.3,  18.2, 'Diaz / Vercetti mansion'),
    'ken_rosenberg':    (491.0,   -77.7,   10.4, 'Ken Rosenbergs office'),
    'malibu_club':      (487.2,   -81.5,   11.4, 'Malibu Club'),
    'cubans':           (-1170.0, -606.9,  10.6, 'Cuban gang'),
    'haitians':         (-962.4,  143.0,   8.2,  'Haitian gang'),
    'bikers':           (-597.3,  652.9,   10.0, 'Biker gang - Greasy Chopper'),
    'rockstars':        (-875.5,  1159.3,  10.2, 'Love Fist band'),
    'phil_cassidy':     (-1101.1, 343.2,   10.2, 'Phil Cassidy'),
    'film_studio':      (-69.4,   932.7,   9.9,  'InterGlobal Film Studio'),
    'print_works':      (-1059.6, -274.5,  11.4, 'Print Works'),
    'car_showroom':     (-1007.3, -869.9,  12.8, 'Sunshine Autos'),
    'ice_cream':        (-864.3,  -576.6,  11.0, 'Cherry Popper Ice Cream'),
    'kaufman_cabs':     (-1011.7, 203.9,   11.2, 'Kaufman Cabs'),
    'boatyard':         (-685.8,  -1495.6, 12.5, 'Boatyard'),
    'downtown_ammu':    (-665.63, 1231.863,10.1, 'Ammu-Nation Downtown'),
    'airport':          (-1720.3, -239.6,  14.8, 'Airport'),
    'stadium':          (-1110.331,1331.096,20.112,'Stadium'),
    'golf_club':        (257.1,   -231.7,  10.0, 'Golf Club / Cam Jones'),
    'bank':             (463.9,   -58.5,   10.5, 'Bank / El Banco Corrupto Grande'),
}

# Known interiors — coordinates are the in-world entrance/trigger points
# (from gta_vc.dat IPL loading and main.scm interior teleport analysis)
INTERIORS = {
    'bank':       {'center': (-920.0, -330.0, 14.0), 'zone': 'bank'},
    'mall':       {'center': (200.0,  -640.0, 11.0), 'zone': 'mall'},
    'hotel':      {'center': (230.0,  -1280.0,12.0), 'zone': 'hotel'},
    'lawyers':    {'center': (120.0,  -820.0,  9.5), 'zone': 'lawyers'},
    'stripclub':  {'center': (100.0,  -1467.0,10.0), 'zone': 'stripclb'},
    'concerth':   {'center': (-450.0, 590.0,  11.0), 'zone': 'concerth'},
    'mansion':    {'center': (-380.0, -560.0, 18.0), 'zone': 'mansion'},
    'yacht':      {'center': (-376.0, -1322.0, 9.8), 'zone': 'yacht'},
    'club_malibu':{'center': (490.0,  -80.0,  11.0), 'zone': 'club'},
}

# Ground Z estimates by area bounding box: (x_min, x_max, y_min, y_max, z)
# Derived from SCM coordinate analysis across all missions
GROUND_Z_BY_AREA = [
    ('ocean_beach',   -200,  700, -1800, -400, 10.5),
    ('washington',    -200,  300,  -400,  100, 10.5),
    ('vice_point',     300,  900,  -400,  700, 10.8),
    ('little_haiti', -1100, -600,  -400,  400, 11.0),
    ('downtown',      -700,  200,   400, 1400, 10.5),
    ('airport',      -1950,-1200,  -400,  300, 12.0),
    ('docks',        -1250, -550, -1800,-1100, 11.5),
    ('starfish',      -400,  100,  -800, -400, 11.0),
]


@dataclass
class MapLocation:
    name: str
    x: float
    y: float
    z: float
    zone: str
    location_type: str  # 'mission_trigger', 'safe_house', 'business', 'gang_turf', 'interior', 'ipl_object'
    description: str = ''


class MapGraph:
    """
    Central spatial database for Vice City.

    Provides:
    - Coordinate validation against VC world bounds
    - Zone lookup (from parsed info.zon zones or fallback heuristic)
    - Ground Z estimation by area
    - Known location registry (mission triggers, interiors, businesses)
    - IPL object placement enrichment (so the AI knows where buildings/objects are)
    - Random valid spawn point selection
    - Distance calculations
    """

    def __init__(self, zones_json: str = None, ipl_json: str = None):
        self.zones: List[Dict] = []
        self.ipl_instances: List[Dict] = []
        self.locations: Dict[str, MapLocation] = {}

        # 1. Load parsed zone data (from info.zon)
        if zones_json:
            try:
                with open(zones_json) as f:
                    raw = json.load(f)
                # Support both list-of-dicts (ZonParser output) and dict formats
                if isinstance(raw, list):
                    self.zones = raw
                elif isinstance(raw, dict):
                    self.zones = list(raw.values())
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"[MapGraph] Warning: could not load zones_json: {e}")

        # 2. Load IPL instance data
        if ipl_json:
            try:
                with open(ipl_json) as f:
                    self.ipl_instances = json.load(f).get('instances', [])
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"[MapGraph] Warning: could not load ipl_json: {e}")

        # 3. Build known locations from hardcoded + INTERIORS
        self._build_known_locations()

        # 4. Enrich from IPL instances (so AI knows actual placed objects)
        if self.ipl_instances:
            self._enrich_from_ipl()

    def _build_known_locations(self):
        """Populate locations dict from KNOWN_LOCATIONS and INTERIORS constants."""
        for name, data in KNOWN_LOCATIONS.items():
            x, y, z, desc = data
            self.locations[name] = MapLocation(
                name=name, x=x, y=y, z=z,
                zone=self.get_zone_for_coord(x, y),
                location_type='mission_trigger',
                description=desc
            )
        for name, data in INTERIORS.items():
            cx, cy, cz = data['center']
            self.locations[f'interior_{name}'] = MapLocation(
                name=f'interior_{name}', x=cx, y=cy, z=cz,
                zone=data['zone'],
                location_type='interior',
                description=f'Interior: {name}'
            )

    def _enrich_from_ipl(self):
        """
        Add significant IPL-placed objects to the location registry.
        Only indexes objects that are NOT LODs (lod >= 0 with lod != obj_id
        is a LOD reference; we skip those) and are in interior 0 (outdoor world).
        This gives the AI awareness of where actual map geometry is placed.
        """
        added = 0
        seen_models: Dict[str, int] = {}
        for inst in self.ipl_instances:
            try:
                interior = int(inst.get('interior', 0))
                if interior != 0:
                    continue  # skip interior-only objects
                x = float(inst['x'])
                y = float(inst['y'])
                z = float(inst['z'])
                model = str(inst.get('model_name', '')).lower()
                if not model or model == 'null':
                    continue
                # Deduplicate: only keep first occurrence of each model per ~100u grid cell
                cell = (int(x / 100), int(y / 100), model)
                if cell in seen_models:
                    continue
                seen_models[cell] = 1
                loc_name = f'ipl_{model}_{added}'
                self.locations[loc_name] = MapLocation(
                    name=loc_name, x=x, y=y, z=z,
                    zone=self.get_zone_for_coord(x, y),
                    location_type='ipl_object',
                    description=f'IPL object: {model}'
                )
                added += 1
                if added >= 500:  # cap to keep memory reasonable
                    break
            except (KeyError, ValueError, TypeError):
                continue
        if added:
            print(f"[MapGraph] Enriched with {added} IPL object locations")

    def _guess_zone(self, x: float, y: float) -> str:
        """Rough zone estimation from coordinates when zone file not loaded."""
        if x > 300 and y < -400:
            return 'OCEAN_BEACH'
        elif x > 300 and y > -400:
            return 'VICE_POINT'
        elif -200 < x < 300 and y < 0:
            return 'WASHINGTON'
        elif x < -600 and y < -800:
            return 'DOCKS'
        elif x < -1200:
            return 'AIRPORT'
        elif x < -600 and y > 0:
            return 'LITTLE_HAITI'
        elif y > 400:
            return 'DOWNTOWN'
        else:
            return 'UNKNOWN'

    def get_zone_for_coord(self, x: float, y: float) -> str:
        """Look up zone name for a coordinate, using parsed zones or fallback heuristic."""
        for zone in self.zones:
            try:
                x1 = min(float(zone['x1']), float(zone['x2']))
                x2 = max(float(zone['x1']), float(zone['x2']))
                y1 = min(float(zone['y1']), float(zone['y2']))
                y2 = max(float(zone['y1']), float(zone['y2']))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    return zone.get('name', 'UNKNOWN')
            except (KeyError, ValueError, TypeError):
                continue
        return self._guess_zone(x, y)

    def get_z_for_coord(self, x: float, y: float) -> float:
        """Estimate ground Z height for a coordinate based on area bounds."""
        for area_name, x_min, x_max, y_min, y_max, z in GROUND_Z_BY_AREA:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return z
        return 10.5  # default ground Z

    def is_valid_coord(self, x: float, y: float, z: float) -> bool:
        """Check if coordinate is within Vice City world bounds."""
        return (-2000 <= x <= 1000 and
                -1900 <= y <= 1800 and
                0 <= z <= 200)

    def get_nearest_location(self, x: float, y: float,
                              location_type: str = None) -> Optional[MapLocation]:
        """Find the nearest registered location to a given coordinate."""
        best = None
        best_dist = float('inf')
        for loc in self.locations.values():
            if location_type and loc.location_type != location_type:
                continue
            d = math.sqrt((loc.x - x) ** 2 + (loc.y - y) ** 2)
            if d < best_dist:
                best_dist = d
                best = loc
        return best

    def get_locations_by_type(self, location_type: str) -> List[MapLocation]:
        """Get all locations of a given type."""
        return [loc for loc in self.locations.values()
                if loc.location_type == location_type]

    def get_random_outdoor_coord(self, zone_name: str = None) -> Tuple[float, float, float]:
        """
        Returns a random known outdoor coordinate, optionally filtered by zone name.
        Adds a small random offset so multiple calls give varied positions.
        """
        candidates = [
            (x, y, z) for x, y, z, _desc in KNOWN_LOCATIONS.values()
        ]
        if zone_name:
            candidates = [
                (x, y, z) for x, y, z, _desc in KNOWN_LOCATIONS.values()
                if self.get_zone_for_coord(x, y).upper() == zone_name.upper()
            ] or candidates  # fall back to all if no match
        x, y, z = random.choice(candidates)
        x = round(x + random.uniform(-15, 15), 3)
        y = round(y + random.uniform(-15, 15), 3)
        # Clamp to valid bounds
        x = max(-2000, min(1000, x))
        y = max(-1900, min(1800, y))
        return (x, y, z)

    def distance(self, x1: float, y1: float, z1: float,
                 x2: float, y2: float, z2: float) -> float:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def export_locations_json(self, out_path: str):
        """Export all registered locations to JSON for inspection/training."""
        data = {
            name: {
                'x': loc.x, 'y': loc.y, 'z': loc.z,
                'zone': loc.zone,
                'type': loc.location_type,
                'desc': loc.description
            }
            for name, loc in self.locations.items()
        }
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[MapGraph] {len(self.locations)} locations → {out_path}")
