"""
MapGraph: spatial awareness layer for GTA Vice City.
Confirmed file in repo: spatial/map_graph.py
Corrected: added path node loading (_enrich_from_paths) so get_random_outdoor_coord
returns coordinates confirmed to be on actual road/ped nodes, not just zone-random.
"""
import json
import math
import random
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ── Known named locations hardcoded from main.scm analysis ──────────────────
KNOWN_LOCATIONS = {
    'ocean_beach_hotel':     (72.0,   -831.0, 11.0,  'OCEAN_BEACH',   'mission_trigger'),
    'ocean_beach_safehouse': (338.0,  -1022.0, 28.0, 'OCEAN_BEACH',   'safe_house'),
    'print_works':           (-620.0, -1300.0, 11.0, 'LITTLE_HAVANA', 'business'),
    'kaufman_cabs':          (-1006.0, 193.0,  12.0, 'LITTLE_HAITI',  'business'),
    'sunshine_autos':        (-619.0,  -634.0, 11.0, 'LITTLE_HAVANA', 'business'),
    'malibu_club':           (-522.0,  -1307.0, 11.0,'VICE_POINT',    'business'),
    'pole_position':         (-702.0,  -847.0,  11.0,'WASHINGTON',    'business'),
    'downtown_safe_house':   (37.0,    854.0,  11.0, 'DOWNTOWN',      'safe_house'),
    'mansion':               (-814.0,  492.0,  11.0, 'STARFISH_ISLAND','safe_house'),
    'airport_main':          (-1767.0,-14.0,   14.0, 'AIRPORT',       'mission_trigger'),
    'airport_north':         (-1511.0, 21.0,   14.0, 'AIRPORT',       'mission_trigger'),
    'docks_main':            (-1029.0,-1458.0, 11.0, 'DOCKS',         'mission_trigger'),
    'docks_north':           (-785.0, -1388.0, 11.0, 'DOCKS',         'mission_trigger'),
    'vice_point_mall':       (178.0,  -1320.0, 11.0, 'VICE_POINT',    'mission_trigger'),
    'vice_point_beach':      (470.0,  -820.0,  10.5, 'VICE_POINT',    'mission_trigger'),
    'little_haiti_north':    (-963.0,  109.0,  11.0, 'LITTLE_HAITI',  'gang_turf'),
    'little_haiti_south':    (-882.0, -538.0,  11.0, 'LITTLE_HAITI',  'gang_turf'),
    'little_havana_main':    (-518.0, -633.0,  11.0, 'LITTLE_HAVANA', 'gang_turf'),
    'downtown_main':         (226.0,   793.0,  11.0, 'DOWNTOWN',      'mission_trigger'),
    'downtown_north':        (168.0,  1154.0,  11.0, 'DOWNTOWN',      'mission_trigger'),
    'leaf_links_golf':       (120.0,   282.0,  10.5, 'LEAF_LINKS',    'mission_trigger'),
    'star_island':           (-829.0,  493.0,  11.0, 'STARFISH_ISLAND','mission_trigger'),
    'prawn_island':          (-391.0, -162.0,  11.0, 'PRAWN_ISLAND',  'mission_trigger'),
    'washington_beach':      (113.0,  -738.0,  11.0, 'WASHINGTON',    'mission_trigger'),
    'boatyard':              (-619.0,-1477.0,  11.0, 'DOCKS',         'business'),
    'hyman_stadium':         (-366.0,  694.0,  11.0, 'DOWNTOWN',      'mission_trigger'),
    'film_studio':           (-611.0,  932.0,  11.0, 'DOWNTOWN',      'business'),
    'hospital_wash':         (102.0,  -590.0,  11.0, 'WASHINGTON',    'mission_trigger'),
    'police_hq':             (373.0,  -610.0,  11.0, 'WASHINGTON',    'mission_trigger'),
    'vnc_building':          (185.0,   907.0,  11.0, 'DOWNTOWN',      'mission_trigger'),
}

# Interior entrance coordinates from main.scm (interior != 0 in IPL means inside)
INTERIORS = {
    'bank_vault':       (-807.0, -1042.0,  14.0,  'WASHINGTON'),
    'malibu_inside':    (-473.0, -1234.0,  11.0,  'VICE_POINT'),
    'pole_inside':      (-703.0,  -848.0,  11.0,  'WASHINGTON'),
    'printworks_inside':(-617.0, -1244.0,  11.0,  'LITTLE_HAVANA'),
    'mansion_inside':   (-814.0,   493.0,  11.0,  'STARFISH_ISLAND'),
    'mall_inside':      ( 176.0, -1300.0,  11.0,  'VICE_POINT'),
    'icecream_inside':  (-878.0,  -580.0,  11.0,  'LITTLE_HAITI'),
    'film_studio_inside':(-604.0, 973.0,   11.0,  'DOWNTOWN'),
    'stripclub_inside': (-701.0,  -849.0,  11.0,  'WASHINGTON'),
    'airport_inside':   (-1728.0,  -3.0,   14.0,  'AIRPORT'),
}

# Ground Z lookup by broad area — used when no path node is available
GROUND_Z_BY_AREA = [
    # (x_min, x_max, y_min, y_max, z)
    (-2000, 1500,  -2000,  2000, 10.5),   # default VC ground
    (-1950,-1300,  -1000,   300, 14.0),   # airport tarmac
    (  -50,  500,  -1900, -1500, 10.5),   # ocean beach south
    (-1300, -700,  -1800, -1100, 11.0),   # docks
    (   50,  500,   -100,   600, 10.5),   # vice point / leaf links
    ( -800, -350,   400,   950, 11.0),    # starfish / prawn
    (  100,  550,   800,  1400, 11.0),    # downtown north
]

# Zone name aliases for get_random_outdoor_coord
ZONE_ALIASES = {
    'OCEAN_BEACH':     (200.0,  -800.0),
    'WASHINGTON':      (100.0,  -600.0),
    'VICE_POINT':      (350.0,  -1100.0),
    'LITTLE_HAITI':    (-900.0,  200.0),
    'LITTLE_HAVANA':   (-500.0, -600.0),
    'DOWNTOWN':        (200.0,   900.0),
    'AIRPORT':         (-1700.0,-100.0),
    'DOCKS':           (-1000.0,-1400.0),
    'STARFISH_ISLAND': (-800.0,  500.0),
    'PRAWN_ISLAND':    (-380.0, -150.0),
    'LEAF_LINKS':      ( 120.0,  300.0),
}


@dataclass
class MapLocation:
    name: str
    x: float
    y: float
    z: float
    zone: str
    location_type: str  # 'mission_trigger','safe_house','business','gang_turf','interior','ipl_object'
    description: str = ''


class MapGraph:
    """
    Spatial awareness for GTA VC SCM generation.
    Loads: zones (info.zon JSON), IPL instances, and path nodes.
    Path nodes are the most reliable source of real drivable/walkable coordinates.
    """

    def __init__(self, zones_json: str = None, ipl_json: str = None,
                 paths_json: str = None):
        self.locations: List[MapLocation] = []
        self.zones_data: List[Dict] = []
        self.ipl_instances: List[Dict] = []

        # Path nodes — keyed by type
        self.road_nodes: List[Tuple[float, float, float]] = []
        self.ped_nodes: List[Tuple[float, float, float]] = []
        self.boat_nodes: List[Tuple[float, float, float]] = []

        # Load zone data
        if zones_json and os.path.exists(zones_json):
            try:
                with open(zones_json) as f:
                    self.zones_data = json.load(f)
                print(f"[MapGraph] Loaded {len(self.zones_data)} zones from {zones_json}")
            except Exception as e:
                print(f"[MapGraph] Warning: could not load zones: {e}")

        # Load IPL instances
        if ipl_json and os.path.exists(ipl_json):
            try:
                with open(ipl_json) as f:
                    raw = json.load(f)
                    self.ipl_instances = raw.get('instances', raw) if isinstance(raw, dict) else raw
                print(f"[MapGraph] Loaded {len(self.ipl_instances)} IPL instances from {ipl_json}")
            except Exception as e:
                print(f"[MapGraph] Warning: could not load IPL: {e}")

        # Load path nodes (NEW)
        if paths_json and os.path.exists(paths_json):
            self._enrich_from_paths(paths_json)
        else:
            if paths_json:
                print(f"[MapGraph] Info: paths JSON not found at {paths_json} — "
                      f"using zone-random coords. Run parse step to generate it.")

        # Build known locations from hardcoded data
        self._build_known_locations()

        # Enrich with outdoor IPL object positions
        self._enrich_from_ipl()

        print(f"[MapGraph] Total locations: {len(self.locations)}, "
              f"Road nodes: {len(self.road_nodes)}, "
              f"Ped nodes: {len(self.ped_nodes)}")

    # ── Path node loading ────────────────────────────────────────────────────

    def _enrich_from_paths(self, paths_json: str):
        """Load path nodes from the output of PathsParser.to_json()."""
        try:
            with open(paths_json) as f:
                data = json.load(f)

            for n in data.get('road_nodes', []):
                self.road_nodes.append((n['x'], n['y'], n['z']))

            for n in data.get('ped_nodes', []):
                self.ped_nodes.append((n['x'], n['y'], n['z']))

            for n in data.get('boat_nodes', []):
                self.boat_nodes.append((n['x'], n['y'], n['z']))

            print(f"[MapGraph] Path nodes loaded — "
                  f"road: {len(self.road_nodes)}, "
                  f"ped: {len(self.ped_nodes)}, "
                  f"boat: {len(self.boat_nodes)}")
        except Exception as e:
            print(f"[MapGraph] Warning: could not load paths JSON: {e}")

    # ── Location building ────────────────────────────────────────────────────

    def _build_known_locations(self):
        """Add all hardcoded known locations."""
        for name, data in KNOWN_LOCATIONS.items():
            x, y, z, zone, loc_type = data
            self.locations.append(MapLocation(
                name=name, x=x, y=y, z=z,
                zone=zone, location_type=loc_type,
                description=name.replace('_', ' ').title()
            ))
        for name, data in INTERIORS.items():
            x, y, z, zone = data
            self.locations.append(MapLocation(
                name=name, x=x, y=y, z=z,
                zone=zone, location_type='interior',
                description=name.replace('_', ' ').title()
            ))

    def _enrich_from_ipl(self):
        """Add outdoor IPL objects as potential ipl_object locations."""
        added = 0
        for inst in self.ipl_instances:
            if inst.get('interior', 1) != 0:
                continue  # skip interiors
            x, y, z = inst.get('x', 0), inst.get('y', 0), inst.get('z', 0)
            if not self.is_valid_coord(x, y, z):
                continue
            model = inst.get('model_name', '')
            # Only use large landmark objects, not tiny props
            if len(model) < 4:
                continue
            zone = self._guess_zone(x, y)
            self.locations.append(MapLocation(
                name=model,
                x=round(x, 2), y=round(y, 2), z=round(z, 2),
                zone=zone,
                location_type='ipl_object',
                description=f'IPL object: {model}'
            ))
            added += 1
            if added >= 500:  # cap to avoid memory bloat
                break

    # ── Spatial queries ──────────────────────────────────────────────────────

    def _guess_zone(self, x: float, y: float) -> str:
        """Guess zone name from coordinate using loaded zone data."""
        for zone in self.zones_data:
            x1, y1 = zone.get('x1', 0), zone.get('y1', 0)
            x2, y2 = zone.get('x2', 0), zone.get('y2', 0)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone.get('name', 'UNKNOWN')
        # Fallback: broad area guess
        if x < -1300:
            return 'AIRPORT'
        elif x < -600 and y < -1000:
            return 'DOCKS'
        elif x < -600:
            return 'LITTLE_HAITI'
        elif x < 0 and y < 0:
            return 'LITTLE_HAVANA'
        elif y > 600:
            return 'DOWNTOWN'
        elif x > 300:
            return 'VICE_POINT'
        elif x > 0 and y < -600:
            return 'OCEAN_BEACH'
        elif x < 0 and y > 300:
            return 'STARFISH_ISLAND'
        return 'UNKNOWN'

    def get_zone_for_coord(self, x: float, y: float) -> str:
        return self._guess_zone(x, y)

    def get_z_for_coord(self, x: float, y: float) -> float:
        """Look up ground Z for a coordinate using area table."""
        # First: try to find a nearby path node for accurate Z
        if self.road_nodes or self.ped_nodes:
            closest_z = self._find_nearest_node_z(x, y)
            if closest_z is not None:
                return closest_z

        # Fallback: area-based table
        for x_min, x_max, y_min, y_max, z in GROUND_Z_BY_AREA:
            if x_min <= x <= x_max and y_min <= y <= y_max:
                return z
        return 10.5  # VC default ground

    def _find_nearest_node_z(self, x: float, y: float,
                              max_dist: float = 100.0) -> Optional[float]:
        """Find Z of the nearest road or ped node within max_dist."""
        best_dist = max_dist * max_dist
        best_z = None
        for nx, ny, nz in self.road_nodes + self.ped_nodes:
            d = (nx - x) ** 2 + (ny - y) ** 2
            if d < best_dist:
                best_dist = d
                best_z = nz
        return best_z

    def is_valid_coord(self, x: float, y: float, z: float) -> bool:
        """Check coordinate is within Vice City world bounds."""
        return (-2000 < x < 1500 and
                -2000 < y < 2000 and
                -10 < z < 200)

    def get_nearest_location(self, x: float, y: float,
                             location_type: str = None) -> Optional[MapLocation]:
        """Find nearest known location to a coordinate."""
        candidates = self.locations
        if location_type:
            candidates = [l for l in self.locations if l.location_type == location_type]
        if not candidates:
            return None

        best = min(candidates, key=lambda l: (l.x - x) ** 2 + (l.y - y) ** 2)
        return best

    def get_locations_by_type(self, location_type: str) -> List[MapLocation]:
        return [l for l in self.locations if l.location_type == location_type]

    def get_random_outdoor_coord(self, zone_name: str = None) -> Tuple[float, float, float]:
        """
        Return a random outdoor coordinate.
        Priority:
          1. If path nodes loaded: pick a random road/ped node in the requested zone
          2. Fallback: pick a random known location + small random offset
          3. Last resort: zone-centre + random offset
        """
        all_nav_nodes = self.road_nodes + self.ped_nodes

        if all_nav_nodes:
            if zone_name and zone_name.upper() in ZONE_ALIASES:
                # Filter nodes to the requested zone area
                zone_nodes = [
                    n for n in all_nav_nodes
                    if self._guess_zone(n[0], n[1]).upper() == zone_name.upper()
                ]
                if zone_nodes:
                    node = random.choice(zone_nodes)
                    return (round(node[0], 3), round(node[1], 3), round(node[2], 3))

            # No zone filter, or no nodes in that zone — pick any road node
            node = random.choice(all_nav_nodes)
            return (round(node[0], 3), round(node[1], 3), round(node[2], 3))

        # Fallback: use known outdoor locations
        outdoor_types = ['mission_trigger', 'safe_house', 'business', 'gang_turf']
        candidates = [l for l in self.locations if l.location_type in outdoor_types]

        if zone_name and candidates:
            zone_candidates = [l for l in candidates
                               if l.zone.upper() == zone_name.upper()]
            if zone_candidates:
                candidates = zone_candidates

        if candidates:
            loc = random.choice(candidates)
            # Add small random offset so missions don't stack on the same spot
            dx = random.uniform(-15, 15)
            dy = random.uniform(-15, 15)
            x = round(loc.x + dx, 3)
            y = round(loc.y + dy, 3)
            z = round(self.get_z_for_coord(x, y), 3)
            return (x, y, z)

        # Last resort: zone centre + random offset
        if zone_name and zone_name.upper() in ZONE_ALIASES:
            cx, cy = ZONE_ALIASES[zone_name.upper()]
        else:
            cx, cy = 200.0, -800.0  # Ocean Beach default

        x = round(cx + random.uniform(-80, 80), 3)
        y = round(cy + random.uniform(-80, 80), 3)
        z = round(self.get_z_for_coord(x, y), 3)
        return (x, y, z)

    def distance(self, x1: float, y1: float, z1: float,
                 x2: float, y2: float, z2: float) -> float:
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def export_locations_json(self, out_path: str):
        """Export all known locations to JSON."""
        data = [
            {'name': l.name, 'x': l.x, 'y': l.y, 'z': l.z,
             'zone': l.zone, 'type': l.location_type,
             'description': l.description}
            for l in self.locations
        ]
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[MapGraph] Exported {len(data)} locations → {out_path}")
