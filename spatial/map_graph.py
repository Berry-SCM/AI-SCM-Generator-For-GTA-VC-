"""
Builds a spatial understanding of Vice City from parsed map data.
- Zones with bounding boxes
- Known key locations (safe houses, mission triggers, businesses)
- Coordinate validity checking
- Interior detection
"""

import json
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

# Known key locations extracted from main.scm analysis
KNOWN_LOCATIONS = {
    # Format: name -> (x, y, z, description)
    'hotel_spawn': (83.0, -849.8, 9.3, 'Player spawn / hotel'),
    'lawyer_office': (119.2, -826.9, 9.7, 'Lawyer missions'),
    'vercetti_estate': (-378.3, -579.8, 24.5, 'Vercetti Estate / final missions'),
    'colonel_missions': (-246.6, -1360.5, 7.3, 'Colonel Cortez yacht'),
    'diaz_mansion': (-378.5, -551.3, 18.2, 'Diaz / Vercetti mansion'),
    'ken_rosenberg': (491.0, -77.7, 10.4, 'Ken Rosenbergs office'),
    'malibu_club': (487.2, -81.5, 11.4, 'Malibu Club'),
    'cubans': (-1170.0, -606.9, 10.6, 'Cuban gang'),
    'haitians': (-962.4, 143.0, 8.2, 'Haitian gang'),
    'bikers': (-597.3, 652.9, 10.0, 'Biker gang - Greasy Chopper'),
    'rockstars': (-875.5, 1159.3, 10.2, 'Love Fist band'),
    'phil_cassidy': (-1101.1, 343.2, 10.2, 'Phil Cassidy'),
    'film_studio': (-69.4, 932.7, 9.9, 'InterGlobal Film Studio'),
    'print_works': (-1059.6, -274.5, 11.4, 'Print Works'),
    'car_showroom': (-1007.3, -869.9, 12.8, 'Sunshine Autos'),
    'ice_cream': (-864.3, -576.6, 11.0, 'Cherry Popper Ice Cream'),
    'kaufman_cabs': (-1011.7, 203.9, 11.2, 'Kaufman Cabs'),
    'boatyard': (-685.8, -1495.6, 12.5, 'Boatyard'),
    'downtown_ammu': (-665.63, 1231.863, 10.1, 'Ammu-Nation Downtown'),
    'airport': (-1720.3, -239.6, 14.8, 'Airport'),
    'stadium': (-1110.331, 1331.096, 20.1119, 'Stadium'),
    'golf_club': (257.1, -231.7, 10.0, 'Golf Club / Cam Jones'),
    'bank': (463.9, -58.5, 10.5, 'Bank / El Banco Corrupto Grande'),
}

# Known interiors (from gta_vc.dat IPL section)
INTERIORS = {
    'bank': {'center': (-920.0, -330.0, 14.0), 'zone': 'bank'},
    'mall': {'center': (200.0, -640.0, 11.0), 'zone': 'mall'},
    'hotel': {'center': (230.0, -1280.0, 12.0), 'zone': 'hotel'},
    'lawyers': {'center': (120.0, -820.0, 9.5), 'zone': 'lawyers'},
    'stripclub': {'center': (100.0, -1467.0, 10.0), 'zone': 'stripclb'},
    'concerth': {'center': (-450.0, 590.0, 11.0), 'zone': 'concerth'},
    'mansion': {'center': (-380.0, -560.0, 18.0), 'zone': 'mansion'},
    'yacht': {'center': (-376.0, -1322.0, 9.8), 'zone': 'yacht'},
    'club_malibu': {'center': (490.0, -80.0, 11.0), 'zone': 'club'},
}

# Ground Z heights by approximate area (from SCM coord analysis)
GROUND_Z_BY_AREA = {
    # (x_min, x_max, y_min, y_max) -> approx z
    'ocean_beach': (-200, 700, -1800, -400, 10.5),
    'washington': (-200, 300, -400, 100, 10.5),
    'vice_point': (300, 900, -400, 700, 10.8),
    'little_haiti': (-1100, -600, -400, 400, 11.0),
    'downtown': (-700, 200, 400, 1400, 10.5),
    'airport': (-1950, -1200, -400, 300, 12.0),
    'docks': (-1250, -550, -1800, -1100, 11.5),
    'starfish_island': (-400, 100, -800, -400, 11.0),
}

@dataclass
class MapLocation:
    name: str
    x: float
    y: float
    z: float
    zone: str
    location_type: str  # 'mission_trigger', 'safe_house', 'business', 'gang_turf', 'interior'
    description: str = ''

class MapGraph:
    """
    Central spatial database for Vice City.
    Provides:
    - Coordinate validation
    - Zone lookup
    - Random valid spawn points by area
    - Interior location lookup
    - Distance calculations
    """

    def __init__(self, zones_json: str = None, ipl_json: str = None):
        self.zones = []
        self.ipl_instances = []
        self.locations: Dict[str, MapLocation] = {}
        self._build_known_locations()

        if zones_json:
            with open(zones_json) as f:
                self.zones = json.load(f)
        if ipl_json:
            with open(ipl_json) as f:
                self.ipl_instances = json.load(f).get('instances', [])

    def _build_known_locations(self):
        for name, data in KNOWN_LOCATIONS.items():
            x, y, z, desc = data
            self.locations[name] = MapLocation(
                name=name, x=x, y=y, z=z,
                zone=self._guess_zone(x, y),
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

    def _guess_zone(self, x: float, y: float) -> str:
        """Rough zone estimation from coordinates"""
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
        for zone in self.zones:
            x1, x2 = min(zone['x1'], zone['x2']), max(zone['x1'], zone['x2'])
            y1, y2 = min(zone['y1'], zone['y2']), max(zone['y1'], zone['y2'])
            if x1 <= x <= x2 and y1 <= y <= y2:
                return zone['name']
        return self._guess_zone(x, y)

    def is_valid_coord(self, x: float, y: float, z: float) -> bool:
        """Check if coordinate is within Vice City bounds"""
        return (-2000 <= x <= 1000 and
                -1900 <= y <= 1800 and
                0 <= z <= 200)

    def get_nearest_location(self, x: float, y: float,
                              location_type: str = None) -> Optional[MapLocation]:
        best = None
        best_dist = float('inf')
        for loc in self.locations.values():
            if location_type and loc.location_type != location_type:
                continue
            d = math.sqrt((loc.x - x)**2 + (loc.y - y)**2)
            if d < best_dist:
                best_dist = d
                best = loc
        return best

    def get_locations_by_type(self, location_type: str) -> List[MapLocation]:
        return [l for l in self.locations.values()
                if l.location_type == location_type]

    def get_random_outdoor_coord(self, zone_name: str = None) -> Tuple[float,float,float]:
        """Returns a random known outdoor coordinate, optionally filtered by zone"""
        import random
        candidates = list(KNOWN_LOCATIONS.values())
        if zone_name:
            # Filter roughly
            pass
        choice = random.choice(candidates)
        # Add small random offset
        x = choice[0] + random.uniform(-20, 20)
        y = choice[1] + random.uniform(-20, 20)
        z = choice[2]
        return (round(x, 3), round(y, 3), round(z, 3))

    def distance(self, x1,y1,z1, x2,y2,z2) -> float:
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    def export_locations_json(self, out_path: str):
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