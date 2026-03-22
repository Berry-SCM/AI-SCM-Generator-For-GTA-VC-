"""
Parses GTA VC .ZON files (info.zon, map.zon, navig.zon).
Zone format: name, type, x1, y1, z1, x2, y2, z2, island
"""

import json
from dataclasses import dataclass
from typing import List

@dataclass
class Zone:
    name: str
    zone_type: int
    x1: float; y1: float; z1: float
    x2: float; y2: float; z2: float
    island: int

    @property
    def center(self):
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
            (self.z1 + self.z2) / 2
        )

    def contains(self, x: float, y: float) -> bool:
        return (min(self.x1,self.x2) <= x <= max(self.x1,self.x2) and
                min(self.y1,self.y2) <= y <= max(self.y1,self.y2))

class ZonParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.zones: List[Zone] = []

    def parse(self):
        with open(self.filepath, 'r', errors='ignore') as f:
            lines = f.readlines()

        in_zone = False
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower() == 'zone':
                in_zone = True
                continue
            if line.lower() == 'end':
                in_zone = False
                continue
            if in_zone:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    try:
                        self.zones.append(Zone(
                            name=parts[0],
                            zone_type=int(parts[1]),
                            x1=float(parts[2]), y1=float(parts[3]), z1=float(parts[4]),
                            x2=float(parts[5]), y2=float(parts[6]), z2=float(parts[7]),
                            island=int(parts[8])
                        ))
                    except (ValueError, IndexError):
                        pass
        return self

    def get_zone_for_coord(self, x: float, y: float) -> List[Zone]:
        return [z for z in self.zones if z.contains(x, y)]

    def to_json(self, out_path: str):
        data = [vars(z) for z in self.zones]
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[ZonParser] {len(self.zones)} zones → {out_path}")