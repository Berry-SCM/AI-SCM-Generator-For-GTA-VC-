"""
Parses GTA VC IPL (Item Placement List) files.
Extracts: object placements, positions, rotations, interior IDs.
IPL inst section: id, modelname, interior, x, y, z, rx, ry, rz, rw, lod
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class IPLInstance:
    obj_id: int
    model_name: str
    interior: int
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    rw: float
    lod: int

@dataclass
class IPLZone:
    name: str
    type: int
    x1: float; y1: float; z1: float
    x2: float; y2: float; z2: float
    island: int

class IPLParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.instances: List[IPLInstance] = []
        self.zones: List[IPLZone] = []

    def parse(self):
        with open(self.filepath, 'r', errors='ignore') as f:
            lines = f.readlines()

        section = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower() == 'inst':
                section = 'inst'
                continue
            elif line.lower() == 'zone':
                section = 'zone'
                continue
            elif line.lower() == 'end':
                section = None
                continue

            if section == 'inst':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 11:
                    try:
                        self.instances.append(IPLInstance(
                            obj_id=int(parts[0]),
                            model_name=parts[1],
                            interior=int(parts[2]),
                            x=float(parts[3]), y=float(parts[4]), z=float(parts[5]),
                            rx=float(parts[6]), ry=float(parts[7]),
                            rz=float(parts[8]), rw=float(parts[9]),
                            lod=int(parts[10])
                        ))
                    except (ValueError, IndexError):
                        pass

            elif section == 'zone':
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 9:
                    try:
                        self.zones.append(IPLZone(
                            name=parts[0], type=int(parts[1]),
                            x1=float(parts[2]), y1=float(parts[3]), z1=float(parts[4]),
                            x2=float(parts[5]), y2=float(parts[6]), z2=float(parts[7]),
                            island=int(parts[8])
                        ))
                    except (ValueError, IndexError):
                        pass
        return self

    def to_json(self, out_path: str):
        data = {
            'instances': [vars(i) for i in self.instances],
            'zones': [vars(z) for z in self.zones]
        }
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[IPLParser] {len(self.instances)} instances, {len(self.zones)} zones → {out_path}")