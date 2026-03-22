"""
Parses GTA VC IDE files for object/car/ped/weapon definitions.
Extracts model IDs, names, and types.
"""

import json
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class IDEEntry:
    id: int
    model_name: str
    txd_name: str
    entry_type: str   # 'obj', 'car', 'peds', 'weap', 'hier'
    extra: Dict

class IDEParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.entries: List[IDEEntry] = []

    def parse(self):
        with open(self.filepath, 'r', errors='ignore') as f:
            lines = f.readlines()

        section = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            low = line.lower()
            if low in ('objs','cars','peds','weap','hier','tobj','anim','2dfx'):
                section = low
                continue
            if low == 'end':
                section = None
                continue
            if section:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    try:
                        entry_id = int(parts[0])
                        model = parts[1] if len(parts) > 1 else ''
                        txd = parts[2] if len(parts) > 2 else ''
                        self.entries.append(IDEEntry(
                            id=entry_id,
                            model_name=model,
                            txd_name=txd,
                            entry_type=section,
                            extra={'raw': line}
                        ))
                    except (ValueError, IndexError):
                        pass
        return self

    def get_cars(self) -> List[IDEEntry]:
        return [e for e in self.entries if e.entry_type == 'cars']

    def get_peds(self) -> List[IDEEntry]:
        return [e for e in self.entries if e.entry_type == 'peds']

    def get_weapons(self) -> List[IDEEntry]:
        return [e for e in self.entries if e.entry_type == 'weap']

    def to_json(self, out_path: str):
        data = [{'id': e.id, 'model': e.model_name, 'type': e.entry_type}
                for e in self.entries]
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[IDEParser] {len(self.entries)} entries → {out_path}")