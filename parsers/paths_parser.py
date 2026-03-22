"""
Parses GTA Vice City path node files from DATA/paths/
These are the actual road (car) and ped (foot) navigation nodes the game uses.
Parsing them gives us real drivable/walkable coordinates.

Files: DATA/paths/ROADBLOCKS.DAT, DATA/paths/CARREC.IMG (binary — skip),
       DATA/paths/*.DAT text files (ROADBLOCKS.DAT is text; individual area path
       data is embedded in IPL files under the 'path' section).

The path data most useful to us is in the IPL files under the 'path' section —
already handled partially by ipl_parser.py. However, GTA VC also stores
simplified path nodes in the NAVIG.ZON and individual area .ZON files.

This parser reads the 'path' sections from IPL files (which ipl_parser.py
currently skips) and extracts node coordinates as drivable/walkable points.

Format of 'path' section in IPL:
  path
  <node_type>, <node_id>, <x>, <y>, <z>, <link_id>, ...
  end

node_type: 0 = car node, 1 = ped node, 2 = boat node
"""
import os
import glob
import json
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PathNode:
    node_type: int      # 0=car/road, 1=ped/foot, 2=boat
    node_id: int
    x: float
    y: float
    z: float
    source_file: str = ''

    @property
    def type_name(self) -> str:
        return {0: 'road', 1: 'ped', 2: 'boat'}.get(self.node_type, 'unknown')


class PathsParser:
    """
    Parses path nodes from:
    1. IPL files — 'path' sections (car + ped nodes per area)
    2. NAVIG.ZON — simplified navigation zone hints

    These give us real coordinates that are confirmed drivable or walkable.
    """

    # IPL path section line formats vary; handle both:
    # type, id, x, y, z[, link...]
    # or just: x y z (simplified)
    PATH_SECTION_RE = re.compile(
        r'^\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',
        re.MULTILINE
    )
    # Simpler fallback: just coordinates on a line
    COORD_LINE_RE = re.compile(
        r'^\s*(-?\d{1,5}\.?\d{0,4})\s*,\s*(-?\d{1,5}\.?\d{0,4})\s*,\s*(-?\d{1,3}\.?\d{0,4})'
    )

    def __init__(self):
        self.road_nodes: List[PathNode] = []
        self.ped_nodes: List[PathNode] = []
        self.boat_nodes: List[PathNode] = []

    def parse_ipl_file(self, filepath: str):
        """Extract path nodes from a single IPL file's 'path' section."""
        try:
            with open(filepath, encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception:
            return

        fname = os.path.basename(filepath)
        in_path = False
        node_count = 0

        for line in content.splitlines():
            stripped = line.strip().lower()

            if stripped == 'path':
                in_path = True
                continue
            if in_path and stripped == 'end':
                in_path = False
                continue

            if not in_path:
                continue

            # Skip comments
            if stripped.startswith('#') or stripped.startswith('//'):
                continue

            # Try full format: type, id, x, y, z
            m = self.PATH_SECTION_RE.match(line)
            if m:
                node_type = int(m.group(1))
                node_id = int(m.group(2))
                x = float(m.group(3))
                y = float(m.group(4))
                z = float(m.group(5))

                node = PathNode(
                    node_type=node_type,
                    node_id=node_id,
                    x=x, y=y, z=z,
                    source_file=fname
                )
                self._store_node(node)
                node_count += 1
                continue

            # Fallback: bare x, y, z line inside path section
            m2 = self.COORD_LINE_RE.match(line)
            if m2:
                x = float(m2.group(1))
                y = float(m2.group(2))
                z = float(m2.group(3))
                # Default to road node if type unknown
                node = PathNode(
                    node_type=0,
                    node_id=-1,
                    x=x, y=y, z=z,
                    source_file=fname
                )
                self._store_node(node)
                node_count += 1

        if node_count > 0:
            pass  # parsed silently

    def parse_all_ipls(self, maps_dir: str):
        """Parse all .IPL files in a directory tree for path sections."""
        ipl_files = (
            glob.glob(os.path.join(maps_dir, '**', '*.IPL'), recursive=True) +
            glob.glob(os.path.join(maps_dir, '**', '*.ipl'), recursive=True)
        )
        for fp in ipl_files:
            self.parse_ipl_file(fp)

        print(f"[PathsParser] Road nodes: {len(self.road_nodes)}, "
              f"Ped nodes: {len(self.ped_nodes)}, "
              f"Boat nodes: {len(self.boat_nodes)}")

    def parse_paths_dat_dir(self, paths_dir: str):
        """
        Parse any text-format .DAT files from DATA/paths/ directory.
        VC stores ROADBLOCKS.DAT and similar — parse coords from them.
        """
        if not os.path.isdir(paths_dir):
            return

        dat_files = glob.glob(os.path.join(paths_dir, '*.DAT')) + \
                    glob.glob(os.path.join(paths_dir, '*.dat'))

        for fp in dat_files:
            fname = os.path.basename(fp)
            try:
                with open(fp, encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
            except Exception:
                continue

            count = 0
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('//'):
                    continue

                # ROADBLOCKS.DAT format: x y z rx ry rz model_id
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                        # Sanity-check VC world bounds
                        if -2000 < x < 1500 and -2000 < y < 2000:
                            node = PathNode(
                                node_type=0,  # assume road
                                node_id=-1,
                                x=x, y=y, z=z,
                                source_file=fname
                            )
                            self._store_node(node)
                            count += 1
                    except ValueError:
                        pass

            if count > 0:
                print(f"[PathsParser] {fname}: {count} nodes")

    def _store_node(self, node: PathNode):
        if node.node_type == 0:
            self.road_nodes.append(node)
        elif node.node_type == 1:
            self.ped_nodes.append(node)
        elif node.node_type == 2:
            self.boat_nodes.append(node)
        else:
            self.road_nodes.append(node)  # default

    def get_all_nodes(self) -> List[PathNode]:
        return self.road_nodes + self.ped_nodes + self.boat_nodes

    def get_road_coords(self) -> List[Tuple[float, float, float]]:
        return [(n.x, n.y, n.z) for n in self.road_nodes]

    def get_ped_coords(self) -> List[Tuple[float, float, float]]:
        return [(n.x, n.y, n.z) for n in self.ped_nodes]

    def to_json(self, out_path: str):
        """Save parsed nodes to JSON for use by MapGraph."""
        data = {
            'road_nodes': [
                {'x': n.x, 'y': n.y, 'z': n.z, 'id': n.node_id, 'source': n.source_file}
                for n in self.road_nodes
            ],
            'ped_nodes': [
                {'x': n.x, 'y': n.y, 'z': n.z, 'id': n.node_id, 'source': n.source_file}
                for n in self.ped_nodes
            ],
            'boat_nodes': [
                {'x': n.x, 'y': n.y, 'z': n.z, 'id': n.node_id, 'source': n.source_file}
                for n in self.boat_nodes
            ],
        }
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(data, f)
        total = len(self.road_nodes) + len(self.ped_nodes) + len(self.boat_nodes)
        print(f"[PathsParser] Saved {total} total nodes → {out_path}")
