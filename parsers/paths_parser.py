"""
Parses GTA Vice City path files from DATA/paths/ and IPL path sections.

DATA/paths/ contains:
  flight.dat, flight2.dat, flight3.dat  -- AI plane waypoints (aerial, z~450)
  spath0.dat                             -- boat spawn path nodes

These are NOT road/ped nav nodes. VC road/ped pathfinding is compiled binary
(inside gta3.img) and cannot be parsed as plain text.

What this parser actually does:
  1. Parse 'path' sections from IPL files (car=0, ped=1 nodes per map area)
  2. Parse flight.dat/flight2.dat/flight3.dat as aerial waypoints (type=3, ignored
     for ground coord selection)
  3. Parse spath0.dat as boat nodes (type=2)

Ground drivable/walkable coords come from IPL path sections only.
The paths/ DAT files are aerial/boat — not usable for mission ground placement.
"""
import os
import glob
import json
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PathNode:
    node_type: int      # 0=car/road, 1=ped/foot, 2=boat, 3=flight(aerial)
    node_id: int
    x: float
    y: float
    z: float
    source_file: str = ''

    @property
    def type_name(self) -> str:
        return {0: 'road', 1: 'ped', 2: 'boat', 3: 'flight'}.get(self.node_type, 'unknown')


class PathsParser:
    """
    Parses path nodes from:
    1. IPL files — 'path' sections (car + ped nodes per area) — GROUND coords
    2. DATA/paths/*.dat files:
       - flight*.dat  → aerial waypoints (z ~450), stored as type=3, NOT used for ground
       - spath0.dat   → boat path nodes, stored as type=2
    """

    # IPL path section: type, id, x, y, z[, ...]
    PATH_SECTION_RE = re.compile(
        r'^\s*(\d+)\s*,\s*(\d+)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)',
        re.MULTILINE
    )
    # Fallback: bare x, y, z line inside path section
    COORD_LINE_RE = re.compile(
        r'^\s*(-?\d{1,5}\.?\d{0,4})\s*,\s*(-?\d{1,5}\.?\d{0,4})\s*,\s*(-?\d{1,3}\.?\d{0,4})'
    )
    # Space-separated x y z (used in flight.dat and spath0.dat)
    SPACE_COORD_RE = re.compile(
        r'^\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*$'
    )

    def __init__(self):
        self.road_nodes: List[PathNode] = []
        self.ped_nodes: List[PathNode] = []
        self.boat_nodes: List[PathNode] = []
        self.flight_nodes: List[PathNode] = []  # aerial only, not used for ground placement

    # ── IPL path sections ────────────────────────────────────────────────────

    def parse_ipl_file(self, filepath: str):
        """Extract path nodes from a single IPL file's 'path' section."""
        try:
            with open(filepath, encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception:
            return

        fname = os.path.basename(filepath)
        in_path = False

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
            if stripped.startswith('#') or stripped.startswith('//'):
                continue

            # Full format: type, id, x, y, z
            m = self.PATH_SECTION_RE.match(line)
            if m:
                node_type = int(m.group(1))
                node_id = int(m.group(2))
                x, y, z = float(m.group(3)), float(m.group(4)), float(m.group(5))
                node = PathNode(node_type=node_type, node_id=node_id,
                                x=x, y=y, z=z, source_file=fname)
                self._store_node(node)
                continue

            # Fallback: bare x, y, z
            m2 = self.COORD_LINE_RE.match(line)
            if m2:
                x, y, z = float(m2.group(1)), float(m2.group(2)), float(m2.group(3))
                node = PathNode(node_type=0, node_id=-1,
                                x=x, y=y, z=z, source_file=fname)
                self._store_node(node)

    def parse_all_ipls(self, maps_dir: str):
        """Parse all .IPL files in a directory tree for path sections."""
        ipl_files = (
            glob.glob(os.path.join(maps_dir, '**', '*.IPL'), recursive=True) +
            glob.glob(os.path.join(maps_dir, '**', '*.ipl'), recursive=True)
        )
        for fp in ipl_files:
            self.parse_ipl_file(fp)

        print(f"[PathsParser] From IPL path sections — "
              f"Road: {len(self.road_nodes)}, "
              f"Ped: {len(self.ped_nodes)}, "
              f"Boat: {len(self.boat_nodes)}")

    # ── DATA/paths/ DAT files ────────────────────────────────────────────────

    def parse_paths_dat_dir(self, paths_dir: str):
        """
        Parse DAT files from DATA/paths/:

        flight.dat / flight2.dat / flight3.dat:
            Space-separated x y z lines. First line is node count.
            Z is ~450 (aerial). Stored as type=3 (flight), NOT used for ground coords.

        spath0.dat:
            Boat spawn path. Stored as type=2 (boat).

        Any other .dat files: attempted as space-separated x y z.
        Ground-level coords (z < 100) stored as road nodes (type=0).
        """
        if not os.path.isdir(paths_dir):
            print(f"[PathsParser] paths/ dir not found at {paths_dir} — skipping")
            return

        dat_files = (glob.glob(os.path.join(paths_dir, '*.DAT')) +
                     glob.glob(os.path.join(paths_dir, '*.dat')))

        for fp in dat_files:
            fname = os.path.basename(fp).lower()
            self._parse_dat_file(fp, fname)

    def _parse_dat_file(self, filepath: str, fname: str):
        """Parse a single DAT file from the paths/ directory."""
        try:
            with open(filepath, encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        except Exception:
            return

        # Determine file type from name
        is_flight = fname.startswith('flight')
        is_boat   = fname.startswith('spath')

        count = 0
        skip_first = False

        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            # First non-comment line of flight.dat is the node count integer
            if i == 0 and line.isdigit():
                skip_first = True
                continue
            if skip_first and i == 1 and line.isdigit():
                continue

            # Try space-separated x y z
            m = self.SPACE_COORD_RE.match(line)
            if m:
                x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))

                # Sanity check: within expanded VC world + sky bounds
                if not (-2500 < x < 2500 and -2500 < y < 2500 and -50 < z < 1000):
                    continue

                if is_flight:
                    # Aerial waypoints — store as flight type, not used for ground
                    node = PathNode(node_type=3, node_id=count,
                                    x=x, y=y, z=z, source_file=fname)
                    self.flight_nodes.append(node)
                elif is_boat:
                    node = PathNode(node_type=2, node_id=count,
                                    x=x, y=y, z=z, source_file=fname)
                    self.boat_nodes.append(node)
                else:
                    # Unknown dat: use z to classify
                    if z > 100:
                        # Aerial — store as flight
                        node = PathNode(node_type=3, node_id=count,
                                        x=x, y=y, z=z, source_file=fname)
                        self.flight_nodes.append(node)
                    else:
                        # Ground-level — treat as road
                        node = PathNode(node_type=0, node_id=count,
                                        x=x, y=y, z=z, source_file=fname)
                        self.road_nodes.append(node)
                count += 1

        if count > 0:
            kind = 'flight' if is_flight else ('boat' if is_boat else 'other')
            print(f"[PathsParser] {os.path.basename(filepath)}: "
                  f"{count} nodes ({kind})")

    # ── Storage and output ───────────────────────────────────────────────────

    def _store_node(self, node: PathNode):
        if node.node_type == 0:
            self.road_nodes.append(node)
        elif node.node_type == 1:
            self.ped_nodes.append(node)
        elif node.node_type == 2:
            self.boat_nodes.append(node)
        elif node.node_type == 3:
            self.flight_nodes.append(node)
        else:
            self.road_nodes.append(node)

    def get_all_nodes(self) -> List[PathNode]:
        return self.road_nodes + self.ped_nodes + self.boat_nodes

    def get_road_coords(self) -> List[Tuple[float, float, float]]:
        return [(n.x, n.y, n.z) for n in self.road_nodes]

    def get_ped_coords(self) -> List[Tuple[float, float, float]]:
        return [(n.x, n.y, n.z) for n in self.ped_nodes]

    def to_json(self, out_path: str):
        """
        Save parsed nodes to JSON for use by MapGraph.
        Flight nodes are saved separately and will NOT be used for ground coords.
        """
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
            'flight_nodes': [
                {'x': n.x, 'y': n.y, 'z': n.z, 'id': n.node_id, 'source': n.source_file}
                for n in self.flight_nodes
            ],
        }
        os.makedirs(
            os.path.dirname(out_path) if os.path.dirname(out_path) else '.',
            exist_ok=True
        )
        with open(out_path, 'w') as f:
            json.dump(data, f)

        ground_total = len(self.road_nodes) + len(self.ped_nodes)
        print(f"[PathsParser] Saved → {out_path}")
        print(f"  Ground nodes (usable): road={len(self.road_nodes)}, ped={len(self.ped_nodes)}")
        print(f"  Other: boat={len(self.boat_nodes)}, flight={len(self.flight_nodes)} (aerial, not used for placement)")
        if ground_total == 0:
            print(f"  NOTE: No ground path nodes found in IPL path sections.")
            print(f"  MapGraph will fall back to hardcoded known locations for coord selection.")
            print(f"  This is normal for VC — road/ped nav is binary, not in text files.")
