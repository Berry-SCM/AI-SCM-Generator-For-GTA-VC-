"""
Validates generated SCM scripts for:
- Balanced labels (every goto target exists)
- Valid coordinate ranges for Vice City
- Proper $onmission usage
- Presence of required structural patterns
"""

import re
from typing import List, Tuple
from spatial.map_graph import MapGraph

VALID_COORD_RANGE = {
    'x': (-2000, 1000),
    'y': (-1900, 1800),
    'z': (-10, 200),   # -10 to cover tunnels/underpasses; map_graph uses 0 for outdoor only
}

REQUIRED_PATTERNS = [
    (r'Player\.Defined\(\$player_char\)', 'Missing Player.Defined check'),
    (r'\$onmission', 'Missing $onmission usage'),
    (r'wait\s+\d+', 'Missing wait statement'),
]

COORD_RE = re.compile(
    r'(-?\d{1,5}\.?\d{0,3})\s+(-?\d{1,5}\.?\d{0,3})\s+(-?\d{1,3}\.?\d{0,3})'
)
LABEL_DEF_RE = re.compile(r'^:(\w+)', re.MULTILINE)
GOTO_RE = re.compile(r'goto(?:_if_false)?\s+@(\w+)')


class SCMValidator:
    def __init__(self, map_graph: MapGraph = None):
        self.map_graph = map_graph

    def validate(self, scm_text: str) -> Tuple[bool, List[str]]:
        issues = []

        # 1. Check all goto targets have a matching label definition
        defined_labels = set(LABEL_DEF_RE.findall(scm_text))
        goto_targets = set(GOTO_RE.findall(scm_text))
        missing_labels = goto_targets - defined_labels
        for lbl in sorted(missing_labels):
            issues.append(f'goto target @{lbl} has no matching label definition')

        # 2. Check coordinate ranges
        for m in COORD_RE.finditer(scm_text):
            try:
                x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
            except ValueError:
                continue
            xmin, xmax = VALID_COORD_RANGE['x']
            ymin, ymax = VALID_COORD_RANGE['y']
            zmin, zmax = VALID_COORD_RANGE['z']
            if not (xmin <= x <= xmax):
                issues.append(f'X coord out of range: {x}')
            if not (ymin <= y <= ymax):
                issues.append(f'Y coord out of range: {y}')
            if not (zmin <= z <= zmax):
                issues.append(f'Z coord out of range: {z}')

        # 3. Check required structural patterns
        for pattern, msg in REQUIRED_PATTERNS:
            if not re.search(pattern, scm_text):
                issues.append(msg)

        return (len(issues) == 0), issues

    def auto_fix_coords(self, scm_text: str) -> str:
        """Clamp obviously out-of-range coordinates to valid Vice City bounds."""
        def fix_coord(m):
            try:
                x = max(VALID_COORD_RANGE['x'][0], min(VALID_COORD_RANGE['x'][1], float(m.group(1))))
                y = max(VALID_COORD_RANGE['y'][0], min(VALID_COORD_RANGE['y'][1], float(m.group(2))))
                z = max(VALID_COORD_RANGE['z'][0], min(VALID_COORD_RANGE['z'][1], float(m.group(3))))
                return f'{x:.3f} {y:.3f} {z:.3f}'
            except ValueError:
                return m.group(0)
        return COORD_RE.sub(fix_coord, scm_text)
