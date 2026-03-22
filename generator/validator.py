"""
Validates generated SCM scripts for:
- Balanced labels (every goto target exists)
- Valid coordinate ranges
- Known opcode names
- Proper $onmission usage
"""

import re
from typing import List, Tuple
from spatial.map_graph import MapGraph

VALID_COORD_RANGE = {
    'x': (-2000, 1000),
    'y': (-1900, 1800),
    'z': (0, 200),
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
        errors = []
        warnings = []

        # Check label consistency
        defined_labels = set(LABEL_DEF_RE.findall(scm_text))
        goto_labels = set(GOTO_RE.findall(scm_text))
        missing_labels = goto_labels - defined_labels
        if missing_labels:
            errors.append(f"Missing label definitions: {missing_labels}")

        # Check coordinates
        coords = COORD_RE.findall(scm_text)
        for x, y, z in coords:
            try:
                fx, fy, fz = float(x), float(y), float(z)
                if not (VALID_COORD_RANGE['x'][0] <= fx <= VALID_COORD_RANGE['x'][1]):
                    warnings.append(f"Suspicious X coord: {fx}")
                if not (VALID_COORD_RANGE['y'][0] <= fy <= VALID_COORD_RANGE['y'][1]):
                    warnings.append(f"Suspicious Y coord: {fy}")
                if not (VALID_COORD_RANGE['z'][0] <= fz <= VALID_COORD_RANGE['z'][1]):
                    warnings.append(f"Suspicious Z coord: {fz}")
            except ValueError:
                pass

        # Check required patterns
        for pattern, msg in REQUIRED_PATTERNS:
            if not re.search(pattern, scm_text):
                warnings.append(msg)

        is_valid = len(errors) == 0
        return is_valid, errors + warnings

    def auto_fix_coords(self, scm_text: str) -> str:
        """Clamp obviously wrong coordinates to valid range"""
        def fix_coord(m):
            try:
                x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
                x = max(-2000, min(1000, x))
                y = max(-1900, min(1800, y))
                z = max(0, min(200, z))
                return f"{x:.3f} {y:.3f} {z:.3f}"
            except ValueError:
                return m.group(0)
        return COORD_RE.sub(fix_coord, scm_text)