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

        # Check label consistency: every goto target must have a label definition
        defined_labels = set(LABEL_DEF_RE.findall(scm_text))
        goto_labels = set(GOTO_RE.findall(scm_text))
        missing_labels = goto_labels - defined_labels
        if missing_labels:
            errors.append(f"Missing label definitions: {missing_labels}")

        # Check coordinates are within Vice City bounds
        coords = COORD_RE.findall(scm_text)
        suspicious_coords = []
        for x, y, z in coords:
            try:
                fx, fy, fz = float(x), float(y), float(z)
                out_of_range = False
                if not (VALID_COORD_RANGE['x'][0] <= fx <= VALID_COORD_RANGE['x'][1]):
                    out_of_range = True
                if not (VALID_COORD_RANGE['y'][0] <= fy <= VALID_COORD_RANGE['y'][1]):
                    out_of_range = True
                if not (VALID_COORD_RANGE['z'][0] <= fz <= VALID_COORD_RANGE['z'][1]):
                    out_of_range = True
                if out_of_range:
                    suspicious_coords.append(f"({fx}, {fy}, {fz})")
            except ValueError:
                pass
        if suspicious_coords:
            # Deduplicate and cap output
            unique = list(dict.fromkeys(suspicious_coords))[:5]
            warnings.append(f"Suspicious out-of-range coords: {unique}")

        # Check structural patterns (warnings only — not all files need all patterns)
        for pattern, msg in REQUIRED_PATTERNS:
            if not re.search(pattern, scm_text):
                warnings.append(msg)

        is_valid = len(errors) == 0
        return is_valid, errors + warnings

    def auto_fix_coords(self, scm_text: str) -> str:
        """Clamp obviously out-of-range coordinates to valid Vice City bounds."""
        def fix_coord(m):
            try:
                x, y, z = float(m.group(1)), float(m.group(2)), float(m.group(3))
                x = max(VALID_COORD_RANGE['x'][0], min(VALID_COORD_RANGE['x'][1], x))
                y = max(VALID_COORD_RANGE['y'][0], min(VALID_COORD_RANGE['y'][1], y))
                z = max(VALID_COORD_RANGE['z'][0], min(VALID_COORD_RANGE['z'][1], z))
                return f"{x:.3f} {y:.3f} {z:.3f}"
            except ValueError:
                return m.group(0)
        return COORD_RE.sub(fix_coord, scm_text)
