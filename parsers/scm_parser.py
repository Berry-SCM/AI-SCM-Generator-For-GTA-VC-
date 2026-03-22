"""
Parses GTA VC decompiled SCM text format into structured data.
Handles: script blocks, opcodes, labels, variables, coordinates.

Supports both:
  - Full single-file main.txt
  - Segmented main[0]_1.txt, main[0]_2.txt, etc.
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class SCMInstruction:
    raw: str
    opcode: Optional[str] = None
    args: List[str] = field(default_factory=list)
    label: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class SCMScript:
    name: str
    label: str
    instructions: List[SCMInstruction] = field(default_factory=list)
    variables_used: List[str] = field(default_factory=list)
    coords_used: List[Tuple[float, float, float]] = field(default_factory=list)
    mission_index: Optional[int] = None


@dataclass
class SCMFile:
    objects: List[str] = field(default_factory=list)
    missions: List[Dict] = field(default_factory=list)
    scripts: List[SCMScript] = field(default_factory=list)
    global_vars: Dict[str, str] = field(default_factory=dict)


class SCMParser:
    LABEL_RE = re.compile(r'^:(\w+)')
    COMMENT_RE = re.compile(r'//(.*)$')
    DEFINE_OBJ_RE = re.compile(r'DEFINE OBJECT (\S+)')
    DEFINE_MISS_RE = re.compile(r'DEFINE MISSION (\d+) AT @(\w+)\s*(?://\s*(.*))?')
    COORD_RE = re.compile(r'(-?\d+\.?\d*),?\s+(-?\d+\.?\d*),?\s+(-?\d+\.?\d*)')
    VAR_RE = re.compile(r'(\$\w+|\d+@)')
    SCRIPT_NAME_RE = re.compile(r"script_name\s+'(\w+)'")

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.scm = SCMFile()

    def parse(self) -> SCMFile:
        with open(self.filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        lines = content.splitlines()
        self._parse_defines(lines)
        self._parse_scripts(lines)
        return self.scm

    def _parse_defines(self, lines: List[str]):
        for line in lines:
            line = line.strip()
            # Strip inline comment before matching
            comment_match = self.COMMENT_RE.search(line)
            if comment_match:
                line = line[:comment_match.start()].strip()
            obj_match = self.DEFINE_OBJ_RE.match(line)
            if obj_match:
                self.scm.objects.append(obj_match.group(1))
                continue
            miss_match = self.DEFINE_MISS_RE.match(line)
            if miss_match:
                self.scm.missions.append({
                    'index': int(miss_match.group(1)),
                    'label': miss_match.group(2),
                    'name': (miss_match.group(3) or '').strip()
                })

    def _parse_scripts(self, lines: List[str]):
        """
        Parse script blocks from decompiled SCM text.

        A new script block starts when:
          - A top-level label (:LABELNAME) is encountered while no script is active, OR
          - A 'script_name' opcode is encountered (always starts/names a new block)

        This correctly handles the structure of main.scm where each thread
        begins with :LABELNAME followed by script_name 'NAME'.
        """
        current_script: Optional[SCMScript] = None
        i = 0
        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.strip()

            # Extract and strip inline comment
            comment = None
            comment_match = self.COMMENT_RE.search(line)
            if comment_match:
                comment = comment_match.group(1).strip()
                line = line[:comment_match.start()].strip()

            # Detect script_name — this always defines/renames the current script
            sname_match = self.SCRIPT_NAME_RE.search(line)
            if sname_match:
                script_name = sname_match.group(1)
                if current_script is None:
                    # No label seen yet — create script now
                    current_script = SCMScript(name=script_name, label=script_name)
                    self.scm.scripts.append(current_script)
                else:
                    current_script.name = script_name
                i += 1
                continue

            # Detect label (:LABELNAME)
            label_match = self.LABEL_RE.match(line)
            if label_match:
                label = label_match.group(1)
                if current_script is None:
                    # Top-level label starts a new script block
                    current_script = SCMScript(name=label, label=label)
                    self.scm.scripts.append(current_script)
                # Sub-labels within a script are just recorded as instructions (below)

            # Record instruction in current script
            if current_script is not None and line:
                instr = SCMInstruction(raw=line, comment=comment)

                # Extract 3D coordinates from line
                for c in self.COORD_RE.findall(line):
                    try:
                        coord = (float(c[0]), float(c[1]), float(c[2]))
                        # Basic sanity check: discard obviously wrong coords
                        if (-2000 <= coord[0] <= 1000 and
                                -1900 <= coord[1] <= 1800 and
                                -10 <= coord[2] <= 300):
                            current_script.coords_used.append(coord)
                    except ValueError:
                        pass

                # Extract variable references
                for v in self.VAR_RE.findall(line):
                    if v not in current_script.variables_used:
                        current_script.variables_used.append(v)

                current_script.instructions.append(instr)

            i += 1
        return self.scm

    def to_json(self, out_path: str):
        data = {
            'objects': self.scm.objects,
            'missions': self.scm.missions,
            'scripts': [
                {
                    'name': s.name,
                    'label': s.label,
                    'instruction_count': len(s.instructions),
                    'coords': s.coords_used,
                    'variables': s.variables_used,
                    'raw': [instr.raw for instr in s.instructions]
                }
                for s in self.scm.scripts
            ]
        }
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[SCMParser] Written {len(self.scm.scripts)} scripts to {out_path}")
