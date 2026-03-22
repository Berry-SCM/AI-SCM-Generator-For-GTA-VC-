"""
Parses GTA VC decompiled SCM text format into structured data.
Handles: script blocks, opcodes, labels, variables, coordinates.

Script boundaries are determined by `script_name 'X'` opcodes, NOT by labels.
Labels (:FOO) within a script are just instructions.
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
    DEFINE_MISS_RE = re.compile(r'DEFINE MISSION (\d+) AT @(\w+)\s*//\s*(.*)')
    # Only match genuine float coordinate triples (each number must have a decimal point)
    COORD_RE = re.compile(
        r'(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)'
    )
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
            obj_match = self.DEFINE_OBJ_RE.match(line)
            if obj_match:
                self.scm.objects.append(obj_match.group(1))
                continue
            miss_match = self.DEFINE_MISS_RE.match(line)
            if miss_match:
                self.scm.missions.append({
                    'index': int(miss_match.group(1)),
                    'label': miss_match.group(2),
                    'name': miss_match.group(3).strip()
                })

    def _parse_scripts(self, lines: List[str]):
        """
        Script boundaries are `script_name 'X'` opcodes.
        Labels within a script are recorded as instructions, not new script starts.
        The entry label for a script is the last `:LABEL` seen before the script_name line.
        """
        current_script: Optional[SCMScript] = None
        pending_label: Optional[str] = None  # most recent :LABEL before a script_name

        for line in lines:
            raw_line = line.strip()

            # Strip comment for structural analysis
            comment = None
            comment_match = self.COMMENT_RE.search(raw_line)
            if comment_match:
                comment = comment_match.group(1).strip()
                clean_line = raw_line[:comment_match.start()].strip()
            else:
                clean_line = raw_line

            # Track the most recent label (used as the entry label when script_name follows)
            label_match = self.LABEL_RE.match(clean_line)
            if label_match:
                pending_label = label_match.group(1)

            # Detect script_name — this is the true script boundary
            sname_match = self.SCRIPT_NAME_RE.search(clean_line)
            if sname_match:
                script_name = sname_match.group(1)
                entry_label = pending_label if pending_label else script_name
                current_script = SCMScript(name=script_name, label=entry_label)
                self.scm.scripts.append(current_script)
                # Don't reset pending_label so the label is recorded as an instruction too
            
            # Add instruction to current script
            if current_script is not None and clean_line:
                instr = SCMInstruction(raw=clean_line, comment=comment)
                # Extract float coordinate triples
                for c in self.COORD_RE.findall(clean_line):
                    try:
                        coord = (float(c[0]), float(c[1]), float(c[2]))
                        current_script.coords_used.append(coord)
                    except ValueError:
                        pass
                # Extract variables
                for v in self.VAR_RE.findall(clean_line):
                    if v not in current_script.variables_used:
                        current_script.variables_used.append(v)
                current_script.instructions.append(instr)

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
                    'raw': [i.raw for i in s.instructions]
                }
                for s in self.scm.scripts
            ]
        }
        with open(out_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[SCMParser] Written {len(self.scm.scripts)} scripts to {out_path}")
