from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Mutant:
    id:             int
    operator:       str
    original_path:  str
    mutant_path:    str
    modified_line:  str
    test_files:     list[Path] = field(default_factory=list)
    test_functions: list[str]  = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Mutant("
            f"id={self.id}, "
            f"operator={self.operator!r}, "
            f"original={self.original_path!r}, "
            f"mutant={self.mutant_path!r}, "
            f"line={self.modified_line!r}, "
            f"test_files={[f.name for f in self.test_files]}, "
            f"test_functions={self.test_functions}"
            f")"
        )
