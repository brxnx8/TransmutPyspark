from dataclasses import dataclass
from pathlib import Path

@dataclass
class Mutant:
    """
    Stores the details of a single generated mutant.

    Attributes
    ----------
    id            : Unique identifier for the mutant.
    operator      : Name of the mutation operator that produced this mutant.
    original_path : Path to the original .py file of the PySpark application.
    mutant_path   : Path to the mutated .py file written to disk.
    modified_line : The line of code that was modified to create the mutant.
    """

    id:            int
    operator:      str
    original_path: str
    mutant_path:   str
    modified_line: str

    def __repr__(self) -> str:
        return (
            f"Mutant("
            f"id={self.id}, "
            f"operator={self.operator!r}, "
            f"original={self.original_path!r}, "
            f"mutant={self.mutant_path!r}, "
            f"line={self.modified_line!r}"
            f")"
        )