from dataclasses import dataclass

@dataclass
class ConfigLoader:
    """
    Dataclass to store mutation testing configurations.

    Validation is the responsibility of the MutationManager orchestrator.

    Attributes:
        program_path   : Path to the .py file of the PySpark application
        tests_path     : Path to the .py file containing pytest tests
        operators_list : List of mutation operator identifiers
    """
    program_path: str
    tests_path: str
    operators_list: list[str]

    def __repr__(self) -> str:
        return (
            f"ConfigLoader("
            f"program_path={self.program_path}, "
            f"tests_path={self.tests_path!r}, "
            f"operators_list={self.operators_list!r} "
            f")"
        )