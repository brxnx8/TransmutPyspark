from dataclasses import dataclass

@dataclass
class TestResult:
    mutant: int
    status: str
    failed_tests: list[str]
    execution_time: float

    def __repr__(self) -> str:
        return (
            f"TestResult("
            f"mutant={self.mutant}, "
            f"status={self.status!r}, "
            f"failed_tests={self.failed_tests!r}, "
            f"execution_time={self.execution_time!r}"
            f")"
        )