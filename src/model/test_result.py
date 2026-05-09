from dataclasses import dataclass

@dataclass
class TestResult:
    """
    Dataclass to store the details of tests results.


    Attributes:
        mutant          : Unique identifier for the mutant (id from Mutant dataclass)
        status          : Killed, Survived, Timeout, etc.
        failed_tests    : Which tests failed (if applicable)
        execution_time  : Time taken to execute the tests against the mutant
    """
    mutant: int
    status: str
    failed_tests: list[str]
    execution_time: float

    def __repr__(self) -> str:
        return (
            f"ConfigLoader("
            f"mutant={self.mutant}, "
            f"status={self.status!r}, "
            f"failed_tests={self.failed_tests!r}, "
            f"execution_time={self.execution_time!r}"
            f")"
        )