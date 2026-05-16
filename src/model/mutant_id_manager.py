class MutantIDManager:
    """
    Gerenciador centralizado de IDs de mutantes.
    Singleton que garante IDs únicos e sequenciais em toda a execução.
    """
    _instance = None
    _counter: int = 0

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

    def reset(self) -> None:
        self._counter = 0

    @property
    def current(self) -> int:
        return self._counter