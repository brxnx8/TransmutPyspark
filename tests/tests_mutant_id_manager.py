from __future__ import annotations

import pytest

from src.model.mutant_id_manager import MutantIDManager


# ===========================================================================
# Fixture de isolamento — obrigatória para testes de Singleton
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """
    Reseta o estado do Singleton antes e depois de cada teste.
    Como MutantIDManager é um Singleton com estado de classe, sem esse
    isolamento os testes se contaminariam mutuamente.
    """
    MutantIDManager._instance = None
    MutantIDManager._counter = 0
    yield
    MutantIDManager._instance = None
    MutantIDManager._counter = 0


# ===========================================================================
# Comportamento Singleton
# ===========================================================================

class TestMutantIDManagerSingleton:

    def test_should_return_same_instance_when_instantiated_twice(self):
        m1 = MutantIDManager()
        m2 = MutantIDManager()
        assert m1 is m2

    def test_should_return_same_instance_when_instantiated_multiple_times(self):
        instances = [MutantIDManager() for _ in range(10)]
        assert all(i is instances[0] for i in instances)

    def test_should_share_counter_state_between_references(self):
        m1 = MutantIDManager()
        m2 = MutantIDManager()
        m1.next_id()
        assert m2.current == 1

    def test_should_persist_instance_after_first_instantiation(self):
        first = MutantIDManager()
        _ = MutantIDManager()
        assert MutantIDManager._instance is first


# ===========================================================================
# next_id — geração sequencial de IDs
# ===========================================================================

class TestMutantIDManagerNextId:

    def test_should_return_one_as_first_id(self):
        m = MutantIDManager()
        assert m.next_id() == 1

    def test_should_return_sequential_ids_on_consecutive_calls(self):
        m = MutantIDManager()
        ids = [m.next_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]

    def test_should_increment_by_one_on_each_call(self):
        m = MutantIDManager()
        first = m.next_id()
        second = m.next_id()
        assert second - first == 1

    def test_should_return_unique_ids_on_every_call(self):
        m = MutantIDManager()
        ids = [m.next_id() for _ in range(100)]
        assert len(ids) == len(set(ids))

    def test_should_produce_ids_starting_from_one_not_zero(self):
        m = MutantIDManager()
        assert m.next_id() != 0

    def test_should_continue_sequence_across_different_references(self):
        m1 = MutantIDManager()
        m2 = MutantIDManager()
        m1.next_id()   # 1
        m1.next_id()   # 2
        result = m2.next_id()  # deve ser 3
        assert result == 3

    def test_should_return_integer_type(self):
        m = MutantIDManager()
        assert isinstance(m.next_id(), int)

    def test_should_generate_large_sequence_without_collision(self):
        m = MutantIDManager()
        ids = [m.next_id() for _ in range(1000)]
        assert ids == list(range(1, 1001))


# ===========================================================================
# reset — reinicialização do contador
# ===========================================================================

class TestMutantIDManagerReset:

    def test_should_reset_counter_to_zero(self):
        m = MutantIDManager()
        m.next_id()
        m.next_id()
        m.reset()
        assert m.current == 0

    def test_should_restart_sequence_from_one_after_reset(self):
        m = MutantIDManager()
        m.next_id()
        m.next_id()
        m.reset()
        assert m.next_id() == 1

    def test_should_allow_multiple_resets_in_sequence(self):
        m = MutantIDManager()
        for _ in range(3):
            m.next_id()
            m.reset()
            assert m.current == 0

    def test_should_reset_when_counter_is_already_zero(self):
        m = MutantIDManager()
        m.reset()  # não deve lançar exceção
        assert m.current == 0

    def test_should_reflect_reset_across_all_references(self):
        m1 = MutantIDManager()
        m2 = MutantIDManager()
        m1.next_id()
        m1.next_id()
        m1.reset()
        assert m2.current == 0

    def test_should_return_none_from_reset(self):
        m = MutantIDManager()
        result = m.reset()
        assert result is None

    def test_should_produce_fresh_sequence_after_reset_and_multiple_calls(self):
        m = MutantIDManager()
        [m.next_id() for _ in range(5)]
        m.reset()
        ids = [m.next_id() for _ in range(5)]
        assert ids == [1, 2, 3, 4, 5]


# ===========================================================================
# current — propriedade de leitura do estado
# ===========================================================================

class TestMutantIDManagerCurrent:

    def test_should_return_zero_before_any_call_to_next_id(self):
        m = MutantIDManager()
        assert m.current == 0

    def test_should_reflect_value_of_last_generated_id(self):
        m = MutantIDManager()
        last = None
        for _ in range(5):
            last = m.next_id()
        assert m.current == last

    def test_should_not_increment_counter_when_accessed(self):
        m = MutantIDManager()
        m.next_id()
        before = m.current
        _ = m.current
        _ = m.current
        assert m.current == before

    def test_should_return_zero_after_reset(self):
        m = MutantIDManager()
        m.next_id()
        m.reset()
        assert m.current == 0

    def test_should_return_integer_type(self):
        m = MutantIDManager()
        assert isinstance(m.current, int)

    def test_should_equal_next_id_result_after_each_call(self):
        m = MutantIDManager()
        for _ in range(10):
            generated = m.next_id()
            assert m.current == generated

    def test_should_be_read_only_property(self):
        m = MutantIDManager()
        with pytest.raises(AttributeError):
            m.current = 99  # type: ignore[misc]


# ===========================================================================
# Integração: fluxo completo de uso pelo MutationManager
# ===========================================================================

class TestMutantIDManagerIntegration:

    def test_should_generate_unique_ids_for_simulated_batch_of_mutants(self):
        manager = MutantIDManager()
        mutant_ids = [manager.next_id() for _ in range(20)]
        assert mutant_ids == list(range(1, 21))
        assert len(set(mutant_ids)) == 20

    def test_should_resume_correct_sequence_after_partial_generation_and_reset(self):
        m = MutantIDManager()
        first_batch = [m.next_id() for _ in range(3)]   # [1, 2, 3]
        m.reset()
        second_batch = [m.next_id() for _ in range(3)]  # [1, 2, 3]
        assert first_batch == second_batch == [1, 2, 3]

    def test_should_maintain_consistency_between_current_and_next_id_throughout(self):
        m = MutantIDManager()
        assert m.current == 0
        for expected in range(1, 6):
            nid = m.next_id()
            assert nid == expected
            assert m.current == expected
        m.reset()
        assert m.current == 0
        assert m.next_id() == 1