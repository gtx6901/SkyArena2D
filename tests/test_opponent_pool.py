from __future__ import annotations

import numpy as np

from skyarena2d.opponents import RULES
from skyarena2d.opponents.pool import RuleOpponentPool


def test_unseen_opponents_start_uniform() -> None:
    pool = RuleOpponentPool(["no_attack_rule", "fix_rule_v2"], RULES, seed=1)
    np.testing.assert_allclose(pool.probabilities(), [0.5, 0.5])


def test_sampling_focuses_on_unsolved_boundary_not_easy_opponent() -> None:
    pool = RuleOpponentPool(
        ["no_attack_rule", "fix_rule_v2"],
        RULES,
        seed=1,
        uniform_mix=0.0,
    )
    for _ in range(20):
        pool.record_result("no_attack_rule", "red")
    for winner in ["red", "blue"] * 10:
        pool.record_result("fix_rule_v2", winner)
    probs = pool.probabilities()
    assert probs[1] > probs[0]


def test_pool_state_round_trip() -> None:
    pool = RuleOpponentPool(["fix_rule_v2"], RULES)
    pool.record_result("fix_rule_v2", "draw")
    state = pool.state_dict()
    restored = RuleOpponentPool(["fix_rule_v2"], RULES)
    restored.load_state_dict(state)
    assert restored.records["fix_rule_v2"].games == 1
    assert restored.records["fix_rule_v2"].red_win_rate == 0.5
