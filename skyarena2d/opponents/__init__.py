from .base import BaseRuleOpponent
from .fix_rule_like import FixRuleLikeOpponent
from .fix_rule_v2 import FixRuleV2Opponent
from .no_attack_rule import NoAttackRuleOpponent
from .patrol_rule import PatrolRuleOpponent
from .random_rule import RandomRuleOpponent
from .rush_rule import RushRuleOpponent

RULES = {
    "no_attack_rule": NoAttackRuleOpponent,
    "rush_rule": RushRuleOpponent,
    "patrol_rule": PatrolRuleOpponent,
    "random_rule": RandomRuleOpponent,
    "fix_rule_like": FixRuleLikeOpponent,
    "fix_rule_v2": FixRuleV2Opponent,
}

__all__ = [
    "BaseRuleOpponent",
    "NoAttackRuleOpponent",
    "RushRuleOpponent",
    "PatrolRuleOpponent",
    "RandomRuleOpponent",
    "FixRuleLikeOpponent",
    "FixRuleV2Opponent",
    "RULES",
]
