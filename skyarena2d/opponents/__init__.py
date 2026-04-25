from .base import BaseRuleOpponent
from .fix_rule_like import FixRuleLikeOpponent
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
}

__all__ = [
    "BaseRuleOpponent",
    "NoAttackRuleOpponent",
    "RushRuleOpponent",
    "PatrolRuleOpponent",
    "RandomRuleOpponent",
    "FixRuleLikeOpponent",
    "RULES",
]
