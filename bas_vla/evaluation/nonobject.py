"""Destination/relation predicates and strict first-completion subgoal order."""
from __future__ import annotations
from .dual_goal import classify_hits


def validate_goal(spec):
    if spec.get('kind') not in {'all', 'ordered'}:
        raise ValueError('goal kind must be all or ordered')
    predicates = spec.get('predicates')
    if not isinstance(predicates, list) or not predicates:
        raise ValueError('goal needs nonempty predicates')
    if any(not isinstance(p, list) or len(p) < 2 or any(not isinstance(x, str) or not x for x in p) for p in predicates):
        raise ValueError('predicates are nonempty string argument lists')
    if len({tuple(p) for p in predicates}) != len(predicates):
        raise ValueError('duplicate subgoal predicates')
    if spec['kind'] == 'ordered' and len(predicates) < 2:
        raise ValueError('ordered goal needs at least two distinct subgoals')


class GoalTracker:
    """Strict order: a future subgoal must not complete before its turn.

    Simultaneous first completion of two subgoals is a violation. Completed
    subgoals may remain true; they need not persist until the final subgoal.
    The event is first predicate satisfaction, not inferred object motion.
    """
    def __init__(self, spec):
        validate_goal(spec)
        self.spec = spec
        self.phase = 0
        self.violated = False
        self.events = []
        self.step = -1

    def update(self, predicate):
        self.step += 1
        values = [bool(predicate(p)) for p in self.spec['predicates']]
        self.last_values = values
        if self.spec['kind'] == 'all':
            return all(values)
        if self.violated:
            return False
        if self.phase == len(values):
            return True
        if any(values[self.phase + 1:]):
            self.violated = True
            self.events.append({'step': self.step, 'violation': 'future_or_simultaneous_subgoal', 'values': values})
            return False
        if values[self.phase]:
            self.events.append({'step': self.step, 'subgoal': self.phase})
            self.phase += 1
        return self.phase == len(values)


class SemanticGoalEvaluator:
    def __init__(self, old_goal, new_goal, other_goals=None):
        self.old = GoalTracker(old_goal)
        self.new = GoalTracker(new_goal)
        self.others = {name: GoalTracker(spec) for name, spec in (other_goals or {}).items()}

    def __call__(self, env):
        predicate = env.env._eval_predicate
        old = self.old.update(predicate)
        new = self.new.update(predicate)
        others = [name for name, tracker in self.others.items() if tracker.update(predicate)]
        hit = classify_hits(old, new, others)
        hit['initial_subgoal_satisfied'] = any(
            tracker.spec['kind'] == 'ordered' and any(tracker.last_values)
            for tracker in (self.old, self.new))
        hit['order_events'] = {'old': list(self.old.events), 'new': list(self.new.events)}
        return hit
