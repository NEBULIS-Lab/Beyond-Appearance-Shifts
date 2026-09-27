"""BAS-VLA public package.

The current public surface follows the paper-facing decomposition:

- a breaking-centered action-calibration core
- an evidence-gated preserving auxiliary
"""

from .pairs import SemanticBreakPair, load_semantic_break_pairs

__all__ = ["SemanticBreakPair", "load_semantic_break_pairs"]
