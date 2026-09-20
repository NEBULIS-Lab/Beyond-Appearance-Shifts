"""Count-based Wilson intervals; each row is an aggregate binary outcome."""
from __future__ import annotations
import math
from statistics import NormalDist
from numbers import Integral


def wilson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (successes, total)):
        raise ValueError("successes and total must be integer counts")
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    z = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def count_rows(rows: list[dict], confidence: float = 0.95) -> list[dict]:
    output = []
    for row in rows:
        low, high = wilson(row["successes"], row["total"], confidence)
        output.append({**row, "rate": row["successes"] / row["total"],
                       "ci_low": low, "ci_high": high, "confidence": confidence})
    return output


def count_table(rows: list[dict], *, latex: bool = False) -> str:
    def escape(value):
        text = str(value)
        if latex:
            return ''.join({'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}.get(c, c) for c in text)
        return text.replace('|', r'\|').replace('\n', ' ')
    lines = [r'\begin{tabular}{lrrr}', r'Label & Count & Rate (\%) & Wilson CI (\%) \\'] if latex else ['| Label | Count | Rate (%) | Wilson CI (%) |', '| --- | ---: | ---: | ---: |']
    for r in rows:
        cells = [escape(r.get('label', '')), f"{r['successes']}/{r['total']}", f"{100*r['rate']:.1f}", f"[{100*r['ci_low']:.1f}, {100*r['ci_high']:.1f}]"]
        lines.append((' & '.join(cells) + r' \\') if latex else '| ' + ' | '.join(cells) + ' |')
    if latex:
        lines.append(r'\end{tabular}')
    return '\n'.join(lines) + '\n'
