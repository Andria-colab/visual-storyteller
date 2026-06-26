"""
Evaluation (Person 2) — BLEU-1..4 over all 5 references (NLTK).

We use NLTK's corpus BLEU with smoothing so short captions don't collapse to
zero. METEOR/CIDEr (pycocoevalcap) are out of scope per the chosen metrics
decision; if we later add them, they slot in alongside `score_bleu`.

Inputs are TOKEN LISTS, not strings, so cleaning is already applied consistently
with training (see data.clean_caption / vocab.decode).
"""

from __future__ import annotations

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


_SMOOTH = SmoothingFunction().method1


def score_bleu(
    hypotheses: dict[str, list[str]],
    references: dict[str, list[list[str]]],
) -> dict[str, float]:
    """Corpus BLEU-1..4.

    hypotheses : {image_id: hyp_tokens}
    references : {image_id: [ref_tokens, ...]}  (all 5 refs per image)
    Only image ids present in both are scored.
    """
    ids = [i for i in hypotheses if i in references and references[i]]
    list_of_refs = [references[i] for i in ids]
    list_of_hyps = [hypotheses[i] for i in ids]

    weights = {
        "BLEU-1": (1.0, 0, 0, 0),
        "BLEU-2": (0.5, 0.5, 0, 0),
        "BLEU-3": (1 / 3, 1 / 3, 1 / 3, 0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25),
    }
    scores = {
        name: corpus_bleu(list_of_refs, list_of_hyps, weights=w, smoothing_function=_SMOOTH)
        for name, w in weights.items()
    }
    scores["n"] = float(len(ids))
    return scores


def format_score_table(rows: dict[str, dict[str, float]]) -> str:
    """Render {run_name: bleu_dict} as a markdown table for results.md / README.

    Example rows: {"baseline-greedy": {...}, "transformer-beam": {...}}
    """
    cols = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    header = "| Model | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for name, sc in rows.items():
        cells = " | ".join(f"{sc.get(c, 0.0):.3f}" for c in cols)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)
