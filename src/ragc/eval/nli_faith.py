"""Natural-language-inference wrapper for faithfulness checks.

This module scores whether a candidate answer is entailed by a source passage. It is
primarily used for local validity checks and can fall back to a lexical heuristic when
an external NLI model is unavailable.
"""

from __future__ import annotations

import re


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9._\-]*[a-z0-9]|[a-z0-9]", (text or "").lower())


class NLIFaith:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall", *, max_length: int = 480, verbose: bool = False):
        self.model_name = model_name
        self.max_length = max_length
        self.verbose = verbose
        self._tokenizer = None
        self._model = None

        if model_name != "dummy":
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self._model.eval()
            except Exception:
                self.model_name = "dummy"

    def _lexical_entail_prob(self, premise: str, hypothesis: str) -> float:
        p_tokens = set(_tokenize(premise))
        h_tokens = set(_tokenize(hypothesis))
        if not h_tokens:
            return 0.1

        overlap = len(p_tokens & h_tokens)
        lexical = overlap / max(1, len(h_tokens))
        if any(word in hypothesis.lower() for word in ["unrelated", "not", "never", "only", "unless"]):
            lexical *= 0.2
        if not p_tokens.intersection(h_tokens):
            return 0.08
        if overlap >= max(2, len(h_tokens) * 0.5):
            return 0.88
        return max(0.1, min(0.9, 0.35 + lexical))

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        if self.model_name == "dummy":
            return self._lexical_entail_prob(premise, hypothesis)

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self._tokenizer is None or self._model is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self._model.eval()

            with torch.no_grad():
                enc = self._tokenizer(premise, hypothesis, truncation=True, max_length=self.max_length, return_tensors="pt")
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[0]
                labels = getattr(self._model.config, "id2label", {})
                if labels:
                    for idx, label in labels.items():
                        if str(label).lower() == "entailment":
                            return float(probs[int(idx)].item())
                return float(probs[0].item())
        except Exception:
            return self._lexical_entail_prob(premise, hypothesis)
