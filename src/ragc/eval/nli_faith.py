"""Natural-language-inference wrapper for faithfulness checks.

Faithfulness metrics must come from the configured NLI model. If the model or required
packages are unavailable, evaluation fails loudly instead of reporting lexical proxy
numbers as if they were real model scores.
"""

from __future__ import annotations


class NLIFaith:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall", *, max_length: int = 480, verbose: bool = False):
        self.model_name = model_name
        self.max_length = max_length
        self.verbose = verbose
        self._tokenizer = None
        self._model = None

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("transformers is required for NLI faithfulness evaluation. Install requirements.txt.") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()

    def entail_prob(self, premise: str, hypothesis: str) -> float:
        import torch

        with torch.no_grad():
            enc = self._tokenizer(premise, hypothesis, truncation=True, max_length=self.max_length, return_tensors="pt")
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            labels = getattr(self._model.config, "id2label", {})
            if labels:
                for idx, label in labels.items():
                    if str(label).lower() == "entailment":
                        return float(probs[int(idx)].item())
            raise RuntimeError(f"NLI model {self.model_name!r} does not expose an entailment label")
