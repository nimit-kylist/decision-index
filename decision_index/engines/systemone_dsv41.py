"""Engine for DeepSeek-V4.1-Flash served by SGLang.

Why this exists
---------------
SGLang's ``/v1/systemone`` and ``/v1/decisions`` routes refuse DeepSeek-V4.1
because its architecture resolves to the ``dsv41`` Python chat encoder
(``chat_encoding_spec is not None`` -> ``_validate_server`` returns a 400). The
model repo ships a Jinja ``chat_template.jinja``, but the arch check wins, so the
route stays closed.

``/v1/score`` has no such guard: it scores the *token ids you send* and reads the
next-token log-probabilities of the label ids (``serving_score.py`` ->
``score_request`` -> ``input_ids = query + item``). This engine therefore does what
``/v1/systemone`` would have done server-side - render the state+question with the
model's own Jinja template and locate one-token answer labels - then ships the ids
to ``/v1/score`` with ``apply_softmax=True``.

Prompt wording mirrors SGLang's ``serving_decisions._render_question``
(PROMPT_FORMAT_VERSION 1) so scores are comparable to the reference routes.

Usage
-----
    python -m decision_index run \
      --engine decision_index.engines.systemone_dsv41:Dsv41ScoreEngine \
      --option base_url=http://127.0.0.1:23456 \
      --option model=deepseek-ai/DeepSeek-V4.1-Flash \
      --rows "$RUNS/sample-100.jsonl.gz" --out "$RUNS/sample-100-score"

``--option tokenizer_path=/luna/.../DeepSeek-V4.1-Flash`` reuses a local snapshot
instead of downloading the tokenizer. A bearer token is read from
``DECISION_INDEX_API_KEY`` if set.

Notes
-----
* The template defaults to *thinking*. The engine always passes
  ``enable_thinking=False``, which closes the reasoning block (``</think>``) and
  puts the answer position right after the generation prompt. Scoring inside an
  open think block would be wrong.
* Labels are ``A``..``Z`` for up to 26 options, then ``AA``, ``AB``, ... Each must
  be exactly one token at the answer position; otherwise the row is refused as
  ``Unsupported`` (SGLang does the same). Rows with >26 options depend on the
  DeepSeek tokenizer's added tokens; check the run log before trusting them.
"""

import json
import math
import string

from decision_index.engines import Engine, Unsupported

# Two-letter labels beyond A-Z, same order SGLang uses (_PAIR_LABELS).
_PAIR_LABELS = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]


def _render_text(value):
    """Match serving_decisions.render_text: str as-is, else compact JSON."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_blank(value):
    return not (value.strip() if isinstance(value, str) else value)


def _render_question(text, kind, question, names, details, labels):
    """Prompt wording of PROMPT_FORMAT_VERSION 1 (serving_decisions._render_question)."""
    question_text = "" if _is_blank(question) else _render_text(question)
    if kind == "choice":
        lines = [f"Question: {question_text}"] if question_text else []
        for label, name, description in zip(labels, names, details):
            detail = _render_text(description)
            lines.append(f"{label}: {name} - {detail}" if detail else f"{label}: {name}")
        lines.append("Answer with the letter of one option only.")
    else:  # yes_no
        lines = [
            f"Is the following true? {question_text}"
            if question_text
            else "Is the following true?"
        ]
        for label, description in zip(labels, details):
            detail = _render_text(description)
            if detail:
                lines.append(f"{label}: {detail}")
        lines.append("Answer with yes or no only.")
    return "\n".join([text, "", *lines])


class Dsv41ScoreEngine(Engine):
    name = "http_score_dsv41"
    latency = (
        "HTTP /v1/score wall time against the configured server, including the "
        "server-side prefill of the client-rendered prompt; excludes tokenizer "
        "load and server startup."
    )

    def __init__(
        self,
        base_url=None,
        model="deepseek-ai/DeepSeek-V4.1-Flash",
        tokenizer_path=None,
        chat_template_file=None,
        trust_remote_code=False,
        timeout=600,
        max_options=255,
        extra=None,
        **options,
    ):
        super().__init__(**options)
        import os

        import httpx
        from transformers import AutoTokenizer

        base_url = base_url or os.environ.get("DECISION_INDEX_BASE_URL")
        if not base_url:
            raise ValueError("Dsv41ScoreEngine needs base_url (or DECISION_INDEX_BASE_URL)")

        source = tokenizer_path or model
        self.tok = AutoTokenizer.from_pretrained(
            source, trust_remote_code=trust_remote_code
        )
        # Prefer the tokenizer's own template; fall back to a local chat_template.jinja.
        if not getattr(self.tok, "chat_template", None):
            from pathlib import Path

            candidates = []
            if chat_template_file:
                candidates.append(Path(chat_template_file))
            if tokenizer_path:
                candidates.append(Path(tokenizer_path) / "chat_template.jinja")
            for path in candidates:
                if path.exists():
                    self.tok.chat_template = path.read_text(encoding="utf-8")
                    break
        if not getattr(self.tok, "chat_template", None):
            raise ValueError(
                "tokenizer has no chat_template; pass tokenizer_path to the model "
                "directory or chat_template_file=.../chat_template.jinja"
            )

        headers = {}
        token = os.environ.get("DECISION_INDEX_API_KEY")
        if token:
            headers["Authorization"] = "Bearer " + token

        self.model = model
        self.max_options = max_options
        self.extra = extra or {}
        self.client = httpx.Client(base_url=base_url, timeout=timeout, headers=headers)
        self.provenance = {
            "kind": "http_score",
            "base_url": base_url,
            "model": model,
            "tokenizer_source": str(source),
            "request_options": self.extra,
            "policy": (
                "Client renders the model's Jinja chat template with thinking off, "
                "maps options/noul to single-token labels, and scores the exact ids "
                "via /v1/score (apply_softmax). Bypasses the /v1/systemone dsv41 "
                "guard; prompt wording matches PROMPT_FORMAT_VERSION 1."
            ),
        }

    # -- rendering ---------------------------------------------------------

    def _answer_position_ids(self, prompt):
        pids = self.tok.encode(prompt, add_special_tokens=False)
        if not pids:
            raise Unsupported("chat template produced an empty prompt")
        return pids

    def _one_token_at(self, prompt, pids, label):
        """The id a label adds after the prompt, or None if it is not one token."""
        one = self.tok.encode(prompt + label, add_special_tokens=False)
        if one[: len(pids)] != pids or len(one) != len(pids) + 1:
            return None
        return one[-1]

    def _choice_labels(self, prompt, pids, count):
        """Single-token, distinct labels, A-Z up to 26, then two-letter labels.

        Mirrors SGLang's systemone/serving._pair_labels: two-letter labels are
        accepted in fixed order only while they add exactly one distinct token at
        the answer position. The prompt is rebuilt by the caller after this, since
        the labels appear in the option lines.
        """
        if count <= 26:
            return list(string.ascii_uppercase[:count])
        labels, seen = [], set()
        for label in _PAIR_LABELS:
            token_id = self._one_token_at(prompt, pids, label)
            if token_id is None or token_id in seen:
                continue
            labels.append(label)
            seen.add(token_id)
            if len(labels) == count:
                break
        if len(labels) < count:
            raise Unsupported(
                f"only {len(labels)} of {count} answer labels are one distinct "
                "token at the answer position"
            )
        return labels

    def _encode_item(self, text, q):
        kind = q["type"]
        if kind == "choice":
            criteria = q["criteria"]
            names = list(criteria)
            details = list(criteria.values())
            if not 1 <= len(names) <= self.max_options:
                raise Unsupported(f"choice has {len(names)} options")
        elif kind == "noul":
            criteria = q.get("criteria") or {}
            names = ["yes", "no"]
            details = [criteria.get("true"), criteria.get("false")]
            if _is_blank(q.get("instructions")) and all(_is_blank(d) for d in details):
                raise Unsupported("noul question has nothing to decide on")
        else:
            raise Unsupported("Unsupported question type " + str(kind))

        def render(labels):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": _render_question(
                    text, kind, q.get("instructions"), names, details, labels)}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        if kind == "noul":
            labels = ["yes", "no"]
        elif len(names) <= 26:
            labels = list(string.ascii_uppercase[: len(names)])
        else:
            # Two passes: the labels appear in the option lines, so discover the
            # two-letter labels on a first render, then re-render with the final set.
            probe = render(list(string.ascii_uppercase))
            labels = self._choice_labels(probe, self._answer_position_ids(probe), len(names))

        prompt = render(labels)
        pids = self._answer_position_ids(prompt)
        label_ids = []
        for label in labels:
            token_id = self._one_token_at(prompt, pids, label)
            if token_id is None:
                raise Unsupported(
                    f"answer label {label!r} is not one token at the answer position"
                )
            label_ids.append(token_id)
        if len(set(label_ids)) != len(label_ids):
            raise Unsupported("answer labels are not distinct tokens")
        return pids, label_ids, labels, names

    # -- engine API --------------------------------------------------------

    def __call__(self, state, questions):
        text = _render_text(state)
        items, label_ids_per_item, metas = [], [], []
        for key, q in questions.items():
            ids, label_ids, labels, names = self._encode_item(text, q)
            items.append(ids)
            label_ids_per_item.append(label_ids)
            metas.append((key, q["type"], labels, names))

        body = {
            "query": [],
            "items": items,
            "label_token_ids": label_ids_per_item,
            "apply_softmax": True,
            "return_token_logprobs": True,
            **self.extra,
        }
        r = self.client.post("/v1/score", json=body)
        if r.status_code in (400, 413, 422):
            raise RuntimeError(
                f"/v1/score returned {r.status_code}: {r.text[:2000]}"
            ) from None
        r.raise_for_status()
        scores = r.json()["scores"]
        if len(scores) != len(items):
            raise RuntimeError(
                f"/v1/score returned {len(scores)} rows for {len(items)} questions"
            )

        answers = {}
        for (key, kind, labels, names), probs in zip(metas, scores):
            if len(probs) != len(labels):
                raise RuntimeError(
                    f"/v1/score returned {len(probs)} values for {len(labels)} labels"
                )
            if any(not math.isfinite(p) for p in probs):
                raise RuntimeError(f"non-finite score for question {key!r}")
            if kind == "choice":
                best = max(range(len(probs)), key=lambda i: probs[i])
                answers[key] = {
                    "type": "choice",
                    "choice": names[best],
                    "probabilities": {name: p for name, p in zip(names, probs)},
                }
            else:  # noul: labels ["yes", "no"], probabilities["yes"] is the answer
                answers[key] = {
                    "type": "noul",
                    "noul": probs[labels.index("yes")],
                }
        return {"model": self.model, "answers": answers}, None

    def close(self):
        self.client.close()
