"""Версии промпта генерации без модели: фейковый токенизатор (символ = токен)."""
from __future__ import annotations

from dump_assembly import generation


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return f"<u>{messages[0]['content']}</u><a>"

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _prompt(passages, version):
    tok = CharTokenizer()
    ids, spans = generation.build_prompt_ids(tok, "Who?", passages, version)
    return tok.decode(ids), spans


def test_v2_asks_for_bare_answer_and_unknown():
    text, _ = _prompt(["Paris is the capital."], "v2")
    assert "reply exactly: unknown" in text and "only the answer itself" in text
    text_cb, _ = _prompt(None, "v2")
    assert "If you do not know the answer, reply exactly: unknown" in text_cb


def test_v1_reproduces_old_instructions_and_default_is_v2():
    text, _ = _prompt(["P"], "v1")
    assert generation.INSTRUCTION_RAG in text and "unknown" not in text
    assert generation.PROMPT_VERSION == "v2"


def test_spans_still_cover_question_in_both_versions():
    tok = CharTokenizer()
    for v in ("v1", "v2"):
        ids, spans = generation.build_prompt_ids(tok, "Who?", ["P1", "P2"], v)
        a, b = spans.question
        assert "Who?" in tok.decode(ids[a:b]) and len(spans.passages) == 2
