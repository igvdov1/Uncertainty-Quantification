"""Тесты правок стенда по runner_folder/C9_bench_issues.md (без GPU и без сети)."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from dump_assembly import ccp, generation, labeling, paraphrases, questions, streaming_retrieval
from dump_assembly.questions import Question


# ---- §0, §11: отбор вопросов --------------------------------------------------

def _fake_loader(source, n_rows, dup_every=0):
    def load(n=None):
        out = []
        for i in range(n_rows):
            text = f"{source} question {i // 2 if dup_every and i % dup_every == 0 else i}?"
            meta = {"row": i}
            if source == "popqa":
                meta = {"row": i, "prop": f"p{i % 4 if i < 400 else 0}", "s_pop": int(i * 7 % 1000)}
            out.append(Question(qid=f"{source}_{i}", question=text, source=source, gold_answers=["x"], meta=meta))
        return out
    return load


def test_question_set_sizes_splits_and_popqa_strata():
    loaders = {s: _fake_loader(s, 500) for s in questions.SHORTFORM_SOURCES}
    loaders["webq"] = _fake_loader("webq", 120)  # сплит меньше per_source — берётся весь
    qs = questions.build_question_set(per_source=200, dev_frac=0.4, seed=0, loaders=loaders)
    by = Counter((q.source, q.split) for q in qs)
    assert by[("nq", "dev")] == 80 and by[("nq", "test")] == 120
    assert by[("webq", "dev")] + by[("webq", "test")] == 120
    assert len({q.qid for q in qs}) == len(qs)
    pop = [q for q in qs if q.source == "popqa"]
    props = Counter(q.meta["prop"] for q in pop)
    assert len(props) == 4 and max(props.values()) - min(props.values()) <= 1  # поровну по отношениям
    assert all("pop_bin" in q.meta for q in pop)
    # детерминизм и смешанный порядок источников
    again = questions.build_question_set(per_source=200, dev_frac=0.4, seed=0, loaders=loaders)
    assert [q.qid for q in again] == [q.qid for q in qs]
    assert len({q.source for q in qs[:20]}) > 1


def test_allocate_evenly_redistributes_small_groups():
    assert questions.allocate_evenly({"a": 2, "b": 100, "c": 100}, 30) == {"a": 2, "b": 14, "c": 14}


def test_dedupe_and_question_file_roundtrip(tmp_path):
    qs = [Question("a_0", "Who?", "nq", ["x"]), Question("a_1", " who? ", "nq", ["y"]),
          Question("a_2", "Where?", "nq", [])]
    assert [q.qid for q in questions.dedupe_questions(qs)] == ["a_0"]
    qs[0].meta = {"row": 0}
    path = tmp_path / "q.jsonl"
    questions.write_question_file(qs[:1], path)
    back = questions.read_question_file(path)
    assert back[0].qid == "a_0" and back[0].gold_answers == ["x"] and back[0].meta == {"row": 0}


# ---- §1: перефразы ------------------------------------------------------------

@pytest.mark.parametrize("text,why", [
    ("Which city is the Pacific end of the Trans-Siberian Railway?", "ok"),
    ("What city is the end?\nThe final answer is: Vladivostok", "multiline"),
    ("Where does it end? Rewrite the following question", "question_mark"),
    ("Which answer is the end of the railway?", "service_words"),
    ("Is it A) Vladivostok B) Moscow?", "service_words"),
    ("Is the Pacific terminus of the Trans Siberian Railway Vladivostok?", "gold_leak"),
    ("What is the Pacific terminus of the Trans Siberian Railway?", "same_as_question"),
    ("Where?", "length"),
])
def test_clean_paraphrase(text, why):
    q = "What is the Pacific terminus of the Trans Siberian Railway?"
    assert paraphrases.clean_paraphrase(text, q, ["Vladivostok"])[1] == why


def test_select_paraphrases_keeps_distinct_clean():
    q = "who wrote war and peace"
    cands = ["Who is the author of War and Peace?", "Who is the author of War and Peace?",
             "Tolstoy wrote it?", "Which writer created War and Peace?", "Who penned War and Peace?",
             "Which novelist authored War and Peace?"]
    reasons = Counter()
    kept = paraphrases.select_paraphrases(cands, q, ["Tolstoy"], reasons=reasons)
    assert len(kept) == 3 and len(set(kept)) == 3 and reasons["duplicate"] == 1 and reasons["gold_leak"] == 1


# ---- §10: ретрив ----------------------------------------------------------------

def test_running_topk_matches_bruteforce():
    import torch
    rng = np.random.default_rng(0)
    q = rng.normal(size=(30, 8)).astype(np.float32)
    shards = [rng.normal(size=(n, 8)).astype(np.float32) for n in (200, 7, 333)]
    tk, off = streaming_retrieval.RunningTopK(30, 5), 0
    for s in shards:
        streaming_retrieval.search_shard(torch.from_numpy(q), s, off, tk, device="cpu", chunk=50, q_block=4)
        off += len(s)
    full = q @ np.concatenate(shards).T
    assert (np.sort(np.argsort(-full, 1)[:, :5], 1) == np.sort(tk.index, 1)).all()
    assert (np.diff(tk.scores, axis=1) <= 0).all()


def test_has_answer_dpr_semantics():
    assert streaming_retrieval.has_answer(["Mount Kilimanjaro"], "west of Mount Kilimanjaro, Tanzania")
    assert not streaming_retrieval.has_answer(["politician"], "many politicians")
    assert streaming_retrieval.has_answer(["U.S."], "the u.s. army")


# ---- §6: метка factual ------------------------------------------------------------

@pytest.mark.parametrize("pred,gold,want", [
    ("16th century", ["the 12th century"], 0),
    ("Kazan Cathedral", ["Cathedral of St Basil"], 0),
    ("virus", ["US"], 0),
    ("Turkish", ["Turkish Language"], 1),
    ("Confederates", ["Confederate"], 1),
    ("Indian subcontinent/South Asia", ["South Asia"], 1),
    ("1979-80 season", ["1979–80 season"], 1),
    ("Barry Parker", ["architect Barry Parker"], 1),
    ("January 25, 2004", ["January 2004"], 1),
    ("unknown", ["Paris"], 0),
])
def test_factuality_v3(pred, gold, want):
    assert labeling.factuality_shortform_v3(pred, gold) == want


# ---- §8: группы внимания ------------------------------------------------------------

def test_template_tokens_are_not_context():
    spans = generation.PromptSpans(instruction=(5, 10), passages=[(10, 20), (20, 30)], question=(30, 35))
    g = [generation._group_for_position(p, 40, spans) for p in range(42)]
    assert g[:5] == [4] * 5 and set(g[10:30]) == {1} and set(g[35:40]) == {4} and g[40:] == [3, 3]


def test_context_share_uses_passages():
    from screener import registry
    rec = {"rag": {"attention_by_group": [[0.1, 0.05, 0.05, 0.0, 0.8]], "attention_by_passage": [[0.03, 0.02]]}}
    assert registry._context_share(rec)[0] == pytest.approx(0.05)


# ---- §18: CCP по словам ---------------------------------------------------------------

def test_word_spans_skip_eot_and_punct():
    strs = ["Mount", "ĠKil", "iman", "jaro", ",", "ĠTanz", "ania", "<|eot_id|>"]
    special = [False] * 7 + [True]
    punct = [False, False, False, False, True, False, False, False]
    assert ccp.word_spans(strs, special, punct) == [(0, 1), (1, 4), (5, 7)]


def test_answer_ccp_words_same_word_is_entail():
    dec = lambda ids: " ".join(map(str, ids))
    steps = [([1, 5], [0.5, 0.5])]
    out = ccp.answer_ccp_words("Q", [1], [(0, 1)], steps, [0.5], dec,
                               lambda pre, alts: ["1"], lambda pairs: [ccp.CONTRA] * len(pairs))
    assert out == [1.0]  # альтернатива дописалась в то же слово -> entail


# ---- §17: NLI с вопросом ----------------------------------------------------------------

def test_gpu_cards_question_variants_prepend_question():
    from dump_assembly import gpu_cards as gc
    seen = []
    def contra(pairs):
        seen.extend(pairs)
        return [0.0] * len(pairs)
    rec = {"qid": "q", "source": "triviaqa", "question": "Which volcano?", "passages": [],
           "rag": {"answer": "Kilimanjaro", "samples": ["Kibo"]},
           "closed_book": {"answer": "Kibo", "samples": ["Kilimanjaro"]}}
    sig = gc.card_signals(rec, contra, contra, lambda *a: [])
    assert sig["ncp_q_rag"] == 1.0
    assert ("Which volcano? Kilimanjaro", "Which volcano? Kibo") in seen


def test_embedding_shards_skip_faiss_index_files(tmp_path):
    import pickle
    for i in range(2):
        pickle.dump(([f"{i}a", f"{i}b"], np.zeros((2, 4), dtype=np.float16)), open(tmp_path / f"passages_0{i}", "wb"))
    (tmp_path / "index.faiss").write_bytes(b"not a pickle")
    (tmp_path / "index_meta.faiss").write_bytes(b"not a pickle")
    shards = list(streaming_retrieval.iter_embedding_shards(str(tmp_path)))
    assert [ids for ids, _ in shards] == [["0a", "0b"], ["1a", "1b"]]


def test_completed_qids_drops_truncated_tail(tmp_path):
    from dump_assembly.resume import completed_qids
    p = tmp_path / "out.jsonl"
    p.write_text('{"qid": "a", "x": 1}\n{"qid": "b", "x": 2}\n{"qid": "c", "x": [1, 2')
    assert completed_qids(p) == {"a", "b"}
    with open(p, "a") as f:
        f.write('{"qid": "c", "x": 3}\n')
    assert [json.loads(l)["qid"] for l in p.read_text().splitlines()] == ["a", "b", "c"]
