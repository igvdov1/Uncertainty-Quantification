"""
Доступ к записи дампа (см. dump_schema_v1.md) без завязки на конкретный
класс — запись остаётся обычным dict, как в JSONL. Здесь только точки
доступа с понятными именами и явной проверкой обязательных полей, чтобы
отсутствие поля падало с внятной ошибкой на синтетике, а не на настоящем
дампе (это и есть смысл гейта A3: скринер спотыкается о недостающие поля
до того, как потрачен GPU).
"""
from __future__ import annotations

from typing import Any

Record = dict[str, Any]

REQUIRED_TOP_LEVEL = ("qid", "question", "split", "passages", "rag", "closed_book")


def validate_record(record: Record) -> None:
    missing = [f for f in REQUIRED_TOP_LEVEL if f not in record]
    if missing:
        raise KeyError(f"Запись {record.get('qid', '?')}: отсутствуют поля {missing}")


def branch(record: Record, mode: str) -> Record:
    """mode: 'rag' или 'closed_book'."""
    if mode not in ("rag", "closed_book"):
        raise ValueError(f"unknown mode: {mode}")
    return record[mode]


def token_logprobs(record: Record, mode: str) -> list[float]:
    return branch(record, mode)["token_logprobs"]


def token_entropy(record: Record, mode: str) -> list[float] | None:
    return branch(record, mode).get("token_entropy")


def samples(record: Record, mode: str) -> list[str]:
    return branch(record, mode).get("samples", [])


def passages(record: Record) -> list[dict]:
    return record.get("passages", [])


def passages_top20_scores(record: Record) -> list[float]:
    return record.get("passages_top20_scores", [])


def signals(record: Record) -> dict:
    return record.get("signals", {})


def claims(record: Record) -> list[dict]:
    return record.get("claims", [])


def label(record: Record, target: str) -> int:
    """target: 'faithful' или 'factual', метка на уровне ответа целиком."""
    key = f"label_{target}"
    if key not in record:
        raise KeyError(f"Запись {record.get('qid', '?')}: нет метки {key}")
    return int(record[key])


def claim_label(claim: dict, target: str) -> int:
    key = f"label_{target}"
    if key not in claim:
        raise KeyError(f"Клейм {claim.get('cid', '?')}: нет метки {key}")
    return int(claim[key])
