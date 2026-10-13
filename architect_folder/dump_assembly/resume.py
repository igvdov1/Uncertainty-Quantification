"""
Продолжение прерванных шагов. Если процесс убит посреди записи строки (OOM-kill
на поде — C4, шаг 7), последняя строка JSONL оборвана: чтение по ней падает, а
дозапись приклеивает новую запись к мусору. completed_qids переписывает файл
только валидными строками и возвращает их qid — вызывать перед дозаписью.
"""
from __future__ import annotations

import json
from pathlib import Path


def completed_qids(out_path: Path) -> set[str]:
    out_path = Path(out_path)
    if not out_path.exists():
        return set()
    valid, done = [], set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qid = json.loads(line)["qid"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            valid.append(line)
            done.add(qid)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for line in valid:
            f.write(line + "\n")
    tmp.replace(out_path)
    return done
