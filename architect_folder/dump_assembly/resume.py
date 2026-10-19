"""
Продолжение прерванных шагов. Если процесс убит посреди записи строки (OOM-kill
или остановка пода), последняя строка JSONL оборвана: чтение по ней падает, а
дозапись приклеивает новую запись к мусору. completed_qids обрезает файл по
концу последней целой строки и возвращает qid готовых записей — вызывать перед
дозаписью.

Дампы большие (~10 ГБ на 6000 записей, сетевой диск): JSON не разбирается
целиком — qid берётся регуляркой из начала строки (у всех наших выходов это
первый ключ), полностью разбирается только последняя строка, файл не
переписывается, а обрезается на месте и только если хвост битый (иначе не
меняется и mtime — шаги, сравнивающие время файлов, не перезапускаются).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

_QID_RE = re.compile(rb'\{"qid": "([^"]+)"')


def completed_qids(out_path: Path) -> set[str]:
    out_path = Path(out_path)
    if not out_path.exists():
        return set()
    done: set[str] = set()
    good_end = 0          # байтовая позиция конца последней целой строки
    last_line, last_start = b"", 0
    with open(out_path, "rb") as f:
        pos = 0
        for line in f:
            start, pos = pos, pos + len(line)
            if not line.strip():
                good_end = pos
                continue
            if not line.endswith(b"\n"):
                break  # оборванная строка — всё начиная с неё отбрасывается
            m = _QID_RE.match(line)
            if m is not None:
                done.add(m.group(1).decode())
            else:  # qid не первым ключом — разбираем строку целиком
                try:
                    done.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError, TypeError):
                    break
            good_end = pos
            last_line, last_start = line, start
    # последняя «целая» строка могла оборваться ровно перед \n — её проверяем разбором
    if last_line:
        try:
            json.loads(last_line)
        except json.JSONDecodeError:
            m = _QID_RE.match(last_line)
            if m:
                done.discard(m.group(1).decode())
            good_end = last_start
    if good_end < os.path.getsize(out_path):
        with open(out_path, "r+b") as f:
            f.truncate(good_end)
    return done
