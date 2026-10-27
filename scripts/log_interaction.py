#!/usr/bin/env python3
"""Pencatat interaksi AI ke ``logs/interactions.jsonl`` (S00-03).

Bisa dipakai sebagai fungsi Python maupun lewat terminal. Contoh:

    python scripts/log_interaction.py --member-id A4 --task-id S00-03 \\
        --phase handoff --model-name claude-opus-5-5 \\
        --prompt-file prompt.txt --response-file respons.txt \\
        --status disetujui --files-changed logs/A4/S00-03-LOG.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

# Path absolut agar skrip benar walau dijalankan dari folder mana pun.
REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / 'logs' / 'interactions.jsonl'
MEMBER_IDS = ('A1', 'A2', 'A3', 'A4')


def log_interaction(
    member_id: str = '',
    session_id: str = '',
    model_name: str = '',
    task_id: str = '',
    phase: str = '',
    prompt: str = '',
    response: str = '',
    status: str = '',
    files_changed: Optional[list[str]] = None,
    git_commit: str = '',
    log_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Menambahkan satu entri interaksi AI ke file JSONL.

    Args:
        member_id: ID anggota (A1-A4).
        session_id: ID atau tautan sesi chat AI.
        model_name: Nama model AI yang dipakai.
        task_id: Step ID, misalnya ``S04-02``.
        phase: planning/implementation/debugging/testing/handoff.
        prompt: Teks prompt lengkap ke AI.
        response: Ringkasan atau teks respons AI.
        status: Status persetujuan langkah.
        files_changed: Daftar file yang berubah.
        git_commit: Hash commit terkait.
        log_path: Lokasi file JSONL; default ``LOG_PATH``.

    Returns:
        Entri yang ditulis, dalam bentuk dict.
    """
    if files_changed is None:
        files_changed = []
    if log_path is None:
        log_path = LOG_PATH
    entry = {
        'timestamp': datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),
        'member_id': member_id,
        'session_id': session_id,
        'model_name': model_name,
        'task_id': task_id,
        'phase': phase,
        'prompt': prompt,
        'response': response,
        'status': status,
        'files_changed': files_changed,
        'git_commit': git_commit,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def _read_text(text: str, file_path: Optional[str]) -> str:
    """Mengambil teks dari file jika diberikan, jika tidak memakai ``text``."""
    if file_path:
        return Path(file_path).read_text(encoding='utf-8')
    return text


def main(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    """Membaca argumen terminal lalu mencatat satu entri interaksi."""
    parser = argparse.ArgumentParser(
        description='Catat satu interaksi AI ke logs/interactions.jsonl.'
    )
    parser.add_argument('--member-id', required=True, choices=MEMBER_IDS)
    parser.add_argument('--task-id', required=True, help='Step ID, mis. S04-02')
    parser.add_argument('--phase', required=True)
    parser.add_argument('--model-name', default='')
    parser.add_argument('--session-id', default='')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--prompt-file', help='Ambil prompt dari file teks')
    parser.add_argument('--response', default='')
    parser.add_argument('--response-file', help='Ambil respons dari file teks')
    parser.add_argument('--status', default='')
    parser.add_argument('--files-changed', nargs='*', default=[])
    parser.add_argument('--git-commit', default='')
    parser.add_argument('--log-path', type=Path, default=None)
    args = parser.parse_args(argv)

    entry = log_interaction(
        member_id=args.member_id,
        session_id=args.session_id,
        model_name=args.model_name,
        task_id=args.task_id,
        phase=args.phase,
        prompt=_read_text(args.prompt, args.prompt_file),
        response=_read_text(args.response, args.response_file),
        status=args.status,
        files_changed=args.files_changed,
        git_commit=args.git_commit,
        log_path=args.log_path,
    )
    print(f"Tercatat: {entry['member_id']} {entry['task_id']} "
          f"({entry['phase']}) pada {entry['timestamp']}")
    return entry


if __name__ == '__main__':
    main()
