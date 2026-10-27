"""Unit test untuk skrip logging (scripts/log_interaction.py) — Stage 0: S00-03."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / 'scripts' / 'log_interaction.py'
EXPECTED_FIELDS = {
    'timestamp', 'member_id', 'session_id', 'model_name', 'task_id', 'phase',
    'prompt', 'response', 'status', 'files_changed', 'git_commit',
}

# Folder scripts/ bukan package, jadi modul dimuat langsung dari path-nya.
_spec = importlib.util.spec_from_file_location('log_interaction', SCRIPT_PATH)
log_interaction_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(log_interaction_module)


def _read_entries(path: Path) -> list:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


class TestLogInteraction:
    """Pengujian fungsi log_interaction."""

    def test_entry_fields_and_timestamp(self, tmp_path):
        """Memverifikasi 11 field lengkap dan timestamp berzona WIB (+07:00)."""
        log_path = tmp_path / 'interactions.jsonl'
        log_interaction_module.log_interaction(
            member_id='A4', task_id='S00-03', log_path=log_path
        )

        entries = _read_entries(log_path)
        assert len(entries) == 1
        assert set(entries[0]) == EXPECTED_FIELDS
        assert entries[0]['timestamp'].endswith('+07:00')

    def test_appends_without_overwrite(self, tmp_path):
        """Memverifikasi entri baru ditambahkan, bukan menimpa entri lama."""
        log_path = tmp_path / 'interactions.jsonl'
        log_interaction_module.log_interaction(member_id='A1', log_path=log_path)
        log_interaction_module.log_interaction(member_id='A2', log_path=log_path)

        entries = _read_entries(log_path)
        assert [e['member_id'] for e in entries] == ['A1', 'A2']

    def test_non_ascii_text_preserved(self, tmp_path):
        """Memverifikasi teks multibaris dan non-ASCII tersimpan utuh."""
        log_path = tmp_path / 'interactions.jsonl'
        prompt = 'Baris 1\nBaris 2 — "kutipan" ✅'
        log_interaction_module.log_interaction(prompt=prompt, log_path=log_path)

        assert _read_entries(log_path)[0]['prompt'] == prompt

    def test_files_changed_default_not_shared(self, tmp_path):
        """Memverifikasi default files_changed tidak terbagi antar panggilan."""
        log_path = tmp_path / 'interactions.jsonl'
        first = log_interaction_module.log_interaction(log_path=log_path)
        first['files_changed'].append('x.py')
        second = log_interaction_module.log_interaction(log_path=log_path)

        assert second['files_changed'] == []

    def test_default_log_path_is_absolute(self):
        """Memverifikasi LOG_PATH tidak bergantung pada folder kerja terminal."""
        assert log_interaction_module.LOG_PATH.is_absolute()
        assert log_interaction_module.LOG_PATH.parts[-2:] == ('logs', 'interactions.jsonl')


class TestCommandLine:
    """Pengujian antarmuka terminal (main)."""

    def test_cli_reads_prompt_and_response_files(self, tmp_path):
        """Memverifikasi argumen terminal dan pembacaan prompt/respons dari file."""
        log_path = tmp_path / 'interactions.jsonl'
        prompt_file = tmp_path / 'prompt.txt'
        prompt_file.write_text('isi prompt', encoding='utf-8')
        response_file = tmp_path / 'respons.txt'
        response_file.write_text('isi respons', encoding='utf-8')

        log_interaction_module.main([
            '--member-id', 'A3', '--task-id', 'S04-02', '--phase', 'planning',
            '--prompt-file', str(prompt_file), '--response-file', str(response_file),
            '--files-changed', 'src/losses.py', 'tests/test_losses.py',
            '--log-path', str(log_path),
        ])

        entry = _read_entries(log_path)[0]
        assert entry['member_id'] == 'A3'
        assert entry['prompt'] == 'isi prompt'
        assert entry['response'] == 'isi respons'
        assert entry['files_changed'] == ['src/losses.py', 'tests/test_losses.py']

    def test_cli_rejects_unknown_member(self, tmp_path):
        """Memverifikasi ID anggota di luar A1-A4 ditolak."""
        with pytest.raises(SystemExit):
            log_interaction_module.main([
                '--member-id', 'A9', '--task-id', 'S00-03', '--phase', 'testing',
                '--log-path', str(tmp_path / 'interactions.jsonl'),
            ])

    def test_run_without_arguments_writes_nothing(self):
        """Memverifikasi skrip tanpa argumen gagal dan tidak menulis entri kosong."""
        real_log = log_interaction_module.LOG_PATH
        before = real_log.read_bytes() if real_log.exists() else None
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)], capture_output=True, text=True
        )

        after = real_log.read_bytes() if real_log.exists() else None
        assert result.returncode != 0
        assert before == after
