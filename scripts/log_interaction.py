#!/usr/bin/env python3
# Script logging interaksi AI (fallback manual/ekspor) - Stage 0/1
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

LOG_PATH = Path('logs/interactions.jsonl')

def log_interaction(member_id='', session_id='', model_name='', task_id='', phase='', prompt='', response='', status='', files_changed=None, git_commit=''):
    if files_changed is None:
        files_changed = []
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
        'git_commit': git_commit
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry

if __name__ == '__main__':
    log_interaction(member_id='A1', phase='stage0', status='scaffold')
