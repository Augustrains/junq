import json
from pathlib import Path
import torch

path = Path('/hard_data/user/guzechen/junq/dppo/checkpoints/happo_recon_attack/latest.pt')
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
buffer = checkpoint.get('stage_buffer', {})
rows = buffer.get('rows', {})
active = buffer.get('active_tasks', {})
print('checkpoint=', path)
print('update=', checkpoint.get('update'))
print('episode=', checkpoint.get('episode'))
print('episode_terminal=', checkpoint.get('episode_terminal'))
print('done_reason=', checkpoint.get('done_reason'))
print('saved_rows=', {name: len(items) for name, items in rows.items()})
print('active_rows=', {name: sum(len(task.get('rows', [])) for task in tasks.values()) for name, tasks in active.items()})
print('active_tasks=', {name: len(tasks) for name, tasks in active.items()})
print('counts_before_update=', checkpoint.get('stage_buffer_counts_before_update'))
print('remaining_counts=', checkpoint.get('stage_buffer_remaining_counts'))
print('episode_sim_seconds=', checkpoint.get('episode_sim_seconds'))
metrics_path = path.parent / 'metrics.jsonl'
episodes = []
if metrics_path.exists():
    for line in metrics_path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if 'episode_total_reward' in row:
            episodes.append(row)
print('metric_episodes=', len(episodes))
print('decision_steps_total=', sum(int(row.get('episode_summary', {}).get('steps', 0)) for row in episodes))
print('decision_steps_last_episode=', int(episodes[-1].get('episode_summary', {}).get('steps', 0)) if episodes else 0)
all_rows = _load_rows = []
for line in metrics_path.read_text(encoding='utf-8').splitlines() if metrics_path.exists() else []:
    _load_rows.append(json.loads(line))
updates = [row for row in _load_rows if row.get('record_type') == 'policy_update']
print('policy_update_records=', len(updates))
print('trained_samples_total=', {name: sum(int(row.get('samples', {}).get(name, 0)) for row in updates) for name in ('recon', 'attack')})
print('collected_samples_total_estimate=', {name: sum(int(row.get('samples', {}).get(name, 0)) for row in updates) + len(rows.get(name, [])) for name in ('recon', 'attack')})