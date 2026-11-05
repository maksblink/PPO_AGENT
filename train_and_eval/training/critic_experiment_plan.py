"""Explicit sidecar opt-in. Ordinary configs without a sidecar are unchanged."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def sidecar(config_path):
    return Path(str(config_path) + '.critic.json')


def validate_steps(values):
    if not isinstance(values, list) or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError('Steps must be positive integers')
    if values != sorted(set(values)):
        raise ValueError('Steps must be sorted and unique')
    return values


def load_plan(config_path, *, root, config_sha256, source, segment_sizes, rollout_steps):
    path = sidecar(config_path)
    if not path.exists():
        if path.is_symlink(): raise ValueError('Dangling diagnostic sidecar symlink')
        return None
    if path.is_symlink(): raise ValueError('Diagnostic sidecar cannot be a symlink')
    plan = json.loads(path.read_text())
    if plan.get('format') != 1 or plan.get('config_sha256') != config_sha256:
        raise ValueError('Diagnostic plan/config checksum mismatch')
    for key in ('capture_steps', 'detail_steps'): validate_steps(plan[key])
    if not plan['capture_steps']: raise ValueError('No captures requested')
    actual_id = source.checkpoint_id if source is not None else None
    if plan.get('source_checkpoint_id') != actual_id:
        raise ValueError('Diagnostic source checkpoint changed')
    if source is not None:
        candidates = { (Path(root)/source.relative_path).resolve(),
                       (Path(root)/'artifacts'/source.relative_path).resolve() }
        found = [p for p in candidates if p.is_file()]
        if len(found) != 1 or sha(found[0]) != plan.get('source_checkpoint_sha256'):
            raise ValueError('Diagnostic source checkpoint checksum mismatch')
    boundaries = set(); start = 0
    for size in segment_sizes:
        boundaries.update(range(start + rollout_steps, start + size, rollout_steps))
        start += size; boundaries.add(start)
    if not set(plan['capture_steps'] + plan['detail_steps']) <= boundaries:
        raise ValueError('Diagnostic steps are outside actual rollout boundaries')
    return plan


def save_plan(directory, plan, *, run_id, config):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory/'critic_experiment.json').open('x') as f:
        json.dump(dict(plan, run_id=run_id, normalized_config=json.loads(config)), f, indent=2)
