"""Add two fixed-LR grids using existing project configurations as templates."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import re
import yaml


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def main():
    root = Path(__file__).resolve().parents[1]
    queue_path = root / 'configs/search_queues/grid_w384x3_w384x4_w768x3_gamma_lr_5dataepochs_ne3_s1.yml'
    original = queue_path.read_text()
    queue = yaml.safe_load(original)
    paths = queue['configs']
    require(len(paths) == len(set(paths)), 'Duplicate queue paths.')
    configs = {p: yaml.safe_load((root / p).read_text()) for p in paths}
    architectures = {(384, 384, 384): 0, (384, 384, 384, 384): 1, (768, 768, 768): 2}
    gammas = {Decimal('0.90'), Decimal('0.95'), Decimal('0.99')}

    def identity(c):
        match = re.fullmatch(r'([fr])_v([0-4])_w(.+)_g(090|095|099)_lr(0p\d{9})_s1_ne3', c['run']['name'])
        require(match is not None, f"Unexpected run name: {c['run']['name']}")
        stage = int(match[2])
        arch = tuple(c['ppo']['hidden_sizes'])
        gamma = Decimal(str(c['ppo']['gamma']))
        lr = Decimal(str(c['ppo']['learning_rate']))
        require(arch in architectures and gamma in gammas, 'Unexpected architecture/gamma.')
        require(c['ppo']['n_epochs'] == 3 and c['run']['seed'] == 1, 'Unexpected n_epochs/seed.')
        require(match[1] == ('f' if stage == 0 else 'r'), 'Incorrect run prefix.')
        token = f'{arch[0]}x{len(arch)}'
        expected = f"{'f' if stage == 0 else 'r'}_v{stage}_w{token}_g{int(gamma * 100):03d}_lr{lr:.9f}_s1_ne3".replace('lr0.', 'lr0p')
        require(c['run']['name'] == expected, 'Name and parameters disagree.')
        return arch, gamma, stage, lr

    templates = {p: c for p, c in configs.items() if Decimal(str(c['ppo']['learning_rate'])) == Decimal('0.000015')}
    expected_keys = {(a, g, v) for a in architectures for g in gammas for v in range(5)}
    require(len(templates) == 45, 'Expected 45 templates for LR 0.000015.')
    require({identity(c)[:3] for c in templates.values()} == expected_keys, 'Incomplete template grid.')
    planned = {}
    for lr in (Decimal('0.000012500'), Decimal('0.000017500')):
        new_token = 'lr' + format(lr, '.9f').replace('.', 'p')
        for source, config in templates.items():
            old_token = 'lr0p000015000'
            target = source.replace(old_token, new_token)
            require(target != source, f'Missing LR token: {source}')
            expected = deepcopy(config)
            expected['ppo']['learning_rate'] = float(lr)
            expected['run']['name'] = config['run']['name'].replace(old_token, new_token)
            if identity(config)[2]:
                parent = config['continuation']['source_run']
                require(old_token in parent, f'Unexpected parent: {source}')
                expected['continuation']['source_run'] = parent.replace(old_token, new_token)
            content = (root / source).read_text().replace(old_token, new_token)
            content, count = re.subn(r'(?m)^([ \t]+learning_rate:[ \t]*)[^#\n]+', lambda m: m[1] + format(lr, '.9f'), content)
            require(count == 1 and yaml.safe_load(content) == expected, f'Unexpected changes: {target}')
            if (root / target).exists():
                require(yaml.safe_load((root / target).read_text()) == expected, f'Existing file differs: {target}')
            require(target not in planned, f'Duplicate target: {target}')
            planned[target] = content
            configs[target] = expected

    def order(p):
        arch, gamma, stage, lr = identity(configs[p])
        return architectures[arch], stage, lr, gamma

    ordered = sorted(set(paths) | set(planned), key=order)
    by_name = {}
    for pos, p in enumerate(ordered):
        c = configs[p]
        require(Path(p).stem == c['run']['name'], f'Filename mismatch: {p}')
        require(c['run']['name'] not in by_name, 'Duplicate run name.')
        by_name[c['run']['name']] = (pos, c)
    links = 0
    for pos, p in enumerate(ordered):
        c = configs[p]
        arch, gamma, stage, lr = identity(c)
        continuation = c['continuation']
        if not stage:
            require(continuation['mode'] == 'fresh' and not continuation.get('source_run'), f'Invalid fresh: {p}')
            continue
        require(continuation['mode'] == 'resume' and continuation['checkpoint'] == 'final', f'Invalid resume: {p}')
        parent_name = continuation['source_run']
        require(parent_name in by_name, f'Missing parent: {p}')
        parent_pos, parent = by_name[parent_name]
        require(parent_pos < pos and identity(parent) == (arch, gamma, stage - 1, lr), f'Wrong predecessor: {p}')
        require(parent['ppo'] == c['ppo'] and parent['data'] == c['data'] and parent['environment'] == c['environment'], f'Parent configuration differs: {p}')
        links += 1
    require(len(planned) == 90, 'Expected 90 target configs.')
    require(queue_path.read_text() == original, 'Queue changed during preparation.')
    created = 0
    for target, content in planned.items():
        path = root / target
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('x') as stream:
                stream.write(content)
            created += 1
    queue['configs'] = ordered
    queue_path.write_text(yaml.safe_dump(queue, sort_keys=False, allow_unicode=True))
    print(f'PASS: {created} new files; 90 target configs; {len(ordered)} queue entries; {links} final continuation links.')
    print('No training started. Database and n_epochs=9 unchanged.')


if __name__ == '__main__':
    main()
