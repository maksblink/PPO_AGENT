"""Two L3 groups, two alternating disjoint four-core masks per group."""
import os
from pathlib import Path


def discover_groups(root=Path('/sys/devices/system/cpu'), allowed=None):
    allowed = os.sched_getaffinity(0) if allowed is None else set(allowed)
    sides = {}
    for cpu in sorted(allowed):
        p = root / f'cpu{cpu}'
        package = (p/'topology/physical_package_id').read_text().strip()
        core = (p/'topology/core_id').read_text().strip()
        caches = [q for q in (p/'cache').glob('index*') if (q/'level').read_text().strip() == '3']
        if len(caches) != 1:
            raise RuntimeError('Expected one L3 cache per logical CPU')
        key = (package, (caches[0]/'id').read_text().strip())
        sides.setdefault(key, {}).setdefault((package, core), cpu)
    if len(sides) != 2 or any(len(cores) != 8 for cores in sides.values()):
        raise RuntimeError('Split mode requires two L3 groups with eight available physical cores each')
    ordered = sorted((sorted(cores.values()) for cores in sides.values()), key=lambda x:x[0])
    return [[cpus[:4], cpus[4:]] for cpus in ordered]


class Allocator:
    def __init__(self, groups):
        self.groups = groups
        self.live = {}
        self.next_mask = [0, 0]

    def choose(self):
        self.live = {side:proc for side,proc in self.live.items() if proc.poll() is None}
        free = [side for side in range(2) if side not in self.live]
        if not free:
            raise RuntimeError('Both L3 groups occupied')
        side = free[0]
        return side, self.groups[side][self.next_mask[side]]

    def attach(self, side, process):
        self.live[side] = process
        self.next_mask[side] = 1-self.next_mask[side]
