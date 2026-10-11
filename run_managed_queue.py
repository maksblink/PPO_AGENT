"""Run the two-worker PPO queue with temporary hardware controls.

Hardware preparation/restoration lives here. Dependency scheduling and monitoring
live in run_search_queue.py and queue_execution. No database schema changes.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog='Pass queue options such as --queue NAME --select 1-10. --dry-run never changes hardware.')
    parser.add_argument('--max-mhz',type=int,default=5000)
    parser.add_argument('--dry-run',action='store_true')
    args,queue_args=parser.parse_known_args()
    controlled={'--workers','--threads','--serial-evaluation','--cpu-affinity',
                '--control-seconds','--sample-seconds','--flush-seconds',
                '--temperature-limit','--temperature-resume','--abort-temperature'}
    if any(a.split('=')[0] in controlled for a in queue_args):
        parser.error('Managed worker/thermal/monitoring settings are fixed; use run_search_queue.py for custom queue settings')
    fixed=['--workers','2','--threads','1','--serial-evaluation','--cpu-affinity','split',
           '--temperature-limit','91.5','--temperature-resume','84.5','--abort-temperature','97.5',
           '--control-seconds','3','--sample-seconds','6','--flush-seconds','30']
    root=Path(__file__).resolve().parent
    import run_search_queue as queue
    parsed=queue.parse_args([*queue_args,*fixed])
    command=[sys.executable,str(root/'run_search_queue.py'),*queue_args,*fixed]
    if args.dry_run or parsed.summary_only:
        return subprocess.call(command+(['--dry-run'] if args.dry_run else []),cwd=root)
    queue.assert_git_clean()
    from train_and_eval.queue_execution.affinity import discover_groups
    from train_and_eval.queue_execution.scheduler import other_local_queue
    from train_and_eval.queue_execution.hardware import thermal_session
    groups=discover_groups()
    import yaml
    for path in queue.load_queue_config_paths(parsed.queue):
        cfg = yaml.safe_load((root/path).read_text())
        if cfg.get('ppo',{}).get('device')!='cuda' or cfg.get('evaluation',{}).get('device')!='cpu':
            raise RuntimeError(f'{path}: managed queue requires CUDA training and CPU evaluation')
    if other_local_queue(root) is not None:raise RuntimeError('Another queue is active; no hardware settings changed')
    # Validate queue selection/dependencies before sudo or hardware writes.
    validation=subprocess.run(command+['--dry-run'],cwd=root,
                              stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    if validation.returncode:
        print(validation.stdout, end='', flush=True)
        raise RuntimeError('Queue validation failed; hardware settings were not changed')
    artifacts=(root/'artifacts').resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    directory=artifacts/'search_queues'/parsed.queue/stamp
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'hardware.json').write_text(json.dumps({'arguments':queue_args,'fixed':fixed,'groups':groups,'max_mhz':args.max_mhz},indent=2))
    lock_path=Path('/tmp')/f'ppo-managed-hardware-{os.getuid()}.lock'
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another managed hardware session is active')
        with thermal_session(directory,args.max_mhz) as cooldown:
            cooldown()
            env=os.environ.copy()
            env['PPO_QUEUE_SESSION_DIR']=str(directory)
            process=subprocess.Popen(command,cwd=root,env=env,start_new_session=True)
            try:
                code=process.wait()
            finally:
                # Stop the scheduler first, allowing it to reap its workers before
                # restoring fan/power limits. Its own stop_children has escalation.
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:process.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        # Do not restore settings while unaccounted compute continues.
                        print('Scheduler shutdown delayed; waiting for owned worker cleanup.',flush=True)
                        process.wait()
    print(f'Hardware session: {directory}',flush=True)
    return code


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
