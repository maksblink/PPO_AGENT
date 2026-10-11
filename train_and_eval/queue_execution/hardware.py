"""Temporary Legion hardware settings; never used by queue planning or workers."""
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import time
from train_and_eval.queue_execution.telemetry import temperatures

def cpu_temperature():
    values=temperatures()
    return max(values.values()) if values else None

class SudoSession:
    """Authenticate once, refresh without prompts until restoration finishes."""
    def __init__(self, interval=45):
        import threading
        self.interval = interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.refresh, daemon=True)

    def __enter__(self):
        subprocess.run(['sudo', '-v'], check=True)
        # Fail before changing hardware if noninteractive validation is blocked.
        subprocess.run(['sudo', '-n', '-v'], check=True)
        self.thread.start()
        return self

    def refresh(self):
        warned = False
        while not self.stop.wait(self.interval):
            try:
                result = subprocess.run(['sudo', '-n', '-v'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, timeout=15)
                if result.returncode:
                    raise RuntimeError(result.stderr.strip() or 'sudo validation failed')
                warned = False
            except Exception as exc:
                if not warned:
                    print(f'SUDO REFRESH FAILED: {exc}; automatic restoration may fail.', file=sys.stderr, flush=True)
                    warned = True

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(20)
        # Do not invalidate the user's shared sudo timestamp with sudo -k.


@contextmanager
def thermal_session(directory, max_mhz, sysroot=Path("/sys")):
    import signal
    legion=sysroot/'devices/platform/legion'
    boost=sysroot/'devices/system/cpu/cpufreq/boost'
    limits=sorted((sysroot/'devices/system/cpu/cpufreq').glob('policy*/scaling_max_freq'))
    if not limits: raise RuntimeError('CPU frequency policies missing')
    if (sysroot/'class/power_supply/ADP0/online').read_text().strip()!='1':
        raise RuntimeError('Connect AC power')
    for path in limits:
        low=int((path.parent/'scaling_min_freq').read_text())
        high=int((path.parent/'cpuinfo_max_freq').read_text())
        if not low<=max_mhz*1000<=high:raise RuntimeError(f'Invalid cap for {path}')
    paths=[boost,legion/'powermode',*limits,legion/'fan_fullspeed']
    previous={str(p):p.read_text().strip() for p in paths}
    record=directory/'hardware.json'
    metadata=json.loads(record.read_text()) if record.exists() else {}
    metadata.update({'max_mhz':max_mhz,'previous':previous,'completed':False})
    record.write_text(json.dumps(metadata,indent=2))
    def write(path,value):
        subprocess.run(['sudo','-n','tee',str(path)],input=value+'\n',text=True,stdout=subprocess.DEVNULL,check=True)
        if path.read_text().strip()!=value:raise RuntimeError(f'Setting readback failed: {path}')
    def cooldown():
        stable=None; deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            temp=cpu_temperature();now=time.monotonic()
            if temp is None:raise RuntimeError('CPU temperature missing')
            if temp<=65:
                if stable is None:stable=now
                if now-stable>=30:return
            else:stable=None
            print(f'Cooldown: {temp:.1f} C; require <=65 C for 30 seconds',flush=True)
            time.sleep(5)
        raise RuntimeError('Cooldown timeout')
    def stop(signum,frame):raise KeyboardInterrupt(f'Signal {signum}')
    with SudoSession():
        handler=signal.signal(signal.SIGTERM,stop)
        try:
            write(legion/'powermode','255')
            time.sleep(3)
            write(legion/'fan_fullspeed','1')
            write(boost,'1')
            for path in limits:write(path,str(max_mhz*1000))
            metadata['applied']={str(p):p.read_text().strip() for p in paths}
            record.write_text(json.dumps(metadata,indent=2))
            yield cooldown
            metadata['completed']=True
        finally:
            errors=[]
            for path in paths:
                try:write(path,previous[str(path)])
                except Exception as exc:errors.append(f'{path}: {exc}')
            metadata['restore_errors']=errors
            record.write_text(json.dumps(metadata,indent=2))
            signal.signal(signal.SIGTERM,handler)
            if errors:
                print('RESTORE ERRORS: '+'; '.join(errors),file=sys.stderr,flush=True)
                raise RuntimeError('Hardware restoration incomplete; see hardware.json')
            print('Restored previous CPU limits, boost and fan settings.',flush=True)
