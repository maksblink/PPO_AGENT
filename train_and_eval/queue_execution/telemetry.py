"""Bounded host telemetry: one CSV, 30-second flush, gzip on close."""
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import subprocess
import time


def temperatures(root=Path('/sys/class/hwmon')):
    values = {}
    for directory in root.glob('hwmon*'):
        try: driver = (directory/'name').read_text().strip()
        except OSError: continue
        if driver not in ('k10temp', 'coretemp', 'zenpower'): continue
        for sensor in directory.glob('temp*_input'):
            try:
                value = float(sensor.read_text())/1000
                if not math.isfinite(value) or not 0<value<150: continue
                labelpath = sensor.with_name(sensor.name.replace('_input', '_label'))
                label = labelpath.read_text().strip() if labelpath.exists() else sensor.stem
                values[f'{driver}/{label}'] = value
            except (OSError,ValueError): continue
    return values


def cpu_ticks(path=Path('/proc/stat')):
    rows = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith('cpu'): continue
        ticks = [int(v) for v in parts[1:9]]
        rows[parts[0]] = (sum(ticks), ticks[3]+ticks[4])
    return rows


def utilization(previous, current, keys):
    if not previous or any(k not in previous or k not in current for k in keys): return None
    total = sum(current[k][0]-previous[k][0] for k in keys)
    idle = sum(current[k][1]-previous[k][1] for k in keys)
    return round(100*(1-idle/total),3) if total>0 else None


def memory():
    entries = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(entries['MemAvailable'].split()[0])/1024


def gpus():
    try:
        out = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu',
                              '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=1,check=True)
        return [line.strip().split(', ') for line in out.stdout.splitlines()]
    except (OSError,subprocess.SubprocessError): return []


class Recorder:
    def __init__(self, path, groups=None, flush_seconds=30):
        self.path=Path(path)
        self.stream=self.path.open('x',newline='',buffering=65536)
        self.groups=[mask for side in (groups or []) for mask in side]
        self.fields=['timestamp','cpu_c','cpu_sensors','cpu_utilization_pct']+[
            f'cpu_group_{i}_pct' for i in range(len(self.groups))]+['available_ram_mb','gpus','workers']
        self.writer=csv.DictWriter(self.stream,fieldnames=self.fields);self.writer.writeheader()
        self.previous=cpu_ticks();self.last_flush=time.monotonic();self.flush_seconds=flush_seconds
        self.stats={};self.rows=0

    def record(self, workers, sensors):
        current=cpu_ticks()
        row={'timestamp':time.time(),'cpu_c':max(sensors.values()) if sensors else None,
             'cpu_sensors':json.dumps(sensors,separators=(',',':')),
             'cpu_utilization_pct':utilization(self.previous,current,['cpu']),
             'available_ram_mb':memory(),'gpus':json.dumps(gpus(),separators=(',',':')),
             'workers':json.dumps(workers,separators=(',',':'))}
        for i,cpus in enumerate(self.groups):
            row[f'cpu_group_{i}_pct']=utilization(self.previous,current,[f'cpu{c}' for c in cpus])
        self.previous=current
        self.writer.writerow(row);self.rows+=1
        numeric=dict(row, **{'sensor_'+k:v for k,v in sensors.items()})
        for k,v in numeric.items():
            if k=='timestamp' or not isinstance(v,(int,float)):continue
            a=self.stats.setdefault(k,{'min':v,'max':v,'sum':0,'count':0})
            a['min']=min(a['min'],v);a['max']=max(a['max'],v);a['sum']+=v;a['count']+=1
        for gpu in json.loads(row['gpus']):
            if len(gpu)!=4:continue
            for label,value in zip(('utilization_pct','vram_mb','temperature_c'),gpu[1:]):
                try:v=float(value)
                except ValueError:continue
                k=f'gpu_{gpu[0]}_{label}';a=self.stats.setdefault(k,{'min':v,'max':v,'sum':0,'count':0})
                a['min']=min(a['min'],v);a['max']=max(a['max'],v);a['sum']+=v;a['count']+=1
        self.flush_due()

    def flush_due(self):
        if time.monotonic()-self.last_flush>=self.flush_seconds:
            self.stream.flush();self.last_flush=time.monotonic()

    def close(self):
        self.stream.close()
        packed=self.path.with_suffix(self.path.suffix+'.gz')
        # Retain original on any compression failure; never overwrite a prior output.
        with self.path.open('rb') as src, packed.open('xb') as out:
            with gzip.GzipFile(fileobj=out,mode='wb') as dest:shutil.copyfileobj(src,dest)
        self.path.unlink()
        summary={k:{'min':v['min'],'max':v['max'],'mean':v['sum']/v['count'],'samples':v['count']}
                 for k,v in self.stats.items()}
        (self.path.parent/'telemetry_summary.json').write_text(json.dumps({'rows':self.rows,'metrics':summary},indent=2))
