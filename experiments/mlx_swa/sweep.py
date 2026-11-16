"""Isolated synthetic attention/model sweep; results are JSONL, one case per process."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('full_reference', 'full_stable', 'full_fused', 'swa_dense', 'swa_tiled', 'swa_stable', 'swa_fused')


def worker(case):
    sys.path.insert(0, str(ROOT / 'src'))
    import mlx.core as mx
    import mlx.nn as nn
    from llm_training.models.attention_mlx import masked_attention, SlidingWindowAttention, CausalSelfAttention
    from llm_training.models.config import GPTConfig
    from llm_training.models.gpt_mlx import GPT
    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(42)
    t, w, h, d, b = (case[k] for k in ('length','window','heads','dim','batch'))
    variant = case['variant']
    def attention(q,k,v):
        if variant.endswith('fused'):
            return mx.fast.scaled_dot_product_attention(q,k,v,scale=d**-0.5,mask='causal',
                window_size=w if variant.startswith('swa') else -1,force_fused=True)
        if variant in ('swa_tiled','swa_stable'):
            return SlidingWindowAttention.attend(attn,q,k,v)
        pos=mx.arange(q.shape[2]); distance=pos[:,None]-pos[None,:]
        allowed=distance>=0
        if variant=='swa_dense': allowed=allowed & (distance<w)
        return masked_attention(q,k,v,allowed,stable=variant.endswith('stable'))
    cfg=GPTConfig(vocab_size=8000,context_length=t,n_layers=case.get('layers',1),
                  n_heads=h,d_model=h*d,sliding_window=w,stable_softmax=variant.endswith('stable'))
    attn=SlidingWindowAttention(cfg)
    if case['kind']=='attention':
        q,k,v,u=[mx.random.normal((b,h,t,d)) for _ in range(4)]
        mx.eval(q,k,v,u)
        fn=mx.value_and_grad(lambda q,k,v: mx.mean(attention(q,k,v)*u),argnums=(0,1,2))
        inputs=(q,k,v)
        params=0
    else:
        # Use each candidate in every layer; include native/corrected model CE.
        CausalSelfAttention.attend=lambda self,q,k,v: attention(q,k,v)
        model=GPT(cfg)
        if case.get('mix'):
            def full(q,k,v):
                if variant.endswith('fused'):
                    return mx.fast.scaled_dot_product_attention(q,k,v,scale=d**-0.5,
                        mask='causal',force_fused=True)
                pos=mx.arange(q.shape[2])
                return masked_attention(q,k,v,pos[:,None]>=pos[None,:],
                                        stable=variant.endswith('stable'))
            for i, block in enumerate(model.blocks):
                if (i+1) % case['mix'] == 0:
                    block.attention.attend = full
        mx.eval(model.parameters())
        from mlx.utils import tree_flatten
        params=sum(x.size for _,x in tree_flatten(model.parameters()))
        ids=mx.random.randint(0,8000,(b,t)); mx.eval(ids)
        fn=nn.value_and_grad(model,lambda x:model(x,x)[1]); inputs=(ids,)
    if case['compiled']:
        fn=mx.compile(fn,**({'inputs':model.state,'outputs':model.state} if case['kind']=='model' else {}))
    mx.clear_cache(); mx.reset_peak_memory()
    start=time.perf_counter(); result=fn(*inputs); mx.eval(result)
    cold=time.perf_counter()-start
    assert mx.isfinite(result[0]).item()
    del result
    for _ in range(2): mx.eval(fn(*inputs))
    times=[]
    for _ in range(5):
        start=time.perf_counter(); result=fn(*inputs); mx.eval(result)
        times.append(time.perf_counter()-start); del result
    return dict(status='ok',median_ms=1000*statistics.median(times),min_ms=1000*min(times),
                max_ms=1000*max(times),cold_seconds=cold,peak_gib=mx.get_peak_memory()/1024**3,
                parameters=params,device=mx.device_info()['device_name'])


def run(case):
    # Conservative bound for dense score intermediates before risking system pressure.
    dense=case['variant'] in ('full_reference','full_stable','swa_dense')
    score=case['batch']*case['heads']*case['length']**2*4
    if dense and score*4*case.get('layers',1) > 16*1024**3:
        return dict(status='estimated_budget_exceeded',score_gib=score/1024**3)
    process=subprocess.Popen([sys.executable,__file__,'--case',json.dumps(case)],
                             stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    start=time.monotonic(); peak_rss=0
    while process.poll() is None:
        if time.monotonic()-start>120:
            process.kill(); process.communicate(); return dict(status='timeout_120s')
        ps=subprocess.run(['ps','-o','rss=','-p',str(process.pid)],capture_output=True,text=True)
        try: rss=int(ps.stdout.strip())*1024
        except ValueError: rss=0
        peak_rss=max(peak_rss,rss)
        if rss>16*1024**3:
            process.kill(); process.communicate(); return dict(status='rss_budget_exceeded',rss_gib=rss/1024**3)
        time.sleep(.25)
    stdout,stderr=process.communicate()
    try: result=json.loads(stdout.strip().splitlines()[-1])
    except (ValueError,IndexError): result=dict(status='process_error',error=(stderr+stdout)[-2000:])
    result['peak_rss_gib']=peak_rss/1024**3
    return result


def cases():
    for t in (256,1024,4096,8192,16384):
        for v in VARIANTS:
            yield dict(kind='attention',variant=v,length=t,window=256,heads=4,dim=64,batch=1,compiled=False)
    for v in VARIANTS:
        yield dict(kind='attention',variant=v,length=4096,window=256,heads=4,dim=64,batch=1,compiled=True)
    for t,w in ((4096,64),(4096,1024),(32768,256),(65536,256)):
        for v in ('swa_tiled','swa_fused','full_fused'):
            yield dict(kind='attention',variant=v,length=t,window=w,heads=4,dim=64,batch=1,compiled=False)
    for t in (65536,131072):
        for v in ('swa_stable','swa_tiled','swa_fused'):
            yield dict(kind='attention',variant=v,length=t,window=256,heads=4,dim=64,batch=1,compiled=False)
    for v in ('swa_tiled','swa_stable','swa_fused','full_fused'):
        yield dict(kind='model',variant=v,length=4096,window=256,heads=8,dim=64,batch=1,compiled=True,layers=6)
    for mix in (2,4):
        for v in ('swa_tiled','swa_stable','swa_fused'):
            yield dict(kind='model',variant=v,length=4096,window=256,heads=4,dim=64,batch=1,compiled=True,layers=4,mix=mix)
    for t,h,layers in ((1024,2,4),(4096,2,4),(4096,8,6)):
        for v in ('full_reference','full_fused','swa_tiled','swa_stable','swa_fused'):
            yield dict(kind='model',variant=v,length=t,window=256,heads=h,dim=64,batch=1,compiled=False,layers=layers)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--case'); p.add_argument('--output',default='artifacts/benchmarks/swa_sweep.jsonl'); a=p.parse_args()
    if a.case:
        try: result=worker(json.loads(a.case))
        except Exception as e: result=dict(status='error',error=str(e))
        print(json.dumps(result),flush=True)
    else:
        path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
        completed=set()
        if path.exists():
            for line in path.read_text().splitlines():
                row=json.loads(line);completed.add(json.dumps(row['case'],sort_keys=True))
        for case in cases():
            if json.dumps(case,sort_keys=True) in completed: continue
            result=run(case); row=dict(case=case,result=result)
            with path.open('a') as f: f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
