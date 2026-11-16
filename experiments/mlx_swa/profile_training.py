"""Profile synthetic updates at real config shapes without changing a training run."""
import argparse
import json
from pathlib import Path
import statistics
import time
import tomllib

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer

from llm_training.models.gpt_mlx import GPT, GPTConfig
from llm_training.pretraining_mlx.data import sequential_batch
from llm_training.pretraining_mlx.train import TrainConfig, make_optimizer, make_loss_and_grad


def timing(fn, count=10):
    for _ in range(3): mx.eval(fn())
    times=[]
    for _ in range(count):
        start=time.perf_counter(); out=fn(); mx.eval(out)
        times.append((time.perf_counter()-start)*1000)
    return {'median_ms':statistics.median(times),'min_ms':min(times),'max_ms':max(times)}


def profile(config, batch):
    cache_limit=mx.set_cache_limit(0)
    mx.set_cache_limit(cache_limit)
    tokenizer=Tokenizer.from_file(config['tokenizer']['path'])
    cfg=GPTConfig(vocab_size=tokenizer.get_vocab_size(),**config['model'])
    train_cfg=TrainConfig(**{**config['training'],'batch_size':batch})
    data=np.memmap(Path(config['cache']['directory'])/'train.bin',dtype='<u4',mode='r')
    x,y=sequential_batch(data,batch,cfg.context_length,0);mx.eval(x,y)
    result={'batch':batch,'tokens_per_step':batch*cfg.context_length,
            'cache_limit_gib':cache_limit/1024**3}
    result['data_batch']=timing(lambda:sequential_batch(data,batch,cfg.context_length,1))
    losses=[]
    for variant in ('current','compiled_optimizer'):
        mx.random.seed(42);model=GPT(cfg);optimizer=make_optimizer(train_cfg)
        mx.eval(model.parameters())
        fg=make_loss_and_grad(model,True)
        def update(grads):
            grads,_=optim.clip_grad_norm(grads,train_cfg.grad_clip)
            optimizer.update(model,grads)
        if variant=='compiled_optimizer':
            # Initialize optimizer buffers before capturing their state.
            _,g=fg(x,y);mx.eval(g);optimizer.init(model.trainable_parameters())
            update=mx.compile(update,inputs=[model.state,optimizer.state],outputs=[model.state,optimizer.state])
        trajectory=[]
        def step():
            loss,g=fg(x,y)
            if not mx.isfinite(loss).item(): raise ValueError('nonfinite loss')
            update(g);mx.eval(model.parameters(),optimizer.state)
            trajectory.append(loss.item())
            return loss
        mx.clear_cache();mx.reset_peak_memory()
        result[variant]=timing(step)
        result[variant]['tokens_per_second']=result['tokens_per_step']/(result[variant]['median_ms']/1000)
        result[variant]['peak_gib']=mx.get_peak_memory()/1024**3
        losses.append(trajectory)
    np.testing.assert_allclose(losses[0],losses[1],atol=1e-4,rtol=1e-4)
    result['optimizer_loss_trajectory_max_error']=float(np.max(np.abs(np.array(losses[0])-losses[1])))
    # Synchronized component times are diagnostic; they need not sum to a full step.
    result['full_forward_backward']=timing(lambda:fg(x,y))
    loss,g=fg(x,y);mx.eval(loss,g)
    result['compiled_update_only']=timing(lambda:(update(g),mx.eval(model.parameters(),optimizer.state)))
    del fg,optimizer,update,g
    upstream=mx.random.normal((batch,cfg.context_length,cfg.d_model))*.01;mx.eval(upstream)
    def trunk(m,x):
        h=m.token_embedding(x)+m.position_embedding(mx.arange(x.shape[1]))
        for block in m.blocks:h=block(h)
        return mx.sum(m.final_norm(h)*upstream)
    trunk_grad=mx.compile(nn.value_and_grad(model,lambda x:trunk(model,x)),inputs=model.state,outputs=model.state)
    result['trunk_forward_backward']=timing(lambda:trunk_grad(x))
    del trunk_grad
    hidden=mx.random.normal(upstream.shape)
    weights=model.token_embedding.weight
    for dtype in (mx.float32,mx.bfloat16):
        h,w=hidden.astype(dtype),weights.astype(dtype);mx.eval(h,w)
        fn=mx.compile(mx.value_and_grad(lambda h,w:nn.losses.cross_entropy(h@w.T,y,reduction='mean'),argnums=(0,1)))
        result['output_head_'+str(dtype).split('.')[-1]]=timing(lambda:fn(h,w))
    for label, module in (('local_attention',model.blocks[0].attention),
                          ('full_attention',model.blocks[-1].attention),
                          ('mlp',model.blocks[0].mlp)):
        for dtype in (mx.float32,mx.bfloat16):
            module.set_dtype(dtype)
            h,u=hidden.astype(dtype),upstream.astype(dtype);mx.eval(module.parameters(),h,u)
            def component(params,h):
                module.update(params)
                return mx.sum((module(h)*u).astype(mx.float32))
            fn=mx.compile(mx.value_and_grad(component,argnums=(0,1)),
                          inputs=module.state,outputs=module.state)
            result[label+'_'+str(dtype).split('.')[-1]]=timing(lambda:fn(module.trainable_parameters(),h))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/pretraining/sequential_gpt_mlx_swa_fused_4k.toml');p.add_argument('--batch',type=int,required=True);a=p.parse_args()
    mx.set_default_device(mx.gpu)
    config=tomllib.loads(Path(a.config).read_text())
    print(json.dumps(profile(config,a.batch)),flush=True)
