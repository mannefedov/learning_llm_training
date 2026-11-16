"""Precision-preserving step audit; temporary models, no training checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from tokenizers import Tokenizer

from profile_training import timing
from llm_training.models.gpt_mlx import GPT, GPTConfig
from llm_training.pretraining_mlx.data import sequential_batch
from llm_training.pretraining_mlx.train import TrainConfig, make_optimizer, make_loss_and_grad


def worker(args):
    config = tomllib.loads(Path('configs/pretraining/sequential_gpt_mlx_swa_fused_4k.toml').read_text())
    cfg = GPTConfig(vocab_size=Tokenizer.from_file(config['tokenizer']['path']).get_vocab_size(), **config['model'])
    tc = TrainConfig(**{**config['training'], 'batch_size': args.batch})
    data = np.memmap(Path(config['cache']['directory'])/'train.bin', dtype='<u4', mode='r')
    x, y = sequential_batch(data, args.batch, cfg.context_length, 0)
    mx.random.seed(42)
    model = GPT(cfg)
    mx.eval(model.parameters(), x, y)
    fg = make_loss_and_grad(model, True)
    loss, grads = fg(x, y)
    mx.eval(loss, grads)
    arrays = {k: np.array(v) for k, v in tree_flatten(grads)}
    reference = Path(args.reference)
    errors = {}
    if reference.exists():
        with np.load(reference) as expected:
            for k, v in arrays.items():
                # Reordered matmuls need not be bitwise identical.
                np.testing.assert_allclose(v, expected[k], atol=3e-4, rtol=3e-3, err_msg=k)
            errors['max_gradient_abs_error'] = max(float(np.max(np.abs(v-expected[k]))) for k, v in arrays.items())
    else:
        np.savez(reference, **arrays)
    result = {'batch':args.batch, 'bq':os.environ.get('MLX_SDPA_VJP_BQ', '1024'),
              'bk':os.environ.get('MLX_SDPA_VJP_BK', '1024'), 'loss':loss.item(),
              'mlx_version': mx.__version__, 'model': config['model'], **errors}
    forward = mx.compile(lambda x,y:model(x,y)[1], inputs=model.state, outputs=model.state)
    result['forward'] = timing(lambda:forward(x,y))
    result['forward_backward'] = timing(lambda:fg(x,y))
    if args.details:
        # Does avoiding one large logits allocation help the output head?
        h = mx.random.normal((args.batch*cfg.context_length,cfg.d_model))*.02
        w = model.token_embedding.weight
        labels = y.reshape(-1)
        mx.eval(h,w,labels)
        expected_head = None
        for chunks in (1,2,4,8):
            def head(h,w):
                return sum(nn.losses.cross_entropy(a@w.T,b,reduction='sum')
                           for a,b in zip(mx.split(h,chunks),mx.split(labels,chunks)))/labels.size
            head_grad = mx.compile(mx.value_and_grad(head,argnums=(0,1)))
            values = head_grad(h,w)
            mx.eval(values)
            if expected_head is None:
                expected_head = values
            else:
                for a,b in zip(tree_flatten(values),tree_flatten(expected_head)):
                    np.testing.assert_allclose(np.array(a[1]),np.array(b[1]),atol=1e-5,rtol=1e-3)
            result[f'head_chunks_{chunks}'] = timing(lambda:head_grad(h,w))
        # Compare independent models with identical initialization and 13 updates.
        trajectories = []
        for mode in ('current', 'evaluate_gradients_before_check', 'compiled_whole_step'):
            mx.random.seed(42)
            m = GPT(cfg)
            opt = make_optimizer(tc)
            grad = make_loss_and_grad(m, True)
            raw_grad = nn.value_and_grad(m, lambda x,y:m(x,y)[1])
            opt.init(m.trainable_parameters())
            mx.eval(m.parameters(), opt.state)
            def update(g):
                g, _ = optim.clip_grad_norm(g, tc.grad_clip)
                opt.update(m, g)
            def whole(x,y):
                l,g = raw_grad(x,y)
                update(g)
                return l
            compiled = mx.compile(whole, inputs=[m.state,opt.state], outputs=[m.state,opt.state])
            trajectory = []
            def step():
                if mode == 'compiled_whole_step':
                    l = compiled(x,y)
                    mx.eval(l,m.parameters(),opt.state)
                    # Diagnostic only: this variant checks AFTER updating.
                    if not mx.isfinite(l).item(): raise ValueError('nonfinite loss')
                else:
                    l,g = grad(x,y)
                    if mode == 'evaluate_gradients_before_check': mx.eval(l,g)
                    if not mx.isfinite(l).item(): raise ValueError('nonfinite loss')
                    update(g)
                    mx.eval(m.parameters(),opt.state)
                trajectory.append(l.item())
                return l
            result[mode] = timing(step)
            trajectories.append(trajectory)
        for trajectory in trajectories[1:]:
            np.testing.assert_allclose(trajectory, trajectories[0], atol=1e-4, rtol=1e-4)
        result['update_loss_max_error'] = float(np.max(np.abs(np.array(trajectories)-trajectories[0])))
        # Isolate layout copies and native attention, excluding projections.
        packed = mx.random.normal((args.batch, cfg.context_length, 3*cfg.d_model))
        q,k,v = [a.reshape(args.batch,cfg.context_length,cfg.n_heads,-1).transpose(0,2,1,3)
                 for a in mx.split(packed,3,axis=-1)]
        u = mx.random.normal(q.shape)*.01
        mx.eval(q,k,v,u)
        for window in (-1, cfg.sliding_window):
            def attention(q,k,v):
                return mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-.5,
                    mask='causal',force_fused=True,window_size=window)
            f = mx.compile(attention)
            g = mx.compile(mx.value_and_grad(lambda q,k,v:mx.sum(attention(q,k,v)*u),argnums=(0,1,2)))
            contiguous = tuple(mx.contiguous(a) for a in (q,k,v))
            mx.eval(contiguous)
            for layout, values in (('strided',(q,k,v)),('contiguous',contiguous)):
                result[f'attention_{window}_{layout}_forward'] = timing(lambda:f(*values))
                result[f'attention_{window}_{layout}_forward_backward'] = timing(lambda:g(*values))
            result[f'attention_{window}_copies'] = timing(lambda:tuple(mx.contiguous(a) for a in (q,k,v)))
    return result


def sweep(args):
    # Separate processes: MLX caches environment settings during initialization.
    cases = [(2,1024,1024,True),(2,256,256,False),(2,512,512,False),
             (2,2048,2048,False),(2,512,1024,False),(2,1024,512,False),
             (2,1024,2048,False),(2,2048,1024,False),(2,128,128,False),
             (8,1024,1024,True),(8,512,512,False),(8,2048,2048,False)]
    if args.confirm:
        cases = [(batch,tile,tile,True) for batch in (2,4,8) for tile in (1024,256)]
    if args.small_tiles:
        cases = [(2,bq,bk,False) for bq,bk in
                 ((1024,1024),(256,256),(256,128),(128,256),(256,512),(512,256),(64,256),(256,64))]
    with tempfile.TemporaryDirectory(prefix='mlx-step-audit-') as temp:
        for batch,bq,bk,details in cases:
            command=[sys.executable,__file__,'--batch',str(batch),'--reference',f'{temp}/batch{batch}.npz']
            if details: command.append('--details')
            env={**os.environ,'MLX_SDPA_VJP_BQ':str(bq),'MLX_SDPA_VJP_BK':str(bk)}
            completed=subprocess.run(command,env=env,text=True,capture_output=True,timeout=180)
            if completed.returncode: raise RuntimeError(completed.stderr)
            print(completed.stdout.strip(),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--sweep',action='store_true')
    p.add_argument('--confirm',action='store_true')
    p.add_argument('--small-tiles',action='store_true')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--reference')
    p.add_argument('--details',action='store_true')
    args=p.parse_args()
    if args.sweep: sweep(args)
    else: print(json.dumps(worker(args)),flush=True)
