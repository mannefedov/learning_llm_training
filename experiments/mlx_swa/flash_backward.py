"""Test and benchmark the opt-in on-chip FP32/D64 attention backward."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

import mlx.core as mx
import numpy as np

from profile_training import timing


def guards(tf32=False):
    cases = [(64,4,33,True,mx.float32)] if tf32 else [
        (128,4,33,True,mx.float32), (64,2,33,True,mx.float32),
        (64,4,65,True,mx.float32), (64,4,33,False,mx.float32),
        (64,4,33,True,mx.float16)]
    for dim,heads,length,causal,dtype in cases:
        q = mx.ones((1,4,33,dim),dtype=dtype)
        k = v = mx.ones((1,heads,length,dim),dtype=dtype)
        def loss(q):
            return mx.sum(mx.fast.scaled_dot_product_attention(
                q,k,v,scale=dim**-.5,mask='causal' if causal else None,force_fused=True))
        try:
            mx.eval(mx.grad(loss)(q))
        except (ValueError,RuntimeError) as error:
            if '[SDPA flash64 experiment]' not in str(error):
                raise
        else:
            raise AssertionError('unsupported experimental configuration accepted')
    return {'guards_passed':len(cases),'tf32_guard':tf32}


def model_check(args):
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from tokenizers import Tokenizer
    from llm_training.models.gpt_mlx import GPT, GPTConfig
    from llm_training.pretraining_mlx.data import sequential_batch
    from llm_training.pretraining_mlx.train import TrainConfig, make_optimizer, make_loss_and_grad

    config = tomllib.loads(Path('configs/pretraining/sequential_gpt_mlx_swa_fused_4k.toml').read_text())
    cfg = GPTConfig(vocab_size=Tokenizer.from_file(config['tokenizer']['path']).get_vocab_size(), **config['model'])
    tc = TrainConfig(**{**config['training'], 'batch_size':args.batch})
    data = np.memmap(Path(config['cache']['directory'])/'train.bin', dtype='<u4', mode='r')
    x,y = sequential_batch(data,args.batch,cfg.context_length,0)
    mx.random.seed(42)
    model = GPT(cfg)
    opt = make_optimizer(tc)
    fg = make_loss_and_grad(model,True)
    mx.eval(model.parameters(),x,y)
    initial_loss, initial_grad = fg(x,y)
    mx.eval(initial_loss,initial_grad)
    initial = {'initial_gradient/'+k:np.array(v) for k,v in tree_flatten(initial_grad)}
    del initial_grad, initial_loss
    losses = []
    def step():
        loss,g = fg(x,y)
        if not mx.isfinite(loss).item():
            raise ValueError('nonfinite loss')
        g,_ = optim.clip_grad_norm(g,tc.grad_clip)
        opt.update(model,g)
        mx.eval(model.parameters(),opt.state)
        losses.append(loss.item())
        return loss
    mx.clear_cache()
    mx.reset_peak_memory()
    measured = timing(step)
    peak = mx.get_peak_memory()/1024**2
    arrays = {'param/'+k:np.array(v) for k,v in tree_flatten(model.parameters())}
    arrays.update({'optimizer/'+k:np.array(v) for k,v in tree_flatten(opt.state)})
    arrays['losses'] = np.array(losses)
    arrays.update(initial)
    errors = {}
    if Path(args.reference).exists():
        with np.load(args.reference) as expected:
            failed = []
            for key,value in arrays.items():
                try:
                    np.testing.assert_allclose(value,expected[key],atol=5e-5,rtol=5e-5,err_msg=key)
                except AssertionError:
                    if not args.diagnostic_model or key.startswith('initial_gradient/'):
                        raise
                    failed.append(key)
            errors = {'loss_max_abs':float(np.max(np.abs(arrays['losses']-expected['losses']))),
                      'parameter_max_abs':max(float(np.max(np.abs(value-expected[key])))
                          for key,value in arrays.items() if key.startswith('param/')),
                      'initial_gradient_max_abs':max(float(np.max(np.abs(value-expected[key])))
                          for key,value in arrays.items() if key.startswith('initial_gradient/')),
                      'strict_failed_arrays':failed}
    else:
        np.savez(args.reference,**arrays)
    return {'batch':args.batch,'model':config['model'],
            'flash64':os.environ.get('MLX_SDPA_VJP_FLASH64','0'),
            'nax_tile':os.environ.get('MLX_SDPA_VJP_FLASH64_TILE','32'),
            'backward_bq':os.environ.get('MLX_SDPA_VJP_BQ','1024'),
            'backward_bk':os.environ.get('MLX_SDPA_VJP_BK','1024'),
            'tf32':os.environ.get('MLX_ENABLE_TF32','1'), 'timing':measured,
            'tokens_per_second':args.batch*cfg.context_length*1000/measured['median_ms'],
            'peak_mib':peak,'losses':losses,'errors':errors}


def model_suite(nax=False,diagnostic=False):
    with tempfile.TemporaryDirectory(prefix='mlx-flash64-model-') as temp:
        cases = ((1,0),(1,2)) if nax else ((1,0),(0,0),(0,1))
        for tf32,flash in cases:
            command = [sys.executable,__file__,'--model','--reference',f'{temp}/tf32-{tf32}.npz']
            if diagnostic:
                command.append('--diagnostic-model')
            env = {**os.environ,'MLX_ENABLE_TF32':str(tf32),'MLX_SDPA_VJP_FLASH64':str(flash)}
            result = subprocess.run(command,env=env,text=True,capture_output=True,timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr)
            print(result.stdout.strip(),flush=True)


def sensitivity_suite(confirm=False):
    # Include a changed tiled reduction order and a repeated baseline as controls.
    with tempfile.TemporaryDirectory(prefix='mlx-nax-sensitivity-') as temp:
        cases = [(2,mode,tile,block) for mode,tile,block in
                 ((0,32,1024),(2,16,1024),(2,32,1024),(2,64,1024),(0,32,256),(0,32,1024))]
        if confirm:
            cases = [(batch,mode,16,256) for batch,mode in
                     ((2,0),(2,2),(2,2),(2,0),(8,0),(8,2))]
        for batch,mode,tile,block in cases:
            command = [sys.executable,__file__,'--model','--diagnostic-model',
                       '--batch',str(batch),'--reference',f'{temp}/native-{batch}.npz']
            env = {**os.environ,'MLX_ENABLE_TF32':'1','MLX_SDPA_VJP_FLASH64':str(mode),
                   'MLX_SDPA_VJP_FLASH64_TILE':str(tile),'MLX_SDPA_VJP_BQ':str(block),
                   'MLX_SDPA_VJP_BK':str(block)}
            result = subprocess.run(command,env=env,text=True,capture_output=True,timeout=180)
            if result.returncode:
                raise RuntimeError(result.stderr)
            print(result.stdout.strip(),flush=True)


def diagnose():
    mx.random.seed(7)
    q,k,v,u = [mx.random.normal((1,4,129,64))*3 for _ in range(4)]
    def attention(q,k,v):
        return mx.fast.scaled_dot_product_attention(q,k,v,scale=1/8,mask='causal',force_fused=True)
    out = attention(q,k,v)
    gradients = mx.grad(lambda q,k,v:mx.sum(attention(q,k,v)*u),argnums=(0,1,2))(q,k,v)
    mx.eval(out,gradients)
    a,b,c,g = [np.array(x).astype(np.float64) for x in (q,k,v,u)]
    scores = (a@b.swapaxes(-1,-2))/8
    scores = np.where(np.arange(129)[:,None]>=np.arange(129)[None,:],scores,-np.inf)
    prob = np.exp(scores-np.max(scores,axis=-1,keepdims=True))
    prob /= prob.sum(axis=-1,keepdims=True)
    dp = g@c.swapaxes(-1,-2)
    ds = prob*(dp-(dp*prob).sum(axis=-1,keepdims=True))/8
    expected = (ds@b,ds.swapaxes(-1,-2)@a,prob.swapaxes(-1,-2)@g)
    errors = {}
    for key,x,y in zip(('output','dq','dk','dv'),(out,*gradients),(prob@c,*expected)):
        difference = np.array(x)-y
        errors[key] = {'max_abs':float(np.abs(difference).max()),
                       'relative_l2':float(np.linalg.norm(difference)/np.linalg.norm(y))}
    return {'tf32':os.environ.get('MLX_ENABLE_TF32','1'),
            'flash64':os.environ.get('MLX_SDPA_VJP_FLASH64','0'),'errors':errors}


def check(args):
    mx.random.seed(7)
    shape = (args.batch, 4, args.length, 64)
    # Split and transpose the same layout produced by the packed QKV projection.
    packed = mx.random.normal((args.batch, args.length, 3 * 256)) * args.input_scale
    q, k, v = [a.reshape(args.batch, args.length, 4, 64).transpose(0, 2, 1, 3)
               for a in mx.split(packed, 3, axis=-1)]
    upstream = mx.random.normal(shape)
    mx.eval(q, k, v, upstream)

    def attention(q, k, v):
        return mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1/8, mask='causal', window_size=args.window,
            force_fused=True)

    def objective(q, k, v):
        return mx.sum(attention(q, k, v) * upstream)

    grad = mx.compile(mx.value_and_grad(objective, argnums=(0, 1, 2)))
    loss, gradients = grad(q, k, v)
    mx.eval(loss, gradients)
    arrays = {name: np.array(value) for name, value in zip(('dq', 'dk', 'dv'), gradients)}
    path = Path(args.reference)
    errors = {}
    reference_errors = {}
    if path.exists():
        with np.load(path) as expected:
            for key, value in arrays.items():
                np.testing.assert_allclose(value, expected[key], atol=5e-5, rtol=5e-5, err_msg=key)
                errors[key] = float(np.max(np.abs(value - expected[key])))
    else:
        np.savez(path, **arrays)
    if args.length <= 257:
        pos = mx.arange(args.length)
        allowed = pos[:, None] >= pos[None, :]
        if args.window > 0:
            allowed = allowed & (pos[:, None] - pos[None, :] < args.window)
        def reference(q,k,v):
            scores = mx.where(allowed, (q @ k.swapaxes(-1,-2))/8, -float('inf'))
            return mx.sum((mx.softmax(scores, axis=-1) @ v)*upstream)
        ref_grad = mx.grad(reference, argnums=(0,1,2))(q,k,v)
        mx.eval(ref_grad)
        for key, expected in zip(('dq','dk','dv'), ref_grad):
            reference_errors[key] = float(np.max(np.abs(arrays[key]-np.array(expected))))
            # TF32's native backward already differs from explicit attention,
            # notably at W1. Keep strict candidate-vs-native checks above;
            # full-precision runs also check the explicit formula strictly.
            if os.environ.get('MLX_ENABLE_TF32','1') == '0':
                np.testing.assert_allclose(arrays[key], np.array(expected), atol=5e-5, rtol=5e-5)
        eager = mx.grad(objective, argnums=(0,1,2))(q,k,v)
        mx.eval(eager)
        for key, value in zip(('dq','dk','dv'), eager):
            np.testing.assert_allclose(arrays[key], np.array(value), atol=1e-5, rtol=1e-5)
        del ref_grad, eager
    # The last output must have no gradient outside its causal window.
    if args.window > 0:
        isolated = mx.grad(lambda q,k,v: mx.sum(attention(q,k,v)[:, :, -1]), argnums=(1,2))(q,k,v)
        mx.eval(isolated)
        for value in isolated:
            np.testing.assert_array_equal(np.array(value[:,:,:max(0,args.length-args.window)]), 0)
        del isolated, value
    del loss, gradients
    mx.clear_cache()
    mx.reset_peak_memory()
    measured = timing(lambda:grad(q,k,v))
    return {'length':args.length, 'window':args.window, 'batch':args.batch,
            'input_scale': args.input_scale,
            'tf32': os.environ.get('MLX_ENABLE_TF32','1'),
            'nax_tile':os.environ.get('MLX_SDPA_VJP_FLASH64_TILE','32'),
            'flash64':os.environ.get('MLX_SDPA_VJP_FLASH64','0'),
            'timing':measured, 'peak_mib':mx.get_peak_memory()/1024**2,
            'max_abs_error_vs_tiled':errors, 'max_abs_error_vs_explicit':reference_errors}


def suite(nax=False):
    cases = [(1,9,1,.3),(1,33,7,.3),(2,65,-1,.3),(1,129,256,.3),
             (2,257,63,.3),(1,129,17,1.),(1,129,-1,3.),
             (2,1024,256,.3),(2,4096,256,.3),(2,4096,-1,.3)]
    with tempfile.TemporaryDirectory(prefix='mlx-flash64-') as temp:
        for batch, length, window, input_scale in cases:
            for enabled in (0,2 if nax else 1):
                command = [sys.executable, __file__, '--batch',str(batch),
                           '--length',str(length),'--window',str(window),
                           '--input-scale',str(input_scale),
                           '--reference',f'{temp}/{batch}-{length}-{window}-{input_scale}.npz']
                env = {**os.environ, 'MLX_SDPA_VJP_FLASH64':str(enabled)}
                completed = subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
                if completed.returncode:
                    raise RuntimeError(f'{command}:\n{completed.stderr}')
                print(completed.stdout.strip(),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite',action='store_true')
    parser.add_argument('--nax',action='store_true')
    parser.add_argument('--nax-grid',action='store_true')
    parser.add_argument('--diagnose',action='store_true')
    parser.add_argument('--model',action='store_true')
    parser.add_argument('--model-suite',action='store_true')
    parser.add_argument('--diagnostic-model',action='store_true')
    parser.add_argument('--sensitivity-suite',action='store_true')
    parser.add_argument('--confirm-performance',action='store_true')
    parser.add_argument('--guards',action='store_true')
    parser.add_argument('--guard-tf32',action='store_true')
    parser.add_argument('--batch',type=int,default=2)
    parser.add_argument('--length',type=int,default=4096)
    parser.add_argument('--window',type=int,default=256)
    parser.add_argument('--input-scale',type=float,default=.3)
    parser.add_argument('--reference')
    args = parser.parse_args()
    if args.sensitivity_suite or args.confirm_performance:
        sensitivity_suite(args.confirm_performance)
    elif args.nax_grid:
        for tile in (16,32,64):
            os.environ['MLX_SDPA_VJP_FLASH64_TILE'] = str(tile)
            suite(nax=True)
    elif args.guards or args.guard_tf32:
        print(json.dumps(guards(args.guard_tf32)),flush=True)
    elif args.model_suite:
        model_suite(args.nax,args.diagnostic_model)
    elif args.model:
        print(json.dumps(model_check(args)),flush=True)
    elif args.diagnose:
        print(json.dumps(diagnose()),flush=True)
    elif args.suite:
        suite(args.nax)
    else:
        print(json.dumps(check(args)),flush=True)
