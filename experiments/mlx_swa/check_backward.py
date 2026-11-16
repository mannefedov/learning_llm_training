"""Small causal-window backward regression grid for the local MLX patch."""
import math
import mlx.core as mx
import numpy as np

mx.set_default_device(mx.gpu)
mx.random.seed(7)


def reference(q, k, v, window):
    k = mx.repeat(k, q.shape[1] // k.shape[1], axis=1)
    v = mx.repeat(v, q.shape[1] // v.shape[1], axis=1)
    pos = mx.arange(q.shape[2])
    distance = pos[:, None] - pos[None, :]
    mask = distance >= 0
    if window > 0:
        mask = mask & (distance < window)
    scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(q.shape[-1])
    return mx.softmax(mx.where(mask, scores, -float('inf')), axis=-1) @ v


for dtype in (mx.float32, mx.float16, mx.bfloat16):
    for length, window in ((9, 1), (33, 7), (65, 16), (33, 64), (129, 32), (33, -1)):
        q = mx.random.normal((2, 4, length, 64)).astype(dtype)
        k, v = [mx.random.normal((2, 2, length, 64)).astype(dtype) for _ in range(2)]
        upstream = mx.random.normal(q.shape).astype(dtype)
        def loss(q, k, v):
            out = mx.fast.scaled_dot_product_attention(q,k,v,scale=1/8,
                mask='causal',window_size=window,force_fused=True)
            return mx.sum(out.astype(mx.float32)*upstream.astype(mx.float32))
        # Float32 oracle isolates low-precision kernel error from reference rounding.
        def oracle(q,k,v):
            return mx.sum(reference(q,k,v,window)*upstream.astype(mx.float32))
        expected = mx.grad(oracle,argnums=(0,1,2))(*[a.astype(mx.float32) for a in (q,k,v)])
        for compiled in (False, True):
            fn = mx.grad(loss,argnums=(0,1,2))
            if compiled:
                fn = mx.compile(fn)
            actual = fn(q,k,v)
            errors = []
            for a,e in zip(actual,expected):
                a,e = np.array(a.astype(mx.float32)),np.array(e)
                assert np.isfinite(a).all()
                atol = 0.004 if dtype == mx.float32 else (0.02 if dtype == mx.float16 else 0.15)
                np.testing.assert_allclose(a,e,atol=atol,rtol=atol)
                errors.append(float(np.max(np.abs(a-e))))
            print(dtype,length,window,'compiled',compiled,'max errors',errors,flush=True)
print('All backward comparisons passed')
