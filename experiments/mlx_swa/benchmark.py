"""Small synchronized attention-only forward/backward timing comparison."""
import statistics
import time
import mlx.core as mx
from probe import reference

mx.set_default_device(mx.gpu)
mx.random.seed(42)
for length in (256, 1024):
    q,k,v,u = [mx.random.normal((1,2,length,64)) for _ in range(4)]
    for kind in ('reference', 'fused'):
        def loss(q,k,v):
            if kind == 'reference':
                out = reference(q,k,v,64)
            else:
                out = mx.fast.scaled_dot_product_attention(q,k,v,scale=1/8,
                    mask='causal',window_size=64,force_fused=True)
            return mx.sum(out*u)
        fn = mx.value_and_grad(loss,argnums=(0,1,2))
        for _ in range(3):
            mx.eval(fn(q,k,v))
        times=[]
        for _ in range(10):
            start=time.perf_counter()
            mx.eval(fn(q,k,v))
            times.append((time.perf_counter()-start)*1000)
        print(length,kind,'median ms',round(statistics.median(times),3),flush=True)
