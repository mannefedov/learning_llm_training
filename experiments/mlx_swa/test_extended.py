"""Long-context/layout checks for the experimental native causal-window VJP."""
import math
import unittest
import mlx.core as mx
import numpy as np


def reference(q,k,v,w):
    k=mx.repeat(k,q.shape[1]//k.shape[1],axis=1)
    v=mx.repeat(v,q.shape[1]//v.shape[1],axis=1)
    pos=mx.arange(q.shape[2]); dist=pos[:,None]-pos[None,:]
    score=(q@k.swapaxes(-1,-2))*q.shape[-1]**-0.5
    return mx.softmax(mx.where((dist>=0)&(dist<w),score,-float('inf')),axis=-1)@v


def fused(q,k,v,w):
    return mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-0.5,
        mask='causal',window_size=w,force_fused=True)


class ExtendedTests(unittest.TestCase):
    def setUp(self):
        mx.set_default_device(mx.gpu); mx.random.seed(17); mx.set_cache_limit(256*1024**2)

    def test_long_context_and_noncontiguous_grouped_heads(self):
        for t,w,d in ((257,63,64),(1025,257,128),(4096,256,64)):
            with self.subTest(length=t,window=w,dim=d):
                q=mx.random.normal((1,t,4,d)).transpose(0,2,1,3)
                k,v=[mx.random.normal((1,t,2,d)).transpose(0,2,1,3) for _ in range(2)]
                u=mx.random.normal(q.shape)
                actual=fused(q,k,v,w); expected=reference(q,k,v,w)
                np.testing.assert_allclose(np.array(actual),np.array(expected),atol=.005,rtol=.005)
                gs=[]
                for f in (fused,reference):
                    gs.append(mx.grad(lambda a,b,c:mx.sum(f(a,b,c,w)*u),argnums=(0,1,2))(q,k,v))
                for a,e in zip(*gs):
                    np.testing.assert_allclose(np.array(a),np.array(e),atol=.008,rtol=.008)
                print('long/layout passed',t,w,d,flush=True)

    def test_compiled_distinct_windows(self):
        q,k,v=[mx.random.normal((1,2,65,64)) for _ in range(3)]
        def loss(a,b,c):
            return mx.sum(fused(a,b,c,7)**2)+mx.sum(fused(a,b,c,31)**2)
        fn=mx.grad(loss,argnums=(0,1,2)); compiled=mx.compile(fn)
        for a,e in zip(compiled(q,k,v),fn(q,k,v)):
            np.testing.assert_allclose(np.array(a),np.array(e),atol=1e-5,rtol=1e-5)

    def test_no_gradient_outside_window(self):
        q,k,v=[mx.random.normal((1,2,129,64)) for _ in range(3)]
        for w in (7,32):
            grads=mx.grad(lambda a,b,c:mx.sum(fused(a,b,c,w)[:,:,80]),argnums=(0,1,2))(q,k,v)
            for g in grads[1:]:
                self.assertEqual(mx.max(mx.abs(g[:,:,:81-w])).item(),0)
                self.assertEqual(mx.max(mx.abs(g[:,:,81:])).item(),0)

    def test_unsupported_shapes_reject_forced_backward(self):
        for length,keys,causal in ((7,7,True),(16,33,True),(16,16,False)):
            q=mx.ones((1,2,length,64)); k=v=mx.ones((1,2,keys,64))
            with self.assertRaisesRegex(ValueError,'windowed backward requires'):
                fn=lambda a:mx.sum(mx.fast.scaled_dot_product_attention(a,k,v,
                    scale=1/8,mask='causal' if causal else None,window_size=7,force_fused=True))
                mx.eval(mx.grad(fn)(q))


if __name__=='__main__': unittest.main()
