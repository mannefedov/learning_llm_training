"""Compare upstream windowed attention with an explicit differentiable reference."""
import argparse
import math
import mlx.core as mx


def reference(q, k, v, window):
    positions = mx.arange(q.shape[2])
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & (distance < window)
    scores = (q @ k.swapaxes(-1, -2)) / math.sqrt(q.shape[-1])
    return mx.softmax(mx.where(allowed, scores, -float('inf')), axis=-1) @ v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect-backward-unavailable', action='store_true')
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    mx.random.seed(42)
    for length, window in ((16, 1), (33, 7), (33, 64), (256, 64)):
        q, k, v, upstream = [mx.random.normal((1, 2, length, 64)) for _ in range(4)]
        def fused(q, k, v):
            return mx.fast.scaled_dot_product_attention(
                q, k, v, scale=1 / 8, mask='causal',
                window_size=window, force_fused=True)
        actual, expected = fused(q, k, v), reference(q, k, v, window)
        error = mx.max(mx.abs(actual - expected)).item()
        assert mx.allclose(actual, expected, atol=0.003, rtol=0.003).item(), error
        print(f'length={length} window={window}: forward max error={error:.6g}')
        try:
            gradients = mx.grad(lambda a,b,c: mx.sum(fused(a,b,c)*upstream),
                                argnums=(0,1,2))(q,k,v)
            mx.eval(gradients)
        except (ValueError, RuntimeError) as exc:
            if args.expect_backward_unavailable and 'window support for backward attention is NYI' in str(exc):
                print('  confirmed upstream windowed backward restriction')
                continue
            raise
        if args.expect_backward_unavailable:
            raise AssertionError('Backward now works; rerun without --expect-backward-unavailable')
        expected_grads = mx.grad(lambda a,b,c: mx.sum(reference(a,b,c,window)*upstream),
                                argnums=(0,1,2))(q,k,v)
        for name, actual, expected in zip(('dQ','dK','dV'), gradients, expected_grads):
            error = mx.max(mx.abs(actual-expected)).item()
            print(f'  {name} max error={error:.6g}')
            assert mx.allclose(actual, expected, atol=0.003, rtol=0.003).item(), (name,error)


if __name__ == '__main__':
    main()
