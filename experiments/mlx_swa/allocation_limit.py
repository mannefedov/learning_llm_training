"""Check the unavoidable dense score buffer against Metal's per-buffer limit."""
import json
import math
import mlx.core as mx
mx.set_default_device(mx.gpu)
length=math.isqrt(mx.device_info()["max_buffer_length"] // (4 * 4)) + 1
q=mx.zeros((1,4,length,64))
mx.eval(q)
try:
    mx.eval(q @ q.swapaxes(-1,-2))
    result={'status':'unexpected_success'}
except RuntimeError as e:
    result={'status':'allocation_error','error':str(e)}
result.update(length=length,score_gib=4*length*length*4/1024**3,
              max_buffer_gib=mx.device_info()['max_buffer_length']/1024**3)
print(json.dumps(result))
