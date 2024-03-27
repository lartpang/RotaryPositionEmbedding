# RotaryPositionEmbedding

## RoPE1D

```python
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)

rope_1d = RoPE1D(dim=512).cuda()
x = torch.randn(3, 64 * 64, 512, dtype=torch.float32, device="cuda")

start = time.perf_counter()
for _ in range(100):
    x1 = rope_1d.forward_v1(x)
torch.cuda.synchronize()
print(f"forward_v1 ({(time.perf_counter() - start) / 100}s): output mean: {x1.mean()}")

start = time.perf_counter()
for _ in range(100):
    x2 = rope_1d.forward_v2(x)
torch.cuda.synchronize()
print(f"forward_v2 ({(time.perf_counter() - start) / 100}s): output mean: {x2.mean()}")

start = time.perf_counter()
for _ in range(100):
    x3 = rope_1d.forward_v3(x)
torch.cuda.synchronize()
print(f"forward_v3 ({(time.perf_counter() - start) / 100}s): output mean: {x3.mean()}")

start = time.perf_counter()
for _ in range(100):
    x4 = rope_1d.forward_v4(x)
torch.cuda.synchronize()
print(f"forward_v4 ({(time.perf_counter() - start) / 100}s): output mean: {x4.mean()}")

start = time.perf_counter()
for _ in range(100):
    x_llama = rope_1d.forward_llama(x)
torch.cuda.synchronize()
print(f"forward_llama ({(time.perf_counter() - start) / 100}s): output mean: {x_llama.mean()}")

start = time.perf_counter()
for _ in range(100):
    x_palm = rope_1d.forward_palm(x)
torch.cuda.synchronize()
print(f"forward_palm ({(time.perf_counter() - start) / 100}s): output mean: {x_palm.mean()}")

'''
forward_v1 (0.0007928510010242462s): output mean: 0.00037511205300688744
forward_v2 (0.0005496779992245137s): output mean: 0.00037511205300688744
forward_v3 (0.0005035390006378293s): output mean: 0.00037511205300688744
forward_v4 (0.0019484849995933472s): output mean: 0.0003751121403183788
forward_llama (0.0001582980016246438s): output mean: 0.00037511205300688744
forward_palm (0.004226168000604958s): output mean: 0.00037511205300688744
'''
```

## RoPE2D

```python
rope_2d = RoPE2D(dim=512).cuda()
x = torch.randn(3, 64, 64, 512, dtype=torch.float32, device="cuda")

start = time.perf_counter()
for _ in range(100):
    x1 = rope_2d.forward_v1(x)
torch.cuda.synchronize()
print(f"forward_v1 ({(time.perf_counter() - start) / 100}s): output mean: {x1.mean()}")

start = time.perf_counter()
for _ in range(100):
    x2 = rope_2d.forward_v2(x)
torch.cuda.synchronize()
print(f"forward_v2 ({(time.perf_counter() - start) / 200}s): output mean: {x2.mean()}")

"""
forward_v1 (0.00029631400015205144s): output mean: 0.00021492868836503476
forward_v2 (0.00013582399929873646s): output mean: 0.00021492868836503476
"""
```

## Reference

- https://blog.csdn.net/PennyYu123/article/details/131717323
