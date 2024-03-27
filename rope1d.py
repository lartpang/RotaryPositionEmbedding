import math
from typing import Tuple

import torch
import torch.nn as nn
import time

import tqdm
from einops import rearrange


class RoPE1D(nn.Module):
    def __init__(self, dim, base: float = 10000):
        super().__init__()
        assert dim % 2 == 0, f"dim={dim} must be an even number!"
        # self.register_buffer(name="theta", tensor=base ** (-2 * torch.arange(dim // 2) / dim).unsqueeze(0))
        # -2 * torch.arange(dim // 2) == torch.arange(0, dim, 2)[: (dim // 2)]
        self.register_buffer(name="theta", tensor=1 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim)))

    def forward_v1(self, x):
        B, L, D = x.shape
        token_idx = torch.arange(L, dtype=x.dtype, device=x.device)
        theta = torch.outer(token_idx, self.theta)  # L,D//2

        cos_pos = torch.stack([theta, theta], dim=-1).reshape(L, D).cos()  # theta0,theta0,theta1,theta1,...
        # sin(-t)=-sin(t), cos(-t)=cos(t)
        sin_pos = torch.stack([-theta, theta], dim=-1).reshape(L, D).sin()  # -theta0,theta0,-theta1,theta1,...

        x0, x1 = x.reshape(B, L, D // 2, 2).chunk(2, dim=-1)  # B,L,1 x1,x0,x3,x2
        x_ = torch.cat([x1, x0], dim=-1).reshape(B, L, D)
        return x * cos_pos + x_ * sin_pos

    def forward_v2(self, x):
        B, L, D = x.shape
        token_idx = torch.arange(L, dtype=x.dtype, device=x.device)
        theta = torch.outer(token_idx, self.theta)  # L,D//2

        # theta0,theta0,theta1,theta1,...
        theta = theta.unsqueeze(-1).repeat(1, 1, 2).reshape(L, D)  # L,D

        x0, x1 = x.reshape(B, L, D // 2, 2).chunk(2, dim=-1)  # B,L,1 x1,x0,x3,x2
        x_ = torch.cat([-x1, x0], dim=-1).reshape(B, L, D)
        return x * theta.cos() + x_ * theta.sin()

    def forward_v3(self, x):
        B, L, D = x.shape
        token_idx = torch.arange(L, dtype=x.dtype, device=x.device)
        theta = torch.outer(token_idx, self.theta)  # L,D//2

        # sin(-t)=-sin(t), cos(-t)=cos(t)
        theta = torch.stack([-theta, theta], dim=-1).reshape(L, D)

        x0, x1 = x.reshape(B, L, D // 2, 2).chunk(2, dim=-1)  # B,L,1 x1,x0,x3,x2
        x_ = torch.cat([x1, x0], dim=-1).reshape(B, L, D)
        return x * theta.cos() + x_ * theta.sin()

    def forward_llama(self, x):
        B, L, D = x.shape

        token_idx = torch.arange(L, dtype=x.dtype, device=x.device)
        theta = torch.outer(token_idx, self.theta)  # L,D//2

        # 1*cos(theta)+1*sin(theta)j
        freqs_cis = torch.polar(torch.ones_like(theta), theta)
        freqs_cis = freqs_cis.view(1, L, D // 2)  # 1,L,D//2 cos(theta0)+sin(theta0)j

        # the input is expected to have the last dimension of size 2. => [x, y]->(x+yj)
        # for torch.float64 and torch.float32
        x_ = torch.view_as_complex(x.float().reshape(B, L, D // 2, 2))  # B,L,D//2 x0+x1j
        # (x+yj)(a+bj)=xa-yb+(xb+ya)j=>(xcost-ysint)+(xsint+ycost)j
        x_ = x_ * freqs_cis  # 执行position-wise复数乘积运算
        # (a+bj)->[a, b] x0cost0-x1sint0,x0sint0+x1cost0,x2cost1-x3sint1,x2sint1+x3cost1
        x_ = torch.view_as_real(x_).flatten(3)  # B,L,D
        return x_.type_as(x)

    def forward_palm(self, x):
        B, L, D = x.shape
        token_idx = torch.arange(L, dtype=x.dtype, device=x.device)
        theta = torch.einsum("i, j -> i j", token_idx, self.theta)  # L,D//2

        # Use the interleaved form that differs from the original form.
        # theta0,theta0,theta1,theta1,...
        theta = theta.unsqueeze(-1).repeat(1, 1, 2).reshape(L, D)  # L,D (theta0,theta0,theta1,theta1,...

        x1, x2 = rearrange(x, "b l (d j) -> b l d j", j=2).chunk(2, dim=-1)  # B,L,D//2,1 x0,x2,... x1,x3,...
        x_ = torch.cat((-x2, x1), dim=-1).reshape(B, L, D)
        return x * theta.cos() + x_ * theta.sin()


if __name__ == '__main__':
    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)

    rope_1d = RoPE1D(dim=512).cuda()
    x = torch.randn(3, 64 * 64, 512, dtype=torch.float32, device="cuda")

    start = time.perf_counter()
    for _ in tqdm.tqdm(range(100), total=100, ncols=78):
        x1 = rope_1d.forward_v1(x)
    torch.cuda.synchronize()
    print(f"forward_v1: {(time.perf_counter() - start) / 100}s: {x1.mean()}")

    start = time.perf_counter()
    for _ in tqdm.tqdm(range(100), total=100, ncols=78):
        x2 = rope_1d.forward_v2(x)
    torch.cuda.synchronize()
    print(f"forward_v2: {(time.perf_counter() - start) / 100}s: {x2.mean()}")

    start = time.perf_counter()
    for _ in tqdm.tqdm(range(100), total=100, ncols=78):
        x3 = rope_1d.forward_v3(x)
    torch.cuda.synchronize()
    print(f"forward_v3: {(time.perf_counter() - start) / 100}s: {x3.mean()}")

    start = time.perf_counter()
    for _ in tqdm.tqdm(range(100), total=100, ncols=78):
        x4 = rope_1d.forward_llama(x)
    torch.cuda.synchronize()
    print(f"forward_llama: {(time.perf_counter() - start) / 100}s: {x4.mean()}")

    start = time.perf_counter()
    for _ in tqdm.tqdm(range(100), total=100, ncols=78):
        x5 = rope_1d.forward_palm(x)
    torch.cuda.synchronize()
    print(f"forward_palm: {(time.perf_counter() - start) / 100}s: {x5.mean()}")
