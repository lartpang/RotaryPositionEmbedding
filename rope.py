import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import tqdm
from einops import rearrange, repeat


class RoPE1D(nn.Module):
    # https://kexue.fm/archives/8265
    def __init__(self, dim, base: float = 10000):
        super().__init__()
        assert dim % 2 == 0, f"dim={dim} must be an even number!"

        # self.register_buffer(name="theta", tensor=base ** (-2 * torch.arange(dim // 2) / dim).unsqueeze(0))
        # -2 * torch.arange(dim // 2) == torch.arange(0, dim, 2)[: (dim // 2)]
        self.register_buffer("theta", base ** (-torch.arange(0, dim, 2)[: (dim // 2)] / dim))

    def forward(self, x):
        return self.forward_llama(x)

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

    def forward_v4(self, x):
        # Directly construct the rotation matrix R and use einsum to calculate the results
        B, L, D = x.shape
        token_idx = torch.arange(L, dtype=x.dtype, device=x.device)
        theta = torch.outer(token_idx, self.theta)  # L,D//2
        cos_pos = theta.cos()
        sin_pos = theta.sin()
        theta = torch.stack([cos_pos, -sin_pos, sin_pos, cos_pos], dim=-1).reshape(L, D // 2, 2, 2)
        x = x.reshape(B, L, D // 2, 2)  # [x0, x1], [x2, x3], ...
        x = torch.einsum("ldxy, bldy -> bldx", theta, x)
        return x.flatten(2)

    def forward_llama(self, x):
        B, L, D = x.shape

        token_idx = torch.arange(L, dtype=x.dtype, device=x.device)
        theta = torch.outer(token_idx, self.theta)  # L,D//2

        # 1*cos(theta)+1*sin(theta)j
        theta = torch.polar(torch.ones_like(theta), theta)
        theta = theta.view(1, L, D // 2)  # 1,L,D//2 cos(theta0)+sin(theta0)j

        # the input is expected to have the last dimension of size 2. => [x, y]->(x+yj)
        # for torch.float64 and torch.float32
        x_ = torch.view_as_complex(x.float().reshape(B, L, D // 2, 2))  # B,L,D//2 x0+x1j
        # (x+yj)(a+bj)=xa-yb+(xb+ya)j=>(xcost-ysint)+(xsint+ycost)j
        x_ = x_ * theta  # 执行position-wise复数乘积运算
        # (a+bj)->[a, b] x0cost0-x1sint0,x0sint0+x1cost0,x2cost1-x3sint1,x2sint1+x3cost1
        x_ = torch.view_as_real(x_).flatten(2)  # B,L,D
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


class RoPE2D(nn.Module):
    # https://kexue.fm/archives/8397
    def __init__(self, dim, base: float = 10000):
        super().__init__()
        assert dim % 4 == 0, f"dim={dim} must be divisible by 4!"

        valid_dim = dim // 2
        self.register_buffer("theta", base ** (-torch.arange(0, valid_dim, 2)[: valid_dim // 2] / valid_dim))

    def forward(self, image):
        return self.forward_v2(image)

    def forward_v1(self, image):
        B, H, W, D = image.shape

        x_idx = torch.arange(H, dtype=image.dtype, device=image.device)
        x_theta = torch.outer(x_idx, self.theta).reshape(1, H, 1, D // 4)
        y_idx = torch.arange(W, dtype=image.dtype, device=image.device)
        y_theta = torch.outer(y_idx, self.theta).reshape(1, 1, W, D // 4)

        x_theta = torch.polar(torch.ones_like(x_theta), x_theta).repeat(1, 1, W, 1)  # 1,H,W,D//4 cos(t0)+sin(t0)j
        y_theta = torch.polar(torch.ones_like(y_theta), y_theta).repeat(1, H, 1, 1)  # 1,H,W,D//4 cos(t0)+sin(t0)j

        image = rearrange(image, "b h w (d xy ab) -> b h w d xy ab", xy=2, ab=2)  # xy for space, ab for channel
        x_image, y_image = image.unbind(-2)  # B,H,W,D//4,2

        x_image_ = torch.view_as_complex(x_image.float())  # B,H,W,D//4 a+bj
        x_image_ = x_image_ * x_theta  # B,H,W,D//4,2
        x_image_ = torch.view_as_real(x_image_)  # B,H,W,D//4,2

        y_image_ = torch.view_as_complex(y_image.float())  # B,H,W,D//4 a+bj
        y_image_ = y_image_ * y_theta  # B,H,W,D//4,2
        y_image_ = torch.view_as_real(y_image_)  # B,H,W,D//4,2

        image_ = torch.stack([x_image_, y_image_], dim=-2).flatten(-3)
        return image_.type_as(image)

    def forward_v2(self, image):
        B, H, W, D = image.shape

        x_idx = torch.arange(H, dtype=image.dtype, device=image.device)
        x_theta = torch.outer(x_idx, self.theta).reshape(1, H, 1, D // 4)
        y_idx = torch.arange(W, dtype=image.dtype, device=image.device)
        y_theta = torch.outer(y_idx, self.theta).reshape(1, 1, W, D // 4)

        x_theta = torch.polar(torch.ones_like(x_theta), x_theta).repeat(1, 1, W, 1)  # 1,H,W,D//4 cos(t0)+sin(t0)j
        y_theta = torch.polar(torch.ones_like(y_theta), y_theta).repeat(1, H, 1, 1)  # 1,H,W,D//4 cos(t0)+sin(t0)j
        xy_theta = torch.stack([x_theta, y_theta], dim=-1)

        image = rearrange(image, "b h w (d xy ab) -> b h w d xy ab", xy=2, ab=2)  # xy for space, ab for channel

        # the input is expected to have the last dimension of size 2. => [x, y]->(x+yj)
        image_ = torch.view_as_complex(image.float())  # B,H,W,D//4,2 xy,a+bj
        # 执行position-wise复数乘积运算 (x+yj)(a+bj)=xa-yb+(xb+ya)j=>(xcost-ysint)+(xsint+ycost)j
        image_ = image_ * xy_theta  # B,H,W,D//4,2
        # (a+bj)->[a,b] x0cost0-x1sint0,x0sint0+x1cost0,x2cost1-x3sint1,x2sint1+x3cost1
        image_ = torch.view_as_real(image_).flatten(-3)  # B,H,W,D//4,2,2 -> B,H,W,D
        return image_.type_as(image)


if __name__ == "__main__":
    torch.manual_seed(1024)
    torch.cuda.manual_seed(1024)

    rope_1d = RoPE1D(dim=512).cuda()
    x = torch.randn(3, 64 * 64, 512, dtype=torch.float32, device="cuda")

    start = time.perf_counter()
    for _ in range(100):
        x1 = rope_1d.forward_v1(x)
    torch.cuda.synchronize()
    print(f"forward_v1 ({(time.perf_counter() - start) / 100}s): output ({x1.shape}) mean: {x1.mean()}")

    start = time.perf_counter()
    for _ in range(100):
        x2 = rope_1d.forward_v2(x)
    torch.cuda.synchronize()
    print(f"forward_v2 ({(time.perf_counter() - start) / 100}s): output ({x2.shape}) mean: {x2.mean()}")

    start = time.perf_counter()
    for _ in range(100):
        x3 = rope_1d.forward_v3(x)
    torch.cuda.synchronize()
    print(f"forward_v3 ({(time.perf_counter() - start) / 100}s): output ({x3.shape}) mean: {x3.mean()}")

    start = time.perf_counter()
    for _ in range(100):
        x4 = rope_1d.forward_v4(x)
    torch.cuda.synchronize()
    print(f"forward_v4 ({(time.perf_counter() - start) / 100}s): output ({x4.shape}) mean: {x4.mean()}")

    start = time.perf_counter()
    for _ in range(100):
        x_llama = rope_1d.forward_llama(x)
    torch.cuda.synchronize()
    print(f"forward_llama ({(time.perf_counter() - start) / 100}s): output ({x_llama.shape}) mean: {x_llama.mean()}")

    start = time.perf_counter()
    for _ in range(100):
        x_palm = rope_1d.forward_palm(x)
    torch.cuda.synchronize()
    print(f"forward_palm ({(time.perf_counter() - start) / 100}s): output ({x_palm.shape}) mean: {x_palm.mean()}")

    """
    forward_v1 (0.0007928510010242462s): output mean: 0.00037511205300688744
    forward_v2 (0.0005496779992245137s): output mean: 0.00037511205300688744
    forward_v3 (0.0005035390006378293s): output mean: 0.00037511205300688744
    forward_v4 (0.0019484849995933472s): output mean: 0.0003751121403183788
    forward_llama (0.0001582980016246438s): output mean: 0.00037511205300688744
    forward_palm (0.004226168000604958s): output mean: 0.00037511205300688744
    """

    rope_2d = RoPE2D(dim=512).cuda()
    x = torch.randn(3, 64, 64, 512, dtype=torch.float32, device="cuda")

    start = time.perf_counter()
    for _ in range(100):
        x1 = rope_2d.forward_v1(x)
    torch.cuda.synchronize()
    print(f"forward_v1 ({(time.perf_counter() - start) / 100}s): output ({x1.shape}) mean: {x1.mean()}")

    start = time.perf_counter()
    for _ in range(100):
        x2 = rope_2d.forward_v2(x)
    torch.cuda.synchronize()
    print(f"forward_v2 ({(time.perf_counter() - start) / 200}s): output ({x2.shape}) mean: {x2.mean()}")

    """
    forward_v1 (0.00029631400015205144s): output mean: 0.00021492868836503476
    forward_v2 (0.00013582399929873646s): output mean: 0.00021492868836503476
    """
