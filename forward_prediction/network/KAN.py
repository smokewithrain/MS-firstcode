import torch
import torch.nn as nn
import math

# 轻量版KAN层实现（核心：切比雪夫基函数+自适应权重）
class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, degree=3):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.degree = degree  # 基函数阶数（阶数越高，拟合非线性越强）
        
        # 切比雪夫基函数的系数权重（核心参数）
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim, degree + 1) / math.sqrt(in_dim))
        # 可选：添加层归一化，提升训练稳定性
        self.norm = nn.LayerNorm(in_dim)

    def chebyshev_polynomial(self, x):
        """计算切比雪夫多项式（一阶到degree阶）"""
        # 先将x归一化到[-1, 1]（切比雪夫多项式的有效区间）
        x = self.norm(x)  # 层归一化到均值0，方差1
        x = torch.tanh(x)  # 进一步压缩到[-1, 1]
        
        # 初始化切比雪夫多项式：T0=1, T1=x
        cheb = [torch.ones_like(x), x]
        # 递推计算更高阶：Tn = 2x*Tn-1 - Tn-2
        for i in range(2, self.degree + 1):
            cheb.append(2 * x * cheb[i-1] - cheb[i-2])
        # 拼接所有阶数的结果：[batch, in_dim, degree+1]
        return torch.stack(cheb, dim=-1)

    def forward(self, x):
        # x: [batch_size, in_dim]
        batch_size = x.shape[0]
        # 计算切比雪夫基函数：[batch, in_dim, degree+1]
        cheb_x = self.chebyshev_polynomial(x)
        # 加权求和：[out_dim, in_dim, degree+1] @ [batch, in_dim, degree+1] → [batch, out_dim]
        out = torch.einsum('oid,bid->bo', self.weight, cheb_x)
        return out

# 基于KAN的正向预测网络（替换原有MLP）
class forward_network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            # 替换Linear+ReLU为KANLayer，保持维度一致
            KANLayer(7, 128, degree=3),  # 输入7维，输出128维，阶数3（平衡拟合能力和计算量）
            KANLayer(128, 256, degree=3),
            KANLayer(256, 512, degree=3),
            KANLayer(512, 1024, degree=3),
            KANLayer(1024, 1001, degree=3),
            nn.Sigmoid()  # 保持Sigmoid，将输出归一化到[0,1]（和原模型一致）
        )

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == "__main__":
    batch_size = 32

    # 初始化KAN版正向预测网络
    fnetwork = forward_network()
    # 生成测试输入（7维几何参数）
    x = torch.randn(size=(batch_size, 7))
    # 前向传播
    output = fnetwork(x)
    # 输出形状：[32, 1001]（和原MLP模型完全一致）
    print(f"输入形状：{x.shape}")
    print(f"输出形状：{output.shape}")
    print(f"输出值范围：[{output.min().item():.4f}, {output.max().item():.4f}]")