# TODO: 想把网络改的简单一些，不冗余
import torch
import torch.nn as nn
import torch.nn.functional as F



# 正向预测
# 输入7个几何参数
# 输出1001个采样点的S11 dB

# class SE1D(nn.Module):
#     """残差连接的通道注意力机制"""
#     def __init__(self, c, r=2):
#         super().__init__()
#         self.fc = nn.Sequential(    # 瓶颈全连接
#             nn.Linear(c, c // r),    # // 向下取整除法
#             nn.ReLU(inplace=True),   # inplace=True, 在数据本身内存上修改值, 节省内存
#             nn.Linear(c // r, c),
#             nn.Sigmoid()            # 感觉有点像概率, 谁更重要
#         )
#         self.relu = nn.ReLU()
#     def forward(self, x):
#         """
#         Args:
#             x: (B, C, L)
#         """
#         residual = x 
#         z = x.mean(dim=-1)   # 压缩空间维度 (B, C)   每个通道提取出一个值表示
#         w = self.fc(z).unsqueeze(-1)  # (B, C, 1)  激活, 增强有用通道, 抑制无用通道
#         out_attn = x*w 
#         out = out_attn + residual
#         # TODO: 是否对out应用relu激活？
#         out = self.relu(out)
#         return out

class up_conv_res_block(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4, is_last=False):
        super(up_conv_res_block, self).__init__()

        self.is_last = is_last
        self.up_layer =  nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)  # L 16 -> 64
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        # 1 x 1 卷积 降低维度
        self.match_channels = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.up_layer(x)                  # 上采样
        residual = self.match_channels(x)     # 对上采样后的特征进行通道映射

        out = self.block(x)                   # 对上采样后的特征进行特征提取
        out += residual                       # 残差融合
        if not self.is_last:
            out = self.relu(out)
        return out 


class forward_network(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器, 学习全局物理条件, 如峰数、共振强弱等
        self.encoder = nn.Sequential(
            nn.Linear(7, 512),
            nn.ReLU(),
            
            nn.Linear(512, 2048),
            nn.ReLU(),
        )

        # 曲线逐渐生成的细节, 如峰的连续性、峰宽、局部振荡
        # self.se1 = SE1D(c=128, r=2)
        self.gen1 = up_conv_res_block(in_channels=128, out_channels=64, scale_factor=4)
        self.gen2 = up_conv_res_block(in_channels=64, out_channels=32, scale_factor=4)
        self.gen3 = up_conv_res_block(in_channels=32, out_channels=1, scale_factor=4, is_last=True)

        self.relu = nn.ReLU()
        

    def forward(self, x):
        B = x.size(0)
        features = self.encoder(x)  # (B, 2048) 学习全局物理条件

        latent = features  # (B, 2048, 1)  2048=32*64
        latent = latent.view(B, 128, 16)  # (B, 128, 16) (B, C, L)

        # latent = self.se1(latent)
        latent = self.gen1(latent)
        # latent = self.se2(latent)
        latent = self.gen2(latent)
        # latent = self.se3(latent)
        latent = self.gen3(latent)
        


        S11 = -self.relu(latent)  # (B, 1, 1024)

        # 进一步生成最终曲线
        S11 = S11[..., :1001].squeeze(1)  # 用...补齐前面的维度
        return  S11


if __name__ == "__main__":
    batch_size = 32

    fnetwork = forward_network()
    x = torch.randn(size=(batch_size, 7))
    S11_output = fnetwork(x)
    print(S11_output)
    print(S11_output.shape)


