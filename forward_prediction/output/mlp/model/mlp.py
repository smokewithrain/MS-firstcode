import torch
import torch.nn as nn 

# 正向预测
# 输入7个几何参数
# 输出1001个采样点的S11值（归一化到[0,1])
class forward_network(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),

            nn.Linear(128, 512),
            nn.ReLU(),    

            nn.Linear(512, 1001)
            # nn.ReLU(),
            
            # nn.Linear(512, 1024),
            # # nn.ReLU(),

            # nn.Linear(1024, 1001),
        )

    def forward(self, x):
        output = self.model(x)
        return output


if __name__ == "__main__":
    batch_size = 32

    fnetwork = forward_network()
    x = torch.randn(size=(batch_size, 7))
    output = fnetwork(x)
    print(output.shape)

    