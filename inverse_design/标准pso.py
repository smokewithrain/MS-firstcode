""" 
逆向设计模型

输入一条1001个采样点的S11曲线
输出7个几何参数

粒子群算法
种群数量100, 随机产生符合约束条件的初始粒子群
适应度函数, 正向预测模型替代CST输出仿真结果, 适应度计算为预测出的S11曲线和目标曲线之间的-MSE, 适应度越大越好

效果: 对于部分曲线效果还行
"""

import torch
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from up_conv import forward_network

from setting import *

plt.rcParams['font.family'] = 'Times New Roman'  # IEEE常用字体
plt.rcParams['font.size'] = 10  # 字体大小10pt
plt.rcParams['axes.linewidth'] = 0.8  # 坐标轴线条宽度
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['figure.figsize'] = [3.5, 2.5] 

class PSO_inverse_design:
    """针对指定的S11_target目标曲线, 逆向设计出对应的超表面几何参数"""
    def __init__(self, 
                S11_target,
                device=torch.device("cpu"),
                num_particles=100, 
                max_iterations=200, 
                w_min=0.4, w_max=0.9,
                c1_min=0.5, c1_max=2.5,  
                c2_min=0.5, c2_max=2.5,  
                title=None, plot=True):
        """初始化"""

        # PSO超参数
        self.num_particles = num_particles    # 种群数量
        self.max_iterations = max_iterations  # 最大迭代次数
        self.w_min = w_min                    # 惯性权重最小值
        self.w_max = w_max                    # 惯性权重最大值
        self.c1_min = c1_min                  # 自身认知权重最小值、最大值
        self.c1_max = c1_max
        self.c2_min = c2_min                  # 全体认知权重最小值、最大值
        self.c2_max = c2_max
        self.title = title
        self.plot = plot

        # 解的约束            
        self.dim_ranges = [(4.7, 6.4), (4.7, 6.4), (11.7, 14.9), [(0.1, 0.4), (1.5, 1.8)], (0.1, 0.4), (0.2, 1.1), (0.6, 1.8)]  # 每个维度的范围
        self.particle_dim = len(self.dim_ranges)  # 粒子维度（7个几何参数）

        # 种群以及状态记录
        self.S11_target = S11_target  # S11线性值
        self.particles = None                       # 粒子种群
        self.velocity = None
        self.pbest = None
        self.pbest_fitness = None
        self.gbest = None
        self.gbest_fitness = None
        self.fitness_progress = []

        # 深度学习设置
        self.device = device
        self.forward_network = self.load_forward_model()

    def load_forward_model(self):
        """加载正向预测网络"""

        fnetwork = forward_network().to(self.device)
        model_path = base_path / 'best_model.pth'
        fnetwork.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True), strict=False)
        fnetwork.eval()
        return fnetwork

    def forward_predict(self, param_np):
        """
        正向网络预测
        
        :param param: 7个几何参数 np.array
        :return output_S11_dB: 输出S11曲线 np.array
        """
        # 输入深度学习网络前，几何参数要先归一化
        single_input = False           # 如果只输入一个样本, 扩充batch的维度和输入多个样本对齐
        if param_np.ndim == 1:
            param_np = param_np[None, :]  # (7, ) -> (1, 7)
            single_input = True

        param = torch.from_numpy(param_np).float().to(self.device)

        param_min = torch.tensor([4.7, 4.7, 11.7, 0.1, 0.1, 0.2, 0.6], dtype=torch.float32)
        param_max = torch.tensor([6.4, 6.4, 14.9, 1.8, 0.4, 1.1, 1.8], dtype=torch.float32)
        range_ = param_max - param_min
        param = (param - param_min) / range_ 


        with torch.no_grad():
            output_S11_dB = self.forward_network(param)
        
        output = output_S11_dB.cpu().numpy()
        if single_input:
            output = output[0]
        return output
    
    def cal_fitness_batch(self, particles):
        """
        批量计算粒子群的适应度
        
        Args:
            particles: (N, 7)
            fitness: (N, )
        """

        output_S11_dB = self.forward_predict(particles)    # (N, 32)   
        output_S11 = 10 ** (output_S11_dB / 20)
        # 对于S11 线性值，算MAE
        mae = np.mean(np.abs(self.S11_target- output_S11), axis=1)
        
        return mae  # 适应度（越小越好）

    @staticmethod
    def check_dimensions(particle):
        """检查维度约束"""
        return (particle[0] + 0.5 * particle[3] + particle[5] <= 7.75) and (particle[1] + 0.5 * particle[3] + particle[6] <= 7.75)

    def random_particle(self):
        """随机生成一个符合维度+区间约束的粒子""" 
        while True:
            p = np.zeros(self.particle_dim)
            for i, dim_range in enumerate(self.dim_ranges): # 对每个维度
                if isinstance(dim_range, list):             # 若该维度不只一个区间
                    dim_range = random.choice(dim_range)    # 随机选一个区间 
                
                values = np.arange(dim_range[0], dim_range[1]+1e-8, 0.1) # 按照几何参数范围，按0.1精度抽样
                p[i] = np.random.choice(values)
            if self.check_dimensions(p):               # 确保满足维度约束 
                return p 
    
    def clamp_particle(self, particle):
        """约束单个粒子的每个维度必须在区间内, 否则截断"""
        for i, dim_range in enumerate(self.dim_ranges):
            value = particle[i]

            if isinstance(dim_range, tuple):
                value = np.clip(value, dim_range[0], dim_range[1])
            else:                                         # 该维度不只一个区间
                in_segment = False                        # 遍历所有区间
                for segment in dim_range:
                    if segment[0] <= value <= segment[1]: # 判断是否在某一区间内
                        value = np.clip(value, segment[0], segment[1])
                        in_segment = True 
                        break
                if not in_segment:                        # 不在任一区间内，截断到最近区间
                    distances = []                        # 计算到各区间的距离
                    for segment in dim_range: 
                        if value < segment[0]:                      # 在区间左边 
                            distances.append(abs(segment[0]-value))
                        else:                                      # 在区间右边
                            distances.append(abs(segment[1]-value))
                    segment = dim_range[np.argmin(distances)]
                    value = np.clip(value, segment[0], segment[1])
            particle[i] = np.round(value, 1)
        return particle
    
    def init_particles(self):
        """初始化整个粒子群"""
        self.particles = np.array([self.random_particle() for _ in range(self.num_particles)])          
        self.velocity = np.random.uniform(-0.1, 0.1, size=(self.num_particles, self.particle_dim))                 # 初始化速度[-0.1, 0.1)

        self.pbest = self.particles.copy()                                                                # 每个个体的历史最优位置
        self.pbest_fitness = self.cal_fitness_batch(self.pbest)

        gbest_idx = np.argmin(self.pbest_fitness)                # MAE越小越好                                         # 找到适应度最大的粒子索引
        self.gbest = self.pbest[gbest_idx].copy()                                                         # 全局历史最优位置
        self.gbest_fitness = self.pbest_fitness[gbest_idx].copy()                                       # 全局最优适应度

    def move_particles(self, iter):
        """更新粒子群"""
        # 动态调整学习因子
        # 初期，c1较大，c2较小，增强个体对自身经验的学习(强化探索)
        # 后期，c1较小，c2较大，增强对群体最优的学习(强化局部寻优)
        c1 = self.c1_max - (self.c1_max - self.c1_min) * (iter / self.max_iterations)  
        c2 = self.c2_min + (self.c2_max - self.c2_min) * (iter / self.max_iterations)
        
        # 非线性惯性权重
        # iter越大，w越小
        w = self.w_max * (1 - (iter / self.max_iterations) ** 2 ) + self.w_min * ((iter / self.max_iterations) ** 2)
        
        # 生成随机因子
        r1 = np.random.rand(self.num_particles, self.particle_dim)
        r2 = np.random.rand(self.num_particles, self.particle_dim)
        
        # 更新速度
        self.velocity = w * self.velocity + c1 * r1 * (self.pbest - self.particles) + c2 * r2 * (self.gbest - self.particles)
        
        # # 粒子每次移动要求是0.1的整数倍
        # step = 0.1
        # self.velocity = np.round(self.velocity / step) * step
        # for p_v in self.velocity:
        #     if sum(p_v == 0) == 7:       # 假如速度全0, 选一个位置更新
        #         idx = random.randint(0, 6)  # [0, 6]
        #         p_v[idx] = random.choice([-0.1, 0.1])

        # # 速度限制最大
        # v_max = 0.3 
        # self.velocity = np.clip(self.velocity, -v_max, v_max)


        # # 更新粒子位置
        # self.particles += self.velocity
        # for i in range(self.num_particles):
        #     self.particles[i] = self.clamp_particle(self.particles[i])   # 约束粒子每个维度在范围内
        #     if not self.check_dimensions(self.particles[i]):             # 检查每个粒子的几何约束，不满足则重新生成
        #         self.particles[i] = self.random_particle()



    # 综合PSO算法
    def run(self):

        self.init_particles()

        unimprove_count = 0  # 连续未改进次数
        
        for iter in range(self.max_iterations):                   # 每轮迭代
            self.move_particles(iter)                             # 更新粒子群
            
            cur_fitness = self.cal_fitness_batch(self.particles)  # 计算当代适应度
            for i in range(self.num_particles):                   # 更新个体最优适应度
                fitness = cur_fitness[i]
                
                # 更新个体最优 , 适应度mae越小越好
                if fitness < self.pbest_fitness[i]:
                    self.pbest[i] = self.particles[i].copy()
                    self.pbest_fitness[i] = fitness
            
            # 更新全局最优适应度，如果连续未改进，需要处理
            cur_gbest_idx = np.argmin(self.pbest_fitness)                                                       # 找到适应度最大的粒子索引
            cur_gbest_fitness = self.pbest_fitness[cur_gbest_idx]
            self.fitness_progress.append(cur_gbest_fitness)
            if cur_gbest_fitness < self.gbest_fitness:
                self.gbest = self.pbest[cur_gbest_idx].copy()                                                  # 全局历史最优位置
                self.gbest_fitness = cur_gbest_fitness                                                             # 全局最优适应度
                unimprove_count = 0
            else:
                unimprove_count  += 1 
            
            # if unimprove_count >= 3:                                                                               # 当连续未改进时，对适应度最差的一半粒子随机扰动
            #     print("连续未改进, 随机扰动")
            #     sorted_idx = np.argsort(-cur_fitness)                            # 默认升序，这里降序排列
            #     worse_half_idx = sorted_idx[:self.num_particles // 2]
            #     # 随机扰动
            #     rand_move = np.random.rand(len(worse_half_idx), self.particle_dim) * 0.4 - 0.2                       # [-0.2, 0.2]                  
            #     self.particles[worse_half_idx] += rand_move
            #     # 重新约束范围和精度
            #     self.particles = np.array([self.clamp_particle(particle) for particle in self.particles])

            #     # 检查每个粒子的几何约束，不满足则重新生成
            #     for i in range(self.num_particles):
            #         if not self.check_dimensions(self.particles[i]):
            #             self.particles[i] = self.random_particle()
                
            #     unimprove_count = 0

            if self.plot:
                print(f"Iteration {iter + 1}/{self.max_iterations}, Best Fitness: {self.gbest_fitness.item()}")
        if self.plot:
            self.plot_fitness_progress()
            self.plot_optimization_result()
            self.save_results()
    
    def plot_fitness_progress(self):
        """绘制适应度随迭代的变化曲线"""
        plt.plot(range(self.max_iterations), self.fitness_progress)
        plt.xlabel('Iterations')
        plt.ylabel('Fitness Value (MAE)')

        text_str = f"Min MAE = {self.gbest_fitness:.6f}"
        plt.text(
            0.5 * self.max_iterations,              # 文本位置（右上角）
            np.max(self.fitness_progress) * 0.95,
            text_str,
            bbox=dict(boxstyle="round", alpha=0.5)
        )

        plt.savefig(output_dir / f'{self.title}/fitness_progress_标准.png', dpi=700, bbox_inches='tight')
        plt.close()

        pd.DataFrame(self.fitness_progress).to_csv(output_dir / f"{self.title}/fitness_标准.csv", header=False, index=False)

    
    def plot_optimization_result(self):
        """绘制优化后的曲线与目标曲线对比"""
        freq = np.linspace(5.0, 12.0, 1001)
        # 预测优化后的曲线
        S11_design_dB = self.forward_predict(self.gbest)    # dB
        
        S11_design = 10 ** (S11_design_dB / 20)

        A_design = 1 - S11_design ** 2
        A_target = 1 - self.S11_target ** 2
        # 我想转化为吸收效率画图

        plt.plot(freq, A_design, label='Optimization Result', color='green', linestyle='--', linewidth=1)
        plt.plot(freq, A_target, label='Target', color='red', linewidth=1)
        plt.axhline(0.9, color='black', linestyle='-.', linewidth=0.7)                     # 90%吸收率阈值 
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Absorption')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(bbox_to_anchor=(1, 0.8), loc="upper right", fontsize=8)
        plt.savefig(output_dir / f'{self.title}/A_curve_comparison_标准.png', dpi=700, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """保存最优参数和预测曲线"""
        # 保存最优几何参数
        best_params = self.gbest.reshape(1, -1)
        best_params = pd.DataFrame(best_params)
        best_params.to_csv(output_dir / f'{self.title}/best_parameters_标准.csv', index=False, header=False)
        
        # 保存预测曲线
        S11_design = self.forward_predict(self.gbest)
        S11_design = pd.DataFrame(S11_design.reshape(1, -1))
        S11_design.to_csv(output_dir / f"{self.title}/S11_design_标准.csv", index=False, header=False)


if __name__ == "__main__":
    id = 0

    (output_dir / f"{id}").mkdir(exist_ok=True)    

    # 目标曲线
    S11_all = pd.read_csv(input_dir / 'dB_123.csv', header=None).values # dB域  NOTE: 用原始吸收光谱引导

    S11_target = 10 ** (S11_all[id] / 20)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                # 设置设备
    torch.set_default_device(device)
    pso_ID_model = PSO_inverse_design(S11_target, device, num_particles=100, max_iterations=100, title=f"{id}",
                                      w_min=0.8, w_max=0.8, c1_min=1, c1_max=1, c2_min=1, c2_max=1)   
    pso_ID_model.run()


    print(S11_target)