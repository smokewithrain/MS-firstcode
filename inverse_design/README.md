# 逆设计模型 (Inverse Design Model)

## 项目概述

本项目实现了一个基于粒子群算法(PSO)的超表面逆设计模型，用于根据目标电磁响应吸收光谱曲线（S11曲线）设计出对应的超表面几何参数。该模型输入一条1001个采样点的S11曲线，输出7个几何结构参数。

## 项目结构

```
inverse_design/
├── input/             # 输入数据
│   └── dB_123.csv     # S11值数据
├── output/            # 输出结果
│   └── 0/             # 案例ID
│       ├── A_curve_comparison.png/pdf  # 吸收曲线对比
│       ├── S11_design.csv              # 设计的S11曲线
│       ├── best_parameters.csv         # 最优几何参数
│       ├── fitness.csv                 # 适应度变化
│       └── fitness_progress.png        # 适应度变化曲线
├── DAPSO.py           # 改进版粒子群算法
├── 标准pso.py         # 标准粒子群算法
├── setting.py         # 路径配置
├── up_conv.py         # 正向预测模型
├── best_model.pth     # 预训练的正向预测模型权重
└── README.md          # 项目说明
```

## 核心功能

1. **逆设计算法**：实现了两种粒子群算法（DAPSO和标准PSO）
2. **正向预测**：使用预训练的深度学习模型预测S11曲线
3. **约束处理**：支持几何参数的区间约束和维度约束
4. **动态参数调整**：根据迭代次数动态调整学习因子和惯性权重
5. **结果可视化**：绘制适应度变化曲线和优化结果对比图
6. **结果保存**：保存最优几何参数和预测曲线

## 环境依赖

- Python 3.7+
- PyTorch
- NumPy
- Pandas
- Matplotlib

## 算法原理

### 1. 粒子群算法 (PSO)

粒子群算法是一种基于群体智能的优化算法，通过模拟鸟群觅食行为来寻找最优解。本项目实现了两种PSO算法：

- **DAPSO**：改进版粒子群算法，包含以下特点：
  - 动态调整学习因子和惯性权重
  - 粒子移动步长限制为0.1的整数倍
  - 当连续未改进时，对适应度差的粒子进行随机扰动
  - 速度限制和几何约束检查

- **标准PSO**：标准粒子群算法实现

### 2. 适应度函数

适应度函数使用预测出的S11曲线和目标曲线之间的平均绝对误差(MAE)，MAE越小表示适应度越好：

```python
def cal_fitness_batch(self, particles):
    output_S11_dB = self.forward_predict(particles)
    output_S11 = 10 ** (output_S11_dB / 20)
    mae = np.mean(np.abs(self.S11_target - output_S11), axis=1)
    return mae  # 适应度（越小越好）
```

### 3. 约束处理

- **区间约束**：每个几何参数都有特定的取值范围
- **维度约束**：确保几何结构的物理可行性
  ```python
  @staticmethod
  def check_dimensions(particle):
      """检查维度约束"""
      return (particle[0] + 0.5 * particle[3] + particle[5] <= 7.75) and (particle[1] + 0.5 * particle[3] + particle[6] <= 7.75)
  ```

## 使用方法

### 1. 准备数据

确保输入数据文件位于 `input/` 目录下：
- `dB_123.csv`：S11值数据（1001个采样点）

### 2. 运行逆设计

#### 运行DAPSO算法

```bash
python DAPSO.py
```

#### 运行标准PSO算法

```bash
python 标准pso.py
```

### 3. 配置参数

可以在代码中修改以下参数：

| 参数           | 描述                   | 默认值   |
| -------------- | ---------------------- | -------- |
| num_particles  | 粒子群数量             | 100      |
| max_iterations | 最大迭代次数           | 100      |
| w_min, w_max   | 惯性权重范围           | 0.4, 0.9 |
| c1_min, c1_max | 自身认知权重范围       | 0.5, 2.5 |
| c2_min, c2_max | 全体认知权重范围       | 0.5, 2.5 |
| id             | 案例ID（用于输出目录） | 0        |

### 4. 示例代码

```python
# 目标曲线
S11_all = pd.read_csv(input_dir / 'dB_123.csv', header=None).values
S11_target = 10 ** (S11_all[id] / 20)

# 运行DAPSO算法
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pso_ID_model = PSO_inverse_design(S11_target, device, num_particles=100, max_iterations=100, title=f"{id}")
pso_ID_model.run()

# 输出最优参数
print("最优几何参数:")
print(pso_ID_model.gbest.reshape(1, -1))
```

## 输出结果

运行完成后，输出结果将保存在 `output/{id}/` 目录下：

```
output/{id}/
├── A_curve_comparison.png/pdf  # 吸收曲线对比图
├── A_curve_comparison_标准.png  # 标准PSO吸收曲线对比图
├── S11_design.csv              # 设计的S11曲线
├── S11_design_标准.csv          # 标准PSO设计的S11曲线
├── best_parameters.csv         # 最优几何参数
├── best_parameters_标准.csv     # 标准PSO最优几何参数
├── fitness.csv                 # 适应度变化数据
├── fitness_标准.csv             # 标准PSO适应度变化数据
├── fitness_progress.png        # 适应度变化曲线
└── fitness_progress_标准.png    # 标准PSO适应度变化曲线
```

## 技术亮点

1. **改进的PSO算法**：DAPSO算法通过动态调整参数和随机扰动机制，提高了优化效果
2. **正向预测模型**：使用预训练的深度学习模型替代CST仿真，大大提高了计算效率
3. **约束处理**：严格的几何约束确保设计结果的物理可行性
4. **可视化结果**：详细的图表展示优化过程和结果
5. **批量计算**：支持批量计算粒子群的适应度，提高计算效率


## 注意事项

1. 确保 `best_model.pth` 文件存在，这是预训练的正向预测模型权重
2. 输入的 `dB_123.csv` 文件应包含1001个采样点的S11值
3. 运行时间取决于粒子群数量和迭代次数，建议在GPU环境下运行以提高速度
4. 对于复杂的目标曲线，可能需要调整算法参数以获得更好的结果

## 结果分析

### 适应度变化曲线

适应度变化曲线展示了优化过程中适应度值（MAE）的变化趋势，理想情况下应该逐渐下降并趋于稳定。

### 吸收曲线对比

吸收曲线对比图展示了优化后的吸收曲线与目标吸收曲线的对比，两者越接近表示优化效果越好。

### 最优参数

最优参数是算法找到的最符合目标曲线的几何参数组合，可以直接用于超表面的设计和制造。


## 许可证

本项目仅供学术研究使用。