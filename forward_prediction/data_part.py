# 这一部分是处理数据的，可以单独调用，也可以集成到train_test_evaluate.py中
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import argparse            # 使用命令行控制模型训练的超参数，不让参数散落到代码中
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import os 

class MyDataset(Dataset):
    """管理数据的类, 给神经网络提供合适(归一化)的数据"""
    def __init__(self, param_data, S11_data, label_data, param_min, param_max):
        """ 
        Args:
        param_data: 输入数据, 7个几何结构参数, 列表
        S11_data: 输出数据, S11(幅度)曲线1001个采样点, 列表
        label_data: 数据的标签
        """
        # 如果是固定预处理，如归一化、类型转换等，可以在初始化中进行，一次计算，全局复用，但内存占用高
        # 转换为tensor张量
        self.param = torch.tensor(param_data, dtype=torch.float32)
        self.S11 = torch.tensor(S11_data, dtype=torch.float32)     
        self.label = torch.tensor(label_data, dtype=torch.int64)  

        # parameter 归一化到[0,1] 根据几何参数取值范围
        self.param_min = torch.tensor(param_min, dtype=torch.float32)
        self.param_max = torch.tensor(param_max, dtype=torch.float32)

        range_ = self.param_max - self.param_min
        self.param = (self.param - self.param_min) / range_ 

        # 应用指数变换, S11转换为[0,1]区间
        self.S11 = 10 ** (self.S11 / 20)
        
    def __getitem__(self, index):
        """ 
        DataLoader通过__getitem__()逐个读取该批次内的样本
        """
        # 如果是随机预处理，如数据增强，可放在此处，随用随处理，保证随机性
        param= self.param[index]
        S11 = self.S11[index]
        label = self.label[index]

        return param, S11, label

    def __len__(self):
        """ 
        DataLoader根据数据集的长度__len__()生成样本索引列表
        """
        return len(self.S11)

def load_data(args):
    """返回训练集、验证集、测试集(Dataset)"""
    # 1. 数据文件路径
    param_file = args.input_dir / 'param_123.csv'
    S11_file = args.input_dir / 'dB_123.csv'
    label_file = args.input_dir / 'label_123.csv'

    # 2. 读取数据
    param_all = pd.read_csv(param_file, header=None).values   # 输入
    S11_all = pd.read_csv(S11_file, header=None).values       # 输出
    label_df = pd.read_csv(label_file, encoding="ANSI")                        # 标签, 50用来分层，9050用来展示

    stratify_label = label_df['valley50'].values

    # 3. ndarray 根据索引划分训练集、验证集、测试集   8:1:1, 分层抽样 
    indices = np.arange(len(param_all))      # 用索引划分

    trainval_idx, test_idx = train_test_split(   
        indices, test_size=0.1, random_state=args.seed, stratify=stratify_label
    )
    # 总训练集 = 训练集+验证集
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=1/9, random_state=args.seed, stratify=stratify_label[trainval_idx]
    )

    # 4. 划分出训练集、验证集、测试集
    param_train = param_all[train_idx]
    S11_train = S11_all[train_idx]
    label_train = stratify_label[train_idx]    # 标签只会在训练中用到

    param_val = param_all[val_idx]
    S11_val = S11_all[val_idx]
    label_val = stratify_label[val_idx]

    param_test = param_all[test_idx]
    S11_test = S11_all[test_idx]
    label_test = stratify_label[test_idx]

    param_min = [4.7, 4.7, 11.7, 0.1, 0.1, 0.2, 0.6]   # 几何参数的取值范围 (归一化要用)
    param_max = [6.4, 6.4, 14.9, 1.8, 0.4, 1.1, 1.8]
    
    train_set = MyDataset(param_train, S11_train, label_train, param_min, param_max)
    val_set = MyDataset(param_val, S11_val, label_val, param_min, param_max)
    test_set = MyDataset(param_test, S11_test, label_test, param_min, param_max)

    print(f'训练集样本个数: {len(train_set)}, 验证集样本个数: {len(val_set)}, 测试集样本个数: {len(test_set)}')
    print('数据形式为：')   
    print(f"数据输入param: {train_set[0][0].size()}, {train_set[0][0]}")
    print(f"数据输出S11: {train_set[0][1].size()}, {train_set[0][1]}")
    print(f"数据标签label为: {train_set[0][2]}")
    print(f"label的不同值为: {set(train_set[:][2].numpy())}")

    # 7. 存储训练集、验证集、测试集对应索引，方便后续调用
    if args.save_index:
        os.makedirs(args.input_dir / "index", exist_ok=True)   
        pd.DataFrame(train_idx, columns=['index']).to_csv(args.input_dir / 'index/train_idx.csv', index=False)
        pd.DataFrame(val_idx, columns=['index']).to_csv(args.input_dir / 'index/val_idx.csv', index=False)
        pd.DataFrame(test_idx, columns=['index']).to_csv(args.input_dir / 'index/test_idx.csv', index=False)
        print('数据集索引已保存在index文件夹')

    return train_set, val_set, test_set


def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    # 调试时，模仿命令行输入参数
    import sys 
    sys.argv = [
        "data_part.py",
        "--seed", "42",
        "--input_dir", r"input",
        "--save_index"
    ]

    # 将超参数整理起来
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="set seed")
    parser.add_argument("--input_dir", default="./input", type=str, help="input data dir")
    parser.add_argument("--save_index", action="store_true", help="if save index of train_val_test")
    
    args = parser.parse_args()

    # 路径转换为Path格式，这样方便后续路径连接
    args.input_dir = Path(args.input_dir)

    # 设置随机数种子
    set_seed(args)

    # 划分训练集、验证集、测试集
    train_set, val_set, test_set = load_data(args) 

