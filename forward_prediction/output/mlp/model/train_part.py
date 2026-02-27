import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np 
import os 
import argparse            # 使用命令行控制模型训练的超参数，不让参数散落到代码中
from pathlib import Path
import pandas as pd
from torch.optim import *


import sys

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))   # 将本地路径加到最前，优先访问
import importlib                       # 支持字符串形式动态导入模块

from data_part import *
from evaluate_part import *         # 主要是评估模型, 基本纯画图, 代码多而且可拆分到另一个文件
from save_part import * 

class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_config):
        self.labels = labels.cpu().numpy()
        self.batch_config = batch_config

        # 各类别索引池
        self.class_indices = {
            k: np.where(self.labels == k)[0]
            for k in batch_config.keys()
        }

        # 关键：epoch长度由“大类”决定（不包含最小类）
        # major_classes = [k for k in batch_config if len(self.class_indices[k]) > 100]
        major_classes = [3]
        self.num_batches = min(
            len(self.class_indices[k]) // batch_config[k]
            for k in major_classes
        )

    def __iter__(self):
        # 每个epoch重新洗牌
        shuffled = {
            k: np.random.permutation(v)
            for k, v in self.class_indices.items()
        }
        ptr = {k: 0 for k in self.batch_config}

        for _ in range(self.num_batches):
            batch = []

            for k, n in self.batch_config.items():
                # 不够就循环采样
                if ptr[k] + n > len(shuffled[k]):
                    shuffled[k] = np.random.permutation(self.class_indices[k])
                    ptr[k] = 0

                batch.extend(shuffled[k][ptr[k]:ptr[k]+n])
                ptr[k] += n

            yield batch

    def __len__(self):
        return self.num_batches


# 训练正向预测模型

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model checkpoint automatically.
    """
    def __init__(self, 
                 patience=20, 
                 min_delta=0.0, 
                 verbose=True, 
                 path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str or Path): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.path = path 
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    # __call__一个魔术方法，让实例像函数一样调用
    def __call__(self, val_loss, model):
        """
        Call this at the end of each epoch.
        
        Args:
            val_loss (float): Current validation loss
            model (nn.Module): Model to save if improved
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self._save_checkpoint(model)
            if self.verbose:
                print(f"✅ Validation loss decreased ({self.best_loss:.6f}). Saving best model.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠️ EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)

class EpochRecorder:
    """保存每轮模型结果"""
    def __init__(self):
        self.history = {}
    
    def update(self, epoch, **metrics):
        self.history.setdefault("epoch", []).append(epoch)

        for k, v in metrics.items():
            self.history.setdefault(k, []).append(float(v))

    def to_dataframe(self):
        return pd.DataFrame(self.history)

def get_mean_metrics(S11_real_all, S11_pred_all):
    """包装函数, 只获取指标mean值"""
    _, summary_df = calculate_metrics(S11_real_all, S11_pred_all)

    metrics = dict(zip(summary_df["Metric"], summary_df["Mean"]))
    return metrics


def compute_loss(S11_linear_true, S11_dB_pred):
    S11_dB_true = 20 * torch.log10(torch.abs(S11_linear_true))

    # 计算S11 dB值的均方误差（MSE）
    loss = F.mse_loss(S11_dB_pred, S11_dB_true, reduction='mean')

    return loss

def train(args, train_set, val_set, forward_network):
    """
    训练模型
    
    Args:
        args: 超参数配置
        train_set: 训练数据集
        val_set: 验证数据集
        forward_network: 使用的正向预测网络架构
    
    Returns:
        train_losses: 每轮训练损失记录
        val_losses: 每轮验证损失记录
    """

    # 1. 创建DataLoader, 使模型按照batch_size小批次读取
    # val drop_last=False, 不丢弃任何样本
    # train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    batch_config = {1: 12, 2: 24, 3: 28}     # 每批次32个，每次随机抽取1，2，3峰的个数

    train_sampler = StratifiedBatchSampler(
        labels=train_set.label, batch_config=batch_config
    )

    train_loader = DataLoader(
        train_set,
        batch_sampler=train_sampler,
    )



    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. 使用的模型
    fnetwork = forward_network().to(args.device)     # 正向模型
    optimizer = AdamW(fnetwork.parameters(), lr=args.lr, weight_decay=1e-5, betas=(0.9, 0.999))  # 优化器
    early_stopping = EarlyStopping(patience=args.patience, min_delta=1e-4, verbose=True, 
                                   path=args.output_dir / "model/best_model.pth")
    # 学习率进一步动态调整
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(   # 学习率调度器
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-5, cooldown=5)
    # mode='min', 监控指标越小越好, 下文中传入每轮的验证损失
    # factor=0.5, 学习率调整系数
    # cooldown=5, 学习率下降后, 5轮内不再监控验证损失

    recorder = EpochRecorder()
    # 4. 开始epoch_num轮训练
    for epoch in range(args.epoch_num):
        all_pred_S11_train = []        # 训练集的
        all_real_S11_train = []

        all_pred_S11_val = []       # 测试集的
        all_real_S11_val = []

        # 训练模型
        train_loss = 0   # 当轮训练损失
        fnetwork.train()
        for param, S11, label in train_loader:         # 每个批次数据
            param, S11, label = param.to(args.device), S11.to(args.device), label.to(args.device)      
            
            outputs_dB = fnetwork(param)           # 模型预测输出
            loss = compute_loss(S11, outputs_dB)

            # 优化器优化模型
            optimizer.zero_grad()               # 必须先对参数梯度进行清零
            loss.backward()                     # 反向传播求梯度,更新参数
            optimizer.step()
            
            train_loss += loss.item()    # 累加每个批次的平均损失值

            outputs_S11 = torch.pow(10, outputs_dB / 20)
            all_pred_S11_train.extend(outputs_S11.detach().cpu().numpy().squeeze())
            all_real_S11_train.extend(S11.detach().cpu().numpy().squeeze())

        train_mean_metrics = get_mean_metrics(all_real_S11_train, all_pred_S11_train)
        train_loss /= len(train_loader)  # 除以批次数

        # 验证模型
        val_loss = 0
        val_samples = 0
        fnetwork.eval()
        with torch.no_grad():
            for param, S11, label in val_loader:   
                # 验证阶段drop_last=False, 所以需要严格按照样本数计算
                val_batch_size = param.size(0)      # 获取当前 batch 的实际样本数
                val_samples += val_batch_size      

                param, S11, label = param.to(args.device), S11.to(args.device), label.to(args.device)

                outputs_dB = fnetwork(param)  # 模型预测输出
                loss = compute_loss(S11, outputs_dB)

                val_loss += loss.item() * val_batch_size  # 累加每个批次的总和损失值

                outputs_S11 = torch.pow(10, outputs_dB / 20)
                all_pred_S11_val.extend(outputs_S11.detach().cpu().numpy().squeeze())
                all_real_S11_val.extend(S11.detach().cpu().numpy().squeeze())



            val_mean_metrics = get_mean_metrics(all_real_S11_val, all_pred_S11_val)
            val_loss /= val_samples                  # 除以样本数

        recorder.update(epoch+1, train_loss=train_loss, val_loss=val_loss,
                        train_MSE=train_mean_metrics["MSE"], train_MAE=train_mean_metrics["MAE"],train_MRE=train_mean_metrics["MRE"],train_ACC=train_mean_metrics["ACC (%)"],
                        val_MSE=val_mean_metrics["MSE"], val_MAE=val_mean_metrics["MAE"],val_MRE=val_mean_metrics["MRE"],val_ACC=val_mean_metrics["ACC (%)"])

        # 每轮保存模型
        torch.save(fnetwork.state_dict(), args.output_dir / f"model/model_epoch_{epoch+1}.pth")

        # 打印当轮模型结果
        print(recorder.to_dataframe().tail(1))

        # 根据验证损失动态调整学习率
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_mean_metrics["MAE"])
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f"Scheduler triggered! lr: {prev_lr:.2e} -> {current_lr:.2e}")

        # 调用早停
        early_stopping(val_mean_metrics["MAE"], fnetwork)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break


        
    # 5. 保存损失
    history_path = args.output_dir / 'evaluate/history.csv'
    recorder.to_dataframe().to_csv(history_path, index=False)
    print(f"History each epoch saved to {history_path}")

def test(args, test_set, forward_network):
    """加载已训练好的模型, 在测试集预测输出并保存"""

    
    # 1. 创建DataLoader, 使模型按照batch_size小批次读取
    # test drop_last=False, 不丢弃任何样本
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. 使用已保存的最后一轮模型，预测输出并保存
    # 加载模型
    fnetwork = forward_network().to(args.device)
    recorder = EpochRecorder()
    
    # 加载权重
    model_path = args.output_dir / f"model/best_model.pth"
    fnetwork.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    
    fnetwork.eval()
    all_predictions = []
    with torch.no_grad():
        for param, _, _ in test_loader:
            param = param.to(args.device)
            outputs_dB = fnetwork(param)           # 模型预测输出
            outputs_S11 = torch.pow(10, outputs_dB / 20)

            all_predictions.append(outputs_S11.cpu())
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    pd.DataFrame(all_predictions).to_csv(args.output_dir / f"evaluate/prediction.csv", index=False, header=False)

    # 3. 每轮测试指标(回溯), 并保存
    for epoch in range(args.epoch_num):
        try:
            model_path = args.output_dir / f"model/model_epoch_{epoch+1}.pth"
            fnetwork.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
        except:
            break 
        
        
        fnetwork.eval()
        test_loss = 0
        test_samples = 0
        all_pred_S11_test = []       # 测试集的
        all_real_S11_test = []
        with torch.no_grad():
            for param, S11, label in test_loader:
                # 测试阶段drop_last=False, 所以需要严格按照样本数计算
                test_batch_size = param.size(0)      # 获取当前 batch 的实际样本数
                test_samples += test_batch_size 

                param, S11, label = param.to(args.device), S11.to(args.device), label.to(args.device)

                outputs_dB = fnetwork(param)  # 模型预测输出
                loss = compute_loss(S11, outputs_dB)

                test_loss += loss.item() * test_batch_size  # 累加批次的总损失

                outputs_S11 = torch.pow(10, outputs_dB / 20)
                all_pred_S11_test.extend(outputs_S11.detach().cpu().numpy().squeeze())
                all_real_S11_test.extend(S11.detach().cpu().numpy().squeeze())
            
            test_mean_metrics = get_mean_metrics(all_real_S11_test, all_pred_S11_test)
            test_loss /= test_samples                           # 除以总样本数
        recorder.update(epoch+1, test_loss=test_loss,
                    test_MSE=test_mean_metrics["MSE"], test_MAE=test_mean_metrics["MAE"],test_MRE=test_mean_metrics["MRE"],test_ACC=test_mean_metrics["ACC (%)"])
    
    history_path = args.output_dir / 'evaluate/history.csv'
    new_df = recorder.to_dataframe().drop("epoch", axis=1)
    new_df

    if history_path.exists():
        old_df = pd.read_csv(history_path)
        merged_df = pd.concat([old_df, new_df], axis=1)  # axis=1,横向拼接
    else:
        merged_df = new_df

    merged_df.to_csv(history_path, index=False)
    print(f"History appended to {history_path}")

def pred_valid(args, val_set, forward_network):
    """加载已训练好的模型, 在验证集预测输出并保存"""

    
    # 1. 创建DataLoader, 使模型按照batch_size小批次读取
    # test drop_last=False, 不丢弃任何样本
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. 使用已保存的最后一轮模型，预测输出并保存
    # 加载模型
    fnetwork = forward_network().to(args.device)
    
    # 加载权重
    model_path = args.output_dir / f"model/best_model.pth"
    fnetwork.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    
    fnetwork.eval()
    all_predictions = []
    with torch.no_grad():
        for param, _, _ in val_loader:
            param = param.to(args.device)
            outputs_dB = fnetwork(param)           # 模型预测输出
            outputs_S11 = torch.pow(10, outputs_dB / 20)

            all_predictions.append(outputs_S11.cpu())
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    pd.DataFrame(all_predictions).to_csv(args.output_dir / f"evaluate/val_prediction.csv", index=False, header=False)

def pred(args, data_set, forward_network):
    """加载已训练好的模型, 在验证集预测输出并保存"""

    data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 2. 使用已保存的最后一轮模型，预测输出并保存
    # 加载模型
    fnetwork = forward_network().to(args.device)
    
    # 加载权重
    model_path = args.output_dir / f"model/model_epoch_88.pth"
    fnetwork.load_state_dict(torch.load(model_path, map_location=args.device, weights_only=True))
    
    fnetwork.train()
    all_predictions = []
    with torch.no_grad():
        for param, _, _ in data_loader:
            param = param.to(args.device)
            outputs_dB = fnetwork(param)           # 模型预测输出
            outputs_S11 = torch.pow(10, outputs_dB / 20)

            all_predictions.append(outputs_S11.cpu())
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    return pd.DataFrame(all_predictions)


def save_model_definition(args):
    """保存所用的网络架构"""
    model_path = Path(__file__).resolve().parent / 'network' / f'{args.model}.py'
    train_part_path = Path(__file__).resolve()

    shutil.copy(model_path, args.output_dir / f'model/{args.model}.py')                       # 复制文件
    shutil.copy(train_part_path, args.output_dir / 'model/train_part.py')
    print(f"Model definition saved.")

def main():
    # 将超参数整理起来
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument("--epoch_num", default=100, type=int, help="train epoch")
    parser.add_argument("--seed", default=42, type=int, help="set seed")
    parser.add_argument("--input_dir", default="./input", type=str, help="input data dir")
    parser.add_argument("--output_dir", default="./output", type=str, help="output data dir")
    parser.add_argument("--save_index", action="store_true", help="if save index of train_val_test")
    parser.add_argument("--model", default="mlp", type=str, help="model type to use")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--patience", default=20, type=int, help="patience for early stop")
   
    args = parser.parse_args()
    

    # 路径转换为Path格式，这样方便后续路径连接
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    # 设置随机数种子 函数在data_part.py中
    set_seed(args)           
    # 设置设备
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立输出所在文件夹
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir / "model", exist_ok=True)   # 存放模型
    os.makedirs(args.output_dir / "evaluate", exist_ok=True)   # 存放评估结果

    # 设置日志记录
    log  = setup_logging(args) 

    # 划分训练集、验证集、测试集
    train_set, val_set, test_set = load_data(args) 

    # 根据命令行参数加载模型
    model_module = importlib.import_module(f"network.{args.model}")
    forward_network = model_module.forward_network
    
    save_model_definition(args)
    save_hyperparameters(args)

    # 训练模型 使用验证集实现模型早停
    train(args, train_set, val_set, forward_network)

    # 输出测试集的预测输出 根据最佳模型
    test(args, test_set, forward_network)

    # 评估测试集预测结果
    evaluate(args)

    # 输出验证集上的预测输出， (NOTE: 如果测试集不够用的话)
    # pred_valid(args, val_set, forward_network)
    # 输出训练集上的预测输出
    # all_predictions = pred(args, train_set, forward_network)
    # all_predictions.to_csv(args.output_dir / f"evaluate/train_prediction.csv", index=False, header=False)

    sys.stdout = sys.__stdout__
    log.close()    # 关闭日志文件, 恢复默认的stdout

    # os.system("/usr/bin/shutdown")

if __name__ == "__main__":
    # 调试时，模仿命令行输入参数
    import sys 
    sys.argv = [
        "train_test_evaluate.py",
        "--batch_size", "64",
        "--epoch_num", "50",
        "--seed", "42",
        "--input_dir", r"input",
        "--output_dir", r"output\mlp",
        "--save_index",         # 保存训练集、验证集、测试集的划分索引
        "--model", "mlp",     # 选择要用的模型
        "--lr", "2e-3",
        "--patience", "5",
    ]


    main()