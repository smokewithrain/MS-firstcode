# 该文件可单独使用，也可直接配套train_test_evaluate.py使用

# train_test_evaluate.py文件输出结果：
# output_dir/evaluate/
#               losses.csv 列Train_Loss, Val_Loss, Test_Loss
#               prediction.csv 

# 在这个文件中
# 进行损失曲线可视化
# 评估测试结果， 要好多个指标
# 保存结果同样存放在output/model/中

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse            # 使用命令行控制模型训练的超参数，不让参数散落到代码中
# IEEE绘图要求
# 保存pdf和png格式
# 分辨率：700dpi
# 尺寸：3.5英尺
# 图中字体：10pt

# 设置全局字体和绘图样式（符合IEEE要求）
plt.rcParams['font.family'] = 'Times New Roman'  # IEEE常用字体
plt.rcParams['font.size'] = 10  # 字体大小10pt
plt.rcParams['axes.linewidth'] = 0.8  # 坐标轴线条宽度
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['figure.figsize'] = [3.5, 2.5] 


def plot_loss(args, loss_df):
    # 1) 训练损失 vs 验证损失 
    plt.figure()
    plt.plot(loss_df.index + 1, loss_df['train_loss'], label='Train Loss', color='red', linewidth=1)
    plt.plot(loss_df.index + 1, loss_df['val_loss'], label='Validation Loss', color='blue', linestyle='--', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # plt.yticks([0, 0.01, 0.02, 0.03, 0.04])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域

    # 保存
    plt.savefig(args.output_dir / 'evaluate/train_val_loss.pdf', dpi=700)
    plt.savefig(args.output_dir / 'evaluate/train_val_loss.png', dpi=700)
    plt.close()

    # 2) 训练损失 vs 测试损失 
    plt.figure()
    plt.plot(loss_df.index + 1, loss_df['train_loss'], label='Train Loss', color='red', linewidth=1)
    plt.plot(loss_df.index + 1, loss_df['test_loss'], label='Test Loss', color='blue', linestyle='--', linewidth=1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # plt.yticks([0, 0.01, 0.02, 0.03, 0.04])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域

    # 保存
    plt.savefig(args.output_dir / 'evaluate/train_test_loss.pdf', dpi=700)
    plt.savefig(args.output_dir / 'evaluate/train_test_loss.png', dpi=700)
    plt.close()


def calculate_metrics(S11_real_all, S11_pred_all, eps=1e-8):
    """
    计算每个样本的指标及所有样本指标均值
    
    Args:
        S11_real_all: 测试集真实输出, S11线性值  (2160, 1001)
        S11_pred_all: 测试集预测输出, S11线性值
        eps: 防止除零错误
    Returns:
        all_metrics_df: 每个样本的指标MSE/MAE/MRE/ACC
        summary_df: 全部样本的指标均值

    """
    # 预测的S11曲线是否能够替代仿真软件
    # 算出每个样本的指标, MSE/MAE/MRE/ACC
    mse_list, mae_list, mre_list, acc_list = [], [], [], []
    for i in range(len(S11_real_all)):
        y_true = S11_real_all[i].astype(np.float32)
        y_pred = S11_pred_all[i].astype(np.float32)

        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        denominator = np.maximum(np.abs(y_true), eps)
        mre = np.mean(np.abs(y_true - y_pred) / denominator)
        acc = (1 - mre) * 100

        mse_list.append(mse)
        mae_list.append(mae)
        mre_list.append(mre)
        acc_list.append(acc)

    all_metrics_df = pd.DataFrame({'mse': mse_list, 'mae': mae_list, 'mre': mre_list, 'acc': acc_list})
    
    # 算出平均指标 并保存
    metrics_summary = {
            'Metric': ['MSE', 'MAE', 'MRE', 'ACC (%)'],
            'Mean': [
                np.mean(mse_list),
                np.mean(mae_list),
                np.mean(mre_list),
                np.mean(acc_list)
            ],
            'Std': [
                np.std(mse_list),
                np.std(mae_list),
                np.std(mre_list),
                np.std(acc_list)
            ]
        }

    summary_df = pd.DataFrame(metrics_summary)

    return all_metrics_df, summary_df

def plot_mse_hist(args, mse_list, bins=50, title="mae_hist"):
    """
    根据全部样本的mse, 绘制mse分布直方图
    """
    plt.figure()
    plt.hist(mse_list, bins=bins, color='#FFA07A', edgecolor='black', alpha=0.7)
    mean_val, median_val = np.mean(mse_list), np.median(mse_list)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.6f}')
    plt.axvline(median_val, color='blue', linestyle='-.', label=f'Median: {median_val:.6f}')
    plt.xlabel('MAE')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    plt.tight_layout() 
    plt.savefig(args.output_dir / f'evaluate/{title}.pdf', dpi=700)
    plt.savefig(args.output_dir / f'evaluate/{title}.png', dpi=700)
    plt.close()

def plot_absorption_comparison(args, S11_real, S11_pred, idx, mae, title):
    freq = np.linspace(5.0, 12.0, 1001)
    absorption_real = 1 - S11_real ** 2
    absorption_pred = 1 - S11_pred ** 2

    plt.figure()
    plt.plot(freq, absorption_real, label='Real', color="red", linewidth=1)
    plt.plot(freq, absorption_pred, label='Prediction', color="blue", linestyle='--', linewidth=1)
    plt.axhline(0.9, color='black', linestyle='-.', linewidth=0.7)                     # 90%吸收率阈值
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Absorption')
    plt.ylim(0, 1)
    plt.title(f'Sample #{idx}|MAE: {mae:.6f}')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1, 0.8), loc="upper right", fontsize=8)

    plt.tight_layout() 
    plt.savefig(args.output_dir / f'evaluate/{title}.pdf', dpi=700)
    plt.savefig(args.output_dir / f'evaluate/{title}.png', dpi=700)
    plt.close()

def evaluate(args):
    # 先要总的测试集上的指标 + mae直方图
    # 再1，2，3峰(50%以上)，每类的评估效果 指标
    # 最后，筛选出最符合工程需要的典型1，2，3峰(9050) 的min, mean, max 样本一览


    # 1. 损失曲线可视化 
    loss_df = pd.read_csv(args.output_dir / "evaluate/history.csv")
    plot_loss(args, loss_df)
    print("损失曲线可视化已完毕")
    
    # 2. 在测试集上评估模型的预测结果
    # 1) 根据index文件夹的数据，找测试集样本
    S11_all = pd.read_csv(args.input_dir / "dB_123.csv", header=None).values         # 全部样本集
    S11_all = 10 ** (S11_all / 20)
    test_idx = pd.read_csv(args.input_dir / "index/test_idx.csv")['index'].values
    S11_test = S11_all[test_idx]
    S11_test_pred = pd.read_csv(args.output_dir / "evaluate/prediction.csv", header=None).values  #       预测输出
    # 2) 总测试集上的指标 + mae直方图
    # 2.1)  计算指标：MSE\MAE\MRE\Accuracy
    all_metrics_df, summary_df = calculate_metrics(S11_test, S11_test_pred, eps=1e-8)
    all_metrics_df.to_csv(args.output_dir / "evaluate/all_metrics.csv", index=False)
    summary_df.to_csv(args.output_dir / "evaluate/metrics_summary.csv", index=False)
    print("测试集上的评估指标计算已完毕")

    # 2.2) 画图, 针对MAE指标, 解析样本的分布情况, 以期找出模型的改进点
    # mae直方图, 全部样本
    mae_list = np.array(all_metrics_df["mae"])
    plot_mse_hist(args, mae_list, bins=50, title='mae_hist')
    print("测试集上全部样本的mae直方图已绘制")

    # 3) 1，2，3峰(50%以上)，每类的评估效果 指标
    label_50 = pd.read_csv(args.input_dir / "label_123.csv", encoding="ANSI")['valley50']                     # 测试集标签 
    label_50_test = label_50[test_idx].values
    all_metrics_df = all_metrics_df.copy()  # 防止链式赋值警告
    all_metrics_df["label_50"] = label_50_test
    group_mean_df = all_metrics_df.groupby("label_50").mean().reset_index()
    group_mean_df.to_csv(
        args.output_dir / "evaluate/metrics_label50.csv",
        index=False
    )

    # 4) 最符合工程需要的典型1，2，3峰(9050) 的min, mean, max 样本一览


    # 2.2) GROUP BY 95分层, 分别绘制mse直方图
    label_95 = pd.read_csv(args.input_dir / "label_123.csv", encoding="ANSI")['valley9050']                     
    label_95_test = label_95[test_idx].values                                                       # 测试集标签 
 
    # for i in ['典型1', '典型2', '典型3', '非典型1', '非典型2']:
    #     mae_95 = mae_list[label_95_test==i]
    #     plot_mse_hist(args, mae_95, bins=30, title=f"mae_hist_{i}")
    # print("测试集上group by 95分层的mae直方图已绘制")
    
    # # 2.3) 95分层 找出其中mse最低，最接近平均值，最高的样本索引
    # # 并绘制样本的real vs pred absorption
    for i in ['典型1', '典型2', '典型3']:
        mask = (label_95_test == i)
        idxs = np.where(mask)[0]                      # 样本在全部测试集中的索引 (0~2159)   

        mae_95= mae_list[label_95_test==i]

        # 找出其中mse最低，最接近平均值，最高的样本索引
        min_idx = idxs[np.argmin(mae_95)]
        max_idx = idxs[np.argmax(mae_95)]
        mean_mae = mae_95.mean()
        mean_closet_idx = idxs[np.argmin(np.abs(mae_95-mean_mae))]

        selected_indices = [min_idx, mean_closet_idx, max_idx]
        titles = [f"{i}_min_sample", f"{i}_mean_sample", f"{i}_max_sample"]

        for idx, title in zip(selected_indices, titles):
            sample_mae = mae_list[idx]
            S11_real = S11_test[idx].astype(np.float32)
            S11_pred = S11_test_pred[idx].astype(np.float32)

            plot_absorption_comparison(args, S11_real, S11_pred, idx, sample_mae, title)
        print(f"测试集上{i}的mae最低、最接近平均值、最高的样本 real vs pred absorption已绘制")
    

if __name__ == "__main__":
    import sys 
    sys.argv = [
        "evaluate_part.py",
        "--input_dir", r"input",
        "--output_dir", r"output\mlp",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./input", type=str, help="input data dir")
    parser.add_argument("--output_dir", default="./output", type=str, help="output data dir")
    args = parser.parse_args()

    # 路径转换为Path格式，这样方便后续路径连接
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)

    evaluate(args)
   









