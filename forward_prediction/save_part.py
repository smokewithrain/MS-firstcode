import shutil
import json
import torch 
from pathlib import Path
import sys 

class Tee(object):
    """实现print内容, 命令行终端和log文件双输出"""
    def __init__(self, *files):
        # *files 可变参数, (sys.__stdout__, log) 即(终端, 文件)
        self.files = files

    def write(self, obj):
        for f in self.files:    # 两个都写入
            f.write(obj)
            f.flush()           # 立刻把缓存中的内容写入文件，防止卡顿错觉

    def flush(self):
        """python标准的输出流要求实现flush方法"""
        for f in self.files:
            f.flush()



def save_hyperparameters(args):
    """将命令行中的超参数保存到JSON文件"""
    hyperparameters = vars(args).copy()                  # 将argparse.Namespace对象转换为字典
    hyperparameter_file = args.output_dir / "model/hyperparameters.json"

    print(hyperparameters)
    for key, value in hyperparameters.items():           # 将不能直接保存的Path类型, 转换为str
        if isinstance(value, Path):                      # 匹配所有Path类型（包括WindowsPath）
            hyperparameters[key] = str(value)
        if isinstance(value, torch.device):              # torch.device也不能直接保存
            hyperparameters[key] = str(value)

    with open(hyperparameter_file, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Hyperparameters saved to {hyperparameter_file}")

def setup_logging(args):
    """设置日志输出, 重定向到文件output.log"""
    log_file = args.output_dir / "model/output.log"
    log = open(log_file, "w", encoding="utf-8")  # log文件, utf-8可以正常输出中文 
    sys.stdout = Tee(sys.__stdout__, log)        # 重定向stdout到文件

    return log