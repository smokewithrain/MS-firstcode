""" 
希望把要设置的路径等, 在这个文件设置好
"""

from pathlib import Path 

base_path = Path(r'inverse_design')

# 设置【输入】数据集、模型路径
input_dir = base_path / "input"
print(f"输入的数据集、模型路径为: {input_dir}")

# 设置【输出】结果路径

output_dir = base_path / 'output'        # 存放输出结果
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)  # parents=True 递归创建父目录
print(f"输出结果存放于: {output_dir}")



