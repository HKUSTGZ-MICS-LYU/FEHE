import ast
import re

import numpy as np

# 定义文件路径
weights_pth = 'src/encrypted/plaintext_params_0_1.txt'
processed_pth = 'src/encrypted/plaintext_params_0_1_processed.txt'

# 读取原始文件内容
with open(weights_pth, 'r') as file:
    data_str = file.read()

# 使用正则表达式在数字后添加逗号（简单示例，可能需要根据具体数据调整）
# 这里假设每个数字后面跟一个空格或换行，非末尾元素
data_str = re.sub(r'(\d(?:\.\d+)?(?:e[+-]?\d+)?)\s+', r'\1, ', data_str)

# 将修改后的内容写入新文件
with open(processed_pth, 'w') as file:
    file.write(data_str)

# 现在读取并解析新文件
with open(processed_pth, 'r') as file:
    processed_data_str = file.read()

try:
    data_list = ast.literal_eval(processed_data_str)
    data_array = np.array(data_list)
    print("Processed Data as NumPy array:")
    print(data_array)
except (ValueError, SyntaxError) as e:
    print(f"Error parsing the processed file: {e}")