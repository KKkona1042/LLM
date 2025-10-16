import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import jieba
import re

# 设置绘图字体为中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 忽略警告信息
import warnings
warnings.filterwarnings("ignore")

# 指定数据集文件的路径
file_path = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\ChnSentiCorp_htl_all.csv"

# 加载 CSV 数据集
data = pd.read_csv(file_path)

# 打印数据集的前几行，确认加载成功
print(data.head())

# 将 review 列转换为字符串类型
data["review"] = data["review"].astype('str')

# 去除数字、地名、无关词语（如“酒店”、“携程”、“年月日”等）
strinfo = re.compile('[0-9]|酒店|携程|年月日|北京|上海|重庆|广州|杭州|南京|成都|苏州|西安|东莞|长沙|济南|深圳|西路|东路')
data["review"] = data["review"].apply(lambda x: strinfo.sub('', x))

# 再次去除数字、“酒店”、“携程”、“年月日”等词语（防止遗漏）
strinfo = re.compile('[0-9]|酒店|携程|年月日')
data["review"] = data["review"].apply(lambda x: strinfo.sub('', x))

# 将空字符串替换为 NaN，方便后续删除
data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)

# 将包含英文字符的行替换为 NaN
data.replace(to_replace=r'[a-zA-Z]', value=np.nan, regex=True, inplace=True)

# 打印清理后的数据集
print(data)

# 删除所有包含 NaN 的行即为Not a Number表示缺失值或无效值
data.dropna(axis=0, how='any', inplace=True)

# 提取正负面评论信息
posdata = data[data['label'] == 1]['review']  # 正面评论
negdata = data[data['label'] == 0]['review']  # 负面评论