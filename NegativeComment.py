from wordcloud import WordCloud
import jieba
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 加载数据集
data = pd.read_csv(r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\ChnSentiCorp_htl_all.csv")

# 提取负面评论并转换为字符串类型
negdata = data[data['label'] == 0]['review'].astype('str')

# 将负面评论保存为 txt 文件
output_file = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\负面评论.txt"
with open(output_file, "w", encoding='utf-8') as f:
    for review in negdata:
        f.write(review + "\n")  # 逐行写入负面评论

# 拼接负面评论
text = ''.join(negdata)  # 使用 join 方法拼接字符串，效率更高

# 中文分词
data_cut = ' '.join(jieba.lcut(text))

# 加载停用词
path = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\停用词汇总.txt"
with open(path, "r", encoding='utf-8') as f:
    stopwords = f.read().splitlines()  # 读取停用词并按行分割

# 自定义颜色函数
def multi_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    colors = [
        "#1f77b4",  # 蓝色
        "#ff7f0e",  # 橙色
        "#2ca02c",  # 绿色
        "#d62728",  # 红色
        "#9467bd",  # 紫色
        "#8c564b",  # 棕色
        "#e377c2",  # 粉红色
        "#7f7f7f",  # 灰色
        "#bcbd22",  # 黄色
        "#17becf"   # 青色
    ]
    return np.random.choice(colors)  # 随机选择颜色

# 生成词云
word_cloud = WordCloud(
    font_path="simsun.ttc",  # 设置字体路径（确保字体文件存在）
    background_color="white",  # 设置背景颜色
    stopwords=set(stopwords),  # 设置停用词（使用集合提高效率）
    width=800,  # 设置词云宽度
    height=600,  # 设置词云高度
    color_func=multi_color_func  # 使用自定义颜色函数
)
word_cloud.generate(data_cut)

# 绘制词云图
plt.figure(figsize=(12, 8))  # 设置画布大小
plt.imshow(word_cloud, interpolation='bilinear')  # 显示词云图
plt.axis("off")  # 关闭坐标轴
plt.show()