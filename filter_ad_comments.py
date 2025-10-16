import pandas as pd

# 读取广告关键词文件
ad_keywords_file_path = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\广告关键词.txt"
with open(ad_keywords_file_path, 'r', encoding='utf-8') as file:
    ad_keywords = [line.strip() for line in file if line.strip()]  # 去除空行和空白字符

# 定义动态计算 α 的函数
def calculate_alpha(text_length):
    """
    根据文本长度动态计算 α 的值
    :param text_length: 文本长度
    :return: α 的值
    """
    if text_length < 50:
        return 1
    elif text_length < 100:
        return 2
    elif text_length < 200:
        return 3
    else:
        return 4

# 判断是否为广告评论的函数
def is_ad(comment, ad_keywords):
    """
    判断一条评论是否是广告
    :param comment: 评论文本
    :param ad_keywords: 广告关键词列表
    :return: True 是广告，False 不是广告
    """
    flag = 0
    for keyword in ad_keywords:
        if keyword in comment:  # 确保 comment 是字符串类型
            flag += 1
    # 动态计算 α
    alpha = calculate_alpha(len(comment))
    return flag >= alpha

# 读取 CSV 文件
csv_file_path = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\ChnSentiCorp_htl_all.csv"
df = pd.read_csv(csv_file_path, encoding='utf-8')

# 检查并处理空值
df = df.dropna(subset=['review'])  # 删除 'review' 列中的空值
df['review'] = df['review'].astype(str)  # 确保 'review' 列是字符串类型

# 过滤广告评论
filtered_comments = []
ad_comments = []
for comment in df['review']:
    if is_ad(comment, ad_keywords):
        ad_comments.append(comment)
    else:
        filtered_comments.append(comment)

# 保存广告评论到文件
with open(r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\filter广告.txt", 'w', encoding='utf-8') as f:
    for ad_comment in ad_comments:
        f.write(ad_comment + '\n')

# 保存过滤后的评论到新的 CSV 文件
filtered_df = pd.DataFrame({'review': filtered_comments})
output_file_path = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\ChnSentiCorp_htl_all_filtered.csv"
filtered_df.to_csv(output_file_path, index=False, encoding='utf-8')

print(f"处理完成，广告评论已保存到 'filter广告.txt'，过滤后的评论已保存到 {output_file_path}")