import jieba
import re

# 读取广告文本文件
file_path = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\广告1.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# 使用 jieba 进行中文分词
words = jieba.lcut(text)

# 定义停用词（可根据需要调整）
stop_words = set(["的", "是", "在", "和", "与", "等", "了", "我", "你", "他", "她", "它"])

# 过滤停用词
filtered_words = [word for word in words if word not in stop_words and len(word) > 1]


# 提取各类关键词
def extract_keywords(text, filtered_words):
    # ① 公司关键字（假设公司名称包含大写字母或特定模式）
    company_keywords = re.findall(r"[A-Z\u4e00-\u9fa5]+有限公司|[A-Z\u4e00-\u9fa5]+集团", text)

    # ② 公众关键字（假设公众关注的关键词是高频词）
    from collections import Counter
    word_counts = Counter(filtered_words)
    public_keywords = [word for word, count in word_counts.most_common(10)]  # 提取前 10 个高频词

    # ③ 语句广告（提取完整的句子）
    sentences = re.split(r"[。！？]", text)  # 按标点符号分割句子
    sentence_ads = [sentence.strip() for sentence in sentences if len(sentence.strip()) > 0]

    # ④ 搜索关键字（假设搜索关键词是名词或短语）
    search_keywords = [word for word in filtered_words if word.isalpha()]  # 过滤掉非字母字符

    # ⑤ 竞价排名广告（假设竞价排名广告包含特定词汇，如“推广”、“竞价”）
    bidding_keywords = re.findall(r"推广|竞价|排名|广告", text)

    return {
        "公司关键字": company_keywords,
        "公众关键字": public_keywords,
        "语句广告": sentence_ads,
        "搜索关键字": search_keywords,
        "竞价排名广告": bidding_keywords
    }


# 提取关键词
keywords = extract_keywords(text, filtered_words)

# 将关键词写入新的 txt 文件
output_file_path = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\广告关键词.txt"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for category, words in keywords.items():
        output_file.write(f"{category}:\n")
        for word in words:
            output_file.write(f"  - {word}\n")
        output_file.write("\n")

print(f"关键词已提取并保存到 {output_file_path}")