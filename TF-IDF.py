import jieba
import pandas as pd
from gensim import corpora, models, similarities

# 加载数据集
data = pd.read_csv(r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\ChnSentiCorp_htl_all.csv")

# 提取评论数据
reviews = data['review'].astype('str').tolist()  # 将评论转换为字符串并提取为列表

# 中文分词
def segment_text(text):
    return [word for word in jieba.lcut(text) if word.strip()]  # 去除空字符

# 对所有评论进行分词
corpus = [segment_text(review) for review in reviews]

# 构建词袋模型
dictionary = corpora.Dictionary(corpus)

# 将分词后的文本转换为词袋向量
corpus_bow = [dictionary.doc2bow(text) for text in corpus]

# 构建 TF-IDF 模型
tfidf_model = models.TfidfModel(corpus_bow)
corpus_tfidf = tfidf_model[corpus_bow]

# 构建相似度矩阵
index = similarities.MatrixSimilarity(corpus_tfidf)

# 计算相似度
similarities_matrix = []
for i, doc_tfidf in enumerate(corpus_tfidf):
    sims = index[doc_tfidf]  # 计算当前文档与其他文档的相似度
    similarities_matrix.append(sims)

# 筛选相似度大于 0.8 的评论对
threshold = 0.8
similar_pairs = []
for i, sims in enumerate(similarities_matrix):
    for j, sim in enumerate(sims):
        if sim > threshold and i != j:  # 排除自身比较
            similar_pairs.append((i, j, sim))

# 打印相似度大于 0.8 的评论对
print("相似度大于 0.8 的评论对：")
for pair in similar_pairs:
    print(f"评论 {pair[0]} 和评论 {pair[1]} 的相似度为 {pair[2]:.4f}")

# 将相似度大于 0.8 的评论对保存到文件
output_file = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\相似度大于0.8的评论对.txt"
with open(output_file, "w", encoding='utf-8') as f:
    for pair in similar_pairs:
        f.write(f"评论 {pair[0]} 和评论 {pair[1]} 的相似度为 {pair[2]:.4f}\n")