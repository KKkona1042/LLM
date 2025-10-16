import jieba
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PositiveSentimentAnalyzer:
    def __init__(self, positive_file, stopwords_file):
        self.output_dir = "正面评论分析结果_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        self.positive_comments = self.read_file(positive_file)
        self.stopwords = set(self.read_file(stopwords_file).splitlines())
        self.words = []
        self.filtered_words = []
        self.sentences = []
        
    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    def preprocess(self):
        # 分词和预处理
        self.sentences = [s.strip() for s in self.positive_comments.split('。') if s.strip()]
        self.words = jieba.lcut(self.positive_comments)
        self.filtered_words = [word for word in self.words if word not in self.stopwords and len(word) > 1]
        
        # 为TF-IDF准备数据
        self.processed_sentences = []
        for sentence in self.sentences:
            words = jieba.lcut(sentence)
            filtered = [word for word in words if word not in self.stopwords and len(word) > 1]
            self.processed_sentences.append(' '.join(filtered))
        
        return self

    def analyze_tfidf(self):
        # TF-IDF分析
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.processed_sentences)
        
        # 打印矩阵信息
        print("\nTF-IDF矩阵信息：")
        print(f"矩阵形状: {tfidf_matrix.shape}")
        print(f"非零元素数量: {tfidf_matrix.nnz}")
        print(f"稀疏度: {100.0 * tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2f}%")
        
        # 获取特征词
        feature_names = vectorizer.get_feature_names_out()
        
        # 保存TF-IDF分析结果
        with open(f'{self.output_dir}/TFIDF分析结果.txt', 'w', encoding='utf-8') as f:
            f.write("TF-IDF矩阵信息：\n")
            f.write(f"矩阵形状: {tfidf_matrix.shape}\n")
            f.write(f"非零元素数量: {tfidf_matrix.nnz}\n")
            f.write(f"稀疏度: {100.0 * tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2f}%\n\n")
            
            # 获取每个文档最重要的词
            for idx, doc in enumerate(tfidf_matrix.toarray()):
                top_words_idx = doc.argsort()[-5:][::-1]  # 获取前5个最重要的词
                f.write(f"\n文档 {idx+1} 最重要的词：\n")
                for word_idx in top_words_idx:
                    f.write(f"{feature_names[word_idx]}: {doc[word_idx]:.4f}\n")

        return tfidf_matrix, vectorizer

    def analyze_lda(self, tfidf_matrix, n_topics=5):
        # LDA主题建模
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = lda.fit_transform(tfidf_matrix)
        
        # 打印主题-词语分布
        feature_names = self.vectorizer.get_feature_names_out()
        
        # 保存LDA分析结果
        with open(f'{self.output_dir}/LDA主题分析结果.txt', 'w', encoding='utf-8') as f:
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10:-1]  # 每个主题取前10个词
                top_words = [feature_names[i] for i in top_words_idx]
                f.write(f"\n主题 {topic_idx + 1}:\n")
                f.write(", ".join(top_words) + "\n")
                f.write(f"权重: {topic[top_words_idx]}\n")
        
        # 可视化主题-文档分布
        plt.figure(figsize=(12, 6))
        plt.imshow(lda_output.T, aspect='auto', cmap='YlOrRd')
        plt.title('文档-主题分布热力图')
        plt.xlabel('文档编号')
        plt.ylabel('主题编号')
        plt.colorbar()
        plt.savefig(f'{self.output_dir}/主题分布热力图.png')
        plt.close()

    def generate_advanced_visualizations(self):
        # 1. 词频分析和词云
        word_counts = Counter(self.filtered_words)
        top_30 = dict(word_counts.most_common(30))
        
        # 词频柱状图
        plt.figure(figsize=(15, 8))
        plt.bar(top_30.keys(), top_30.values())
        plt.title('正面评论词频TOP30')
        plt.xticks(rotation=45)
        plt.xlabel('词汇')
        plt.ylabel('出现频次')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/词频分析.png')
        plt.close()

        # 词云图
        wordcloud = WordCloud(
            font_path='simhei.ttf',
            width=1200,
            height=800,
            background_color='white'
        )
        wordcloud.generate_from_frequencies(word_counts)
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('正面评论词云图')
        plt.savefig(f'{self.output_dir}/词云图.png')
        plt.close()

        # 2. 情感强度分布（基于句子长度）
        sentence_lengths = [len(s) for s in self.sentences]
        plt.figure(figsize=(12, 6))
        sns.histplot(sentence_lengths, bins=30)
        plt.title('评论长度分布')
        plt.xlabel('评论长度')
        plt.ylabel('频次')
        plt.savefig(f'{self.output_dir}/评论长度分布.png')
        plt.close()

        # 3. 词语共现网络
        word_pairs = []
        for i in range(len(self.filtered_words)-1):
            word_pairs.append((self.filtered_words[i], self.filtered_words[i+1]))
        pair_counts = Counter(word_pairs)
        top_pairs = dict(pair_counts.most_common(20))
        
        plt.figure(figsize=(15, 10))
        for i, ((word1, word2), count) in enumerate(top_pairs.items()):
            plt.plot([i, i+0.5], [count, count], 'b-', linewidth=2)
            plt.text(i, count, f'{word1}-{word2}\n({count}次)', rotation=45)
        plt.title('词语共现关系TOP20')
        plt.xlabel('词语对')
        plt.ylabel('共现次数')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/词语共现分析.png')
        plt.close()

    def run_analysis(self):
        print("开始分析正面评论...")
        self.preprocess()
        
        # TF-IDF分析
        tfidf_matrix, self.vectorizer = self.analyze_tfidf()
        
        # LDA主题分析
        self.analyze_lda(tfidf_matrix)
        
        # 生成其他可视化
        self.generate_advanced_visualizations()
        
        print(f"分析完成！结果保存在目录：{self.output_dir}")
        print("生成的文件包括：")
        print("1. TFIDF分析结果.txt - 包含TF-IDF矩阵信息和重要词汇")
        print("2. LDA主题分析结果.txt - 包含主题-词语分布")
        print("3. 主题分布热力图.png - 展示文档-主题分布")
        print("4. 词频分析.png - TOP30高频词柱状图")
        print("5. 词云图.png - 词频可视化")
        print("6. 评论长度分布.png - 评论长度统计")
        print("7. 词语共现分析.png - 词语共现关系图")

if __name__ == "__main__":
    positive_file = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\正面评论.txt"
    stopwords_file = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\停用词汇总.txt"
    
    analyzer = PositiveSentimentAnalyzer(positive_file, stopwords_file)
    analyzer.run_analysis()
