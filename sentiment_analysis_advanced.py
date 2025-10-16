import jieba
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SentimentAnalyzer:
    def __init__(self, negative_file, stopwords_file):
        self.output_dir = "分析结果_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        self.negative_comments = self.read_file(negative_file)
        self.stopwords = set(self.read_file(stopwords_file).splitlines())
        self.words = []
        self.filtered_words = []
        
    @staticmethod
    def read_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()

    def preprocess(self):
        # 分词
        self.words = jieba.lcut(self.negative_comments)
        # 过滤停用词
        self.filtered_words = [word for word in self.words if word not in self.stopwords and len(word) > 1]
        return self

    def analyze_word_frequency(self):
        # 词频统计
        word_counts = Counter(self.filtered_words)
        top_30 = dict(word_counts.most_common(30))
        
        # 创建柱状图
        plt.figure(figsize=(15, 8))
        plt.bar(top_30.keys(), top_30.values())
        plt.title('负面评论词频TOP30')
        plt.xticks(rotation=45)
        plt.xlabel('词汇')
        plt.ylabel('出现频次')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/词频分析.png')
        plt.close()

        # 保存词频统计结果
        with open(f'{self.output_dir}/词频统计.txt', 'w', encoding='utf-8') as f:
            for word, count in word_counts.most_common():
                f.write(f"{word}: {count}次\n")

    def generate_wordcloud(self):
        # 生成词云
        wordcloud = WordCloud(
            font_path='simhei.ttf',  # 请确保此字体文件存在
            width=1200,
            height=800,
            background_color='white'
        )
        word_counts = Counter(self.filtered_words)
        wordcloud.generate_from_frequencies(word_counts)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('负面评论词云图')
        plt.savefig(f'{self.output_dir}/词云图.png')
        plt.close()

    def analyze_word_cooccurrence(self):
        # 词语共现分析
        word_pairs = []
        for i in range(len(self.filtered_words)-1):
            word_pairs.append((self.filtered_words[i], self.filtered_words[i+1]))
        
        pair_counts = Counter(word_pairs)
        top_pairs = dict(pair_counts.most_common(20))
        
        # 创建共现网络图
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

    def analyze_sentence_length(self):
        # 句子长度分析
        sentences = self.negative_comments.split('。')
        sentence_lengths = [len(s.strip()) for s in sentences if s.strip()]
        
        plt.figure(figsize=(12, 6))
        sns.histplot(sentence_lengths, bins=30)
        plt.title('句子长度分布')
        plt.xlabel('句子长度')
        plt.ylabel('频次')
        plt.savefig(f'{self.output_dir}/句子长度分布.png')
        plt.close()

        # 计算基本统计信息
        stats = {
            '平均句子长度': np.mean(sentence_lengths),
            '最长句子长度': max(sentence_lengths),
            '最短句子长度': min(sentence_lengths),
            '句子长度中位数': np.median(sentence_lengths)
        }
        
        with open(f'{self.output_dir}/句子长度统计.txt', 'w', encoding='utf-8') as f:
            for key, value in stats.items():
                f.write(f"{key}: {value:.2f}\n")

    def run_analysis(self):
        print("开始分析...")
        self.preprocess()
        self.analyze_word_frequency()
        self.generate_wordcloud()
        self.analyze_word_cooccurrence()
        self.analyze_sentence_length()
        print(f"分析完成！结果保存在目录：{self.output_dir}")

if __name__ == "__main__":
    negative_file = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\负面评论.txt"
    stopwords_file = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\停用词汇总.txt"
    
    analyzer = SentimentAnalyzer(negative_file, stopwords_file)
    analyzer.run_analysis()
