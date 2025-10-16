import pandas as pd
import numpy as np
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class HotelSentimentAnalyzer:
    def __init__(self, csv_file):
        self.output_dir = "酒店评论分析结果_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = pd.read_csv(csv_file)
        self.sentiment_scores = []
        self.processed_comments = []
        
    def preprocess_data(self):
        """数据预处理"""
        print("正在进行数据预处理...")
        # 删除空值
        self.df.dropna(inplace=True)
        # 计算情感分数
        self.df['sentiment_score'] = self.df['review'].apply(lambda x: SnowNLP(str(x)).sentiments)
        # 添加评论长度
        self.df['comment_length'] = self.df['review'].apply(len)
        
        return self

    def analyze_neutral_comments(self, lower_threshold=0.4, upper_threshold=0.6):
        """分析中性评论"""
        neutral_mask = (self.df['sentiment_score'] >= lower_threshold) & (self.df['sentiment_score'] <= upper_threshold)
        neutral_comments = self.df[neutral_mask]
        
        # 保存中性评论
        with open(f'{self.output_dir}/中性评论.txt', 'w', encoding='utf-8') as f:
            f.write(f"情感分数在{lower_threshold}-{upper_threshold}之间的评论：\n\n")
            for idx, row in neutral_comments.iterrows():
                f.write(f"评论: {row['review']}\n")
                f.write(f"情感分数: {row['sentiment_score']:.4f}\n")
                f.write("-" * 50 + "\n")

        return neutral_comments

    def perform_tfidf_analysis(self):
        """执行TF-IDF分析"""
        print("正在进行TF-IDF分析...")
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(self.df['review'].astype(str))
        
        # 打印矩阵信息
        print("\nTF-IDF矩阵信息：")
        print(f"矩阵形状: {tfidf_matrix.shape}")
        print(f"非零元素数量: {tfidf_matrix.nnz}")
        print(f"稀疏度: {100.0 * tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2f}%")
        
        # 保存TF-IDF分析结果
        with open(f'{self.output_dir}/TFIDF分析结果.txt', 'w', encoding='utf-8') as f:
            f.write("TF-IDF矩阵信息：\n")
            f.write(f"矩阵形状: {tfidf_matrix.shape}\n")
            f.write(f"非零元素数量: {tfidf_matrix.nnz}\n")
            f.write(f"稀疏度: {100.0 * tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2f}%\n\n")
            
            # 获取特征词
            feature_names = vectorizer.get_feature_names_out()
            # 计算每个词的平均TF-IDF值
            avg_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
            # 获取TOP50的词
            top_indices = avg_tfidf.argsort()[-50:][::-1]
            
            f.write("\nTOP50高TF-IDF值的词：\n")
            for idx in top_indices:
                f.write(f"{feature_names[idx]}: {avg_tfidf[idx]:.4f}\n")
        
        return tfidf_matrix, vectorizer

    def perform_lda_analysis(self, tfidf_matrix, vectorizer, n_topics=5):
        """执行LDA主题分析"""
        print("正在进行LDA主题分析...")
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_output = lda.fit_transform(tfidf_matrix)
        
        # 保存LDA分析结果
        feature_names = vectorizer.get_feature_names_out()
        with open(f'{self.output_dir}/LDA主题分析结果.txt', 'w', encoding='utf-8') as f:
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-10:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                f.write(f"\n主题 {topic_idx + 1}:\n")
                f.write(", ".join(top_words) + "\n")
                f.write(f"权重: {topic[top_words_idx]}\n")
        
        return lda_output

    def generate_visualizations(self):
        """生成可视化图表"""
        # 1. 情感分数分布
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df['sentiment_score'], bins=50)
        plt.title('评论情感分数分布')
        plt.xlabel('情感分数')
        plt.ylabel('频次')
        plt.savefig(f'{self.output_dir}/情感分数分布.png')
        plt.close()

        # 2. 评论长度分布
        plt.figure(figsize=(12, 6))
        sns.histplot(self.df['comment_length'], bins=50)
        plt.title('评论长度分布')
        plt.xlabel('评论长度')
        plt.ylabel('频次')
        plt.savefig(f'{self.output_dir}/评论长度分布.png')
        plt.close()

        # 3. 评论长度vs情感分数散点图
        plt.figure(figsize=(12, 6))
        plt.scatter(self.df['comment_length'], self.df['sentiment_score'], alpha=0.5)
        plt.title('评论长度与情感分数关系')
        plt.xlabel('评论长度')
        plt.ylabel('情感分数')
        plt.savefig(f'{self.output_dir}/评论长度vs情感分数.png')
        plt.close()

    def save_statistics(self):
        """保存统计信息"""
        stats = {
            '总评论数': len(self.df),
            '平均评论长度': self.df['comment_length'].mean(),
            '最长评论长度': self.df['comment_length'].max(),
            '最短评论长度': self.df['comment_length'].min(),
            '平均情感分数': self.df['sentiment_score'].mean(),
            '情感分数中位数': self.df['sentiment_score'].median(),
            '正面评论数(>0.6)': len(self.df[self.df['sentiment_score'] > 0.6]),
            '中性评论数(0.4-0.6)': len(self.df[(self.df['sentiment_score'] >= 0.4) & (self.df['sentiment_score'] <= 0.6)]),
            '负面评论数(<0.4)': len(self.df[self.df['sentiment_score'] < 0.4])
        }

        with open(f'{self.output_dir}/统计分析.txt', 'w', encoding='utf-8') as f:
            f.write("评论数据统计信息：\n\n")
            for key, value in stats.items():
                f.write(f"{key}: {value:.2f}\n")

    def run_analysis(self):
        """运行完整的分析流程"""
        print("开始分析酒店评论数据...")
        
        # 数据预处理
        self.preprocess_data()
        
        # 分析中性评论
        neutral_comments = self.analyze_neutral_comments()
        
        # TF-IDF分析
        tfidf_matrix, vectorizer = self.perform_tfidf_analysis()
        
        # LDA主题分析
        self.perform_lda_analysis(tfidf_matrix, vectorizer)
        
        # 生成可视化
        self.generate_visualizations()
        
        # 保存统计信息
        self.save_statistics()
        
        print(f"\n分析完成！结果保存在目录：{self.output_dir}")
        print("\n生成的文件包括：")
        print("1. 中性评论.txt - 情感倾向不明显的评论列表")
        print("2. TFIDF分析结果.txt - TF-IDF矩阵信息和重要词汇")
        print("3. LDA主题分析结果.txt - 主题-词语分布")
        print("4. 情感分数分布.png - 评论情感分布图")
        print("5. 评论长度分布.png - 评论长度统计图")
        print("6. 评论长度vs情感分数.png - 长度与情感关系图")
        print("7. 统计分析.txt - 整体统计信息")

if __name__ == "__main__":
    csv_file = r"F:\三亚学院\大三上\Natural\景区及酒店网评文本有效性分析\ChnSentiCorp_htl_all.csv"
    
    analyzer = HotelSentimentAnalyzer(csv_file)
    analyzer.run_analysis()
