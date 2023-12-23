from pyecharts.charts import Bar
from pyecharts import options as opts
import matplotlib.pyplot as plt
import numpy as np
import jieba
class VisData:
    def main(self):
        so_pmi_scores = self.ReadTxtAll('./AdaptiveDict/sw_All.txt', encoding='gbk')
        stop_word = self.ReadTxtNor('./BasicEmotionalDict/stopword.txt')

        the_list = []
        auc_list = []
        pos_list = []
        neg_list = []
        org_list = []
        for the in range(1, 10, 1):
            # 手动实现min-max scaling函数
            from scipy.stats import zscore
            # 由于Z-score可能会产生大于1或小于0的数值，我们需要将其映射到0-1区间
            # 这里我们使用Sigmoid函数进行映射
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # 示例数据: 一组so-PMI的数值列表
            so_pmi_list = [score for word, score in self.ReadTxtAll('./AdaptiveDict/sw_All.txt', encoding='gbk')]
            # 执行min-max scaling
            scaled_so_pmi = sigmoid(zscore(so_pmi_list))
            # 计算0.2在原始数据中对应的值
            def sigmoid_inverse(x):
                return -np.log((1 / x) - 1)
            z_score_for = sigmoid_inverse(the/10)
            mean_of_original = np.mean(so_pmi_list)
            std_of_original = np.std(so_pmi_list)

            # 计算原始值
            original_value_for = (z_score_for * std_of_original) + mean_of_original

            # 分离正负得分
            negative_scores = [word for word, score in so_pmi_scores if score < original_value_for]
            positive_scores = [word for word, score in so_pmi_scores if score > original_value_for]

            # # 对负得分排序并选取最小的75%
            # negative_scores_sorted = sorted(negative_scores)
            # neg_list_cutoff = int(0.75 * len(negative_scores_sorted))
            # neg_list = negative_scores_sorted[:neg_list_cutoff]
            #
            # # 对正得分排序并选取最大的75%
            # positive_scores_sorted = sorted(positive_scores, reverse=True)
            # pos_list_cutoff = int(0.75 * len(positive_scores_sorted))
            # pos_list = positive_scores_sorted[:pos_list_cutoff]

            # 转换为原始形式，即包含词和分数
            # neg_list_full = [word for word, score in so_pmi_scores if score in neg_list and word not in stop_word]
            # pos_list_full = [word for word, score in so_pmi_scores if score in pos_list and word not in stop_word]

            neg_list_full = [word for word in negative_scores if word not in stop_word]
            pos_list_full = [word for word in positive_scores if word not in stop_word]

            self.WriteTxt('./AdaptiveDict/neg.txt', neg_list_full)
            self.WriteTxt('./AdaptiveDict/pos.txt', pos_list_full)

            import Compare
            auc = Compare.Anti_dict().main()

            the_list.append(the/10)
            auc_list.append(auc)
            pos_list.append(len(pos_list_full))
            neg_list.append(len(neg_list_full))
            org_list.append(original_value_for)
        print(the_list)
        print(auc_list)
        print(pos_list)
        print(neg_list)
        print(org_list)

    def Vismain(self):
        scores = self.ReadTxt('./AdaptiveDict/sw_All.txt', encoding='gbk')
        # 分别为小于0和大于0的得分计算区间边界
        min_negative = min(score for score in scores if score < 0)
        max_negative = 0
        min_positive = 0
        max_positive = max(score for score in scores if score > 0)

        # 计算负面得分和正面得分的五个区间边界
        negative_intervals = np.linspace(min_negative, max_negative, 6)
        positive_intervals = np.linspace(min_positive, max_positive, 6)

        # 计算每个区间的数量
        negative_counts = [0] * 5
        positive_counts = [0] * 5

        for score in scores:
            if score < 0:
                for i in range(5):
                    if negative_intervals[i] <= score < negative_intervals[i + 1]:
                        negative_counts[i] += 1
                        break
            else:
                for i in range(5):
                    if positive_intervals[i] <= score < positive_intervals[i + 1]:
                        positive_counts[i] += 1
                        break

        # 创建标签
        negative_labels = [f"{negative_intervals[i]:.2f} - {negative_intervals[i + 1]:.2f}" for i in range(5)]
        positive_labels = [f"{positive_intervals[i]:.2f} - {positive_intervals[i + 1]:.2f}" for i in range(5)]

        fig, ax = plt.subplots(figsize=(12, 8))
        print(negative_counts)
        print(positive_counts)
        negative_counts = [1, 6, 3, 412, 4786]
        positive_counts = [6040, 2288, 285, 29, 7]

        # 绘制负面得分的柱状图
        bars_negative = ax.bar(negative_labels, negative_counts, color='red', label='Negative Scores')

        # 绘制正面得分的柱状图
        bars_positive = ax.bar(positive_labels, positive_counts, color='green', label='Positive Scores')

        # 在每个柱子上方添加计数
        for bar in bars_negative + bars_positive:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        # 设置X轴标签
        ax.set_xticklabels(negative_labels + positive_labels, rotation=45, ha="center")

        # 设置标题和坐标轴标签
        ax.set_title("SO-PMI Scores Distribution")
        ax.set_xlabel("SO-PMI Score Range")
        ax.set_ylabel("Number of Scores")
        ax.legend()

        # 调整布局并保存图表
        plt.tight_layout()
        # plt.show()
        plt.savefig('./sopmi_scores_distribution.png')

    @staticmethod
    def ReadTxt(file_name, encoding='utf-8'):
        """
        读取txt文件，返回去重后的list
        :param file_name: txt文件地址
        :param encoding: 解码格式
        :return: list
        """
        dict_list = []
        with open(file_name, 'r', encoding=encoding) as file:
            for line in file.readlines():
                dict_list.append(float(line.strip('\n').split(' ')[-1]))
            return list(set(dict_list))

    @staticmethod
    def ReadTxtNor(file_name, encoding='utf-8'):
        """
        读取txt文件，返回去重后的list
        :param file_name: txt文件地址
        :param encoding: 解码格式
        :return: list
        """
        dict_list = []
        with open(file_name, 'r', encoding=encoding) as file:
            for line in file.readlines():
                dict_list.append(line.strip('\n').strip(' '))
            return list(set(dict_list))

    @staticmethod
    def ReadTxtAll(file_name, encoding='utf-8'):
        """
        读取txt文件，返回去重后的list
        :param file_name: txt文件地址
        :param encoding: 解码格式
        :return: list
        """
        dict_list = []
        with open(file_name, 'r', encoding=encoding) as file:
            for line in file.readlines():
                dict_list.append([line.strip('\n').split(' ')[0], float(line.strip('\n').split(' ')[-1])])
            return dict_list

    @staticmethod
    def WriteTxt(file_path, data_list):
        """
        存入txt文件
        :param file_path:
        :param data_list:
        :return:
        """
        with open(file_path, 'w') as file:
            for item in data_list:
                file.write(f"{item}\n")

if __name__ == '__main__':
    VisData().Vismain()