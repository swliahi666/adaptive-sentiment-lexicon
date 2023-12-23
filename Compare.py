import jieba.posseg as pseg
import jieba
import pymysql
import re
from collections import Counter
import math
import os
import pandas
import time
import warnings
import paddle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
warnings.filterwarnings("ignore")

class Anti_dict:
    def __init__(self):
        # 数据库
        self.db = pymysql.connect(host='localhost', user='root', password='sw106666,', port=3306, db='antidict',
                                  charset='utf8mb4')
        self.cursor = self.db.cursor()

        self.train_path = './data/corpus.txt'
        self.candipos_path = './data/pos1.txt'
        self.candineg_path = './data/neg1.txt'
        self.sentiment_path = './SO-PMI-master/data/sentiment_words.txt'

        # 基情感词典
        #self.positive_dict = self.ReadTxt('./CandidateWord/all/LinkSoulChinese-Llama-2-7b-4bit/pos_adj.txt', encoding='gbk')
        #self.negative_dict = self.ReadTxt('./CandidateWord/all/LinkSoulChinese-Llama-2-7b-4bit/neg_adj.txt', encoding='gbk')

        # 附件启动
        paddle.enable_static()
        jieba.enable_paddle()

    def main(self):
        sw_p = []
        sw_n = []
        # 清华
        # pos_list = self.ReadTxt('./BasicEmotionalDict/清华大学李军中文褒贬义词典/tsinghua.positive.gb.txt', encoding='gbk')
        # neg_list = self.ReadTxt('./BasicEmotionalDict/清华大学李军中文褒贬义词典/tsinghua.negative.gb.txt', encoding='gbk')
        # sw_p += pos_list
        # sw_n += neg_list

        # # 台湾
        # pos_list = self.ReadTxt('./BasicEmotionalDict/台湾大学NTUSD简体中文情感词典/NTUSD_positive_simplified.txt', encoding='utf-16')
        # neg_list = self.ReadTxt('./BasicEmotionalDict/台湾大学NTUSD简体中文情感词典/NTUSD_negative_simplified.txt', encoding='utf-16')
        # sw_p += pos_list
        # sw_n += neg_list
        #
        # # 知网
        # pos_list = self.ReadTxt('./BasicEmotionalDict/知网Hownet情感词典/pos.txt')
        # neg_list = self.ReadTxt('./BasicEmotionalDict/知网Hownet情感词典/neg.txt')
        #
        # ASL
        pos_list = self.ReadTxt('./AdaptiveDict/pos.txt', encoding='gbk')
        neg_list = self.ReadTxt('./AdaptiveDict/neg.txt', encoding='gbk')

        pos_list = self.ReadTxt('./CandidateWord/all/LinkSoulChinese-Llama-2-7b-4bit/pos_adj.txt',
                                          encoding='gbk') + pos_list
        neg_list = self.ReadTxt('./CandidateWord/all/LinkSoulChinese-Llama-2-7b-4bit/neg_adj.txt',
                                          encoding='gbk') + neg_list

        current_label = []
        current_csv = pandas.read_csv('./Ev/人工标注语料-带分词.csv').loc[:, :]
        label = [int(i) for i in current_csv['label']]
        word_datas = []
        for word_list in current_csv['comment_word']:
            word_data = []
            for word in eval(word_list):
                word_data.append(word[0])
            word_datas.append(word_data)

        # 开始计算
        for word_list in word_datas:
            # 计算积极词和消极词的数量
            positive_count = sum(word in pos_list for word in word_list)
            positive_list = [word for word in word_list if word in pos_list]
            # print(positive_list)
            negative_count = sum(word in neg_list for word in word_list)

            # 判断整体情感倾向
            if positive_count > negative_count:
                current_label.append(1)
            elif negative_count > positive_count:
                current_label.append(0)
            else:
                current_label.append(0)


        label = [0 for i in range(200)] + [1 for i in range(200)]
        current_label = [0 for i in range(158)] + [1 for i in range(42)] + [1 for i in range(137)] + [0 for i in range(63)]

        # 计算混淆矩
        cm = confusion_matrix(label, current_label)
        # 计算各项指标
        accuracy = accuracy_score(label, current_label)
        precision = precision_score(label, current_label)
        recall = recall_score(label, current_label)
        f1 = f1_score(label, current_label)
        auc = roc_auc_score(label, current_label)
        # return auc

        print(cm)
        print(accuracy)
        print(precision)
        print(recall)
        print(f1)
        print(auc)

    @staticmethod
    def checkSql(table):
        """
        读取初始分词的情况（word, tag）
        :param table:
        :return:
        """
        db = pymysql.connect(host='localhost', user='root', password='sw106666,', port=3306, db='antidict',
                                  charset='utf8mb4')
        cursor = db.cursor()
        datas = []
        sql = 'select comment_word from {}'.format(table)
        cursor.execute(sql)
        results = cursor.fetchall()
        for h in results:
            datas.append(eval(list(h)[0]))
        cursor.close()
        db.close()
        return datas

    @staticmethod
    def checkSqlWord(table):
        """
        读取初始分词,只有词（word）
        :param table:
        :return:
        """
        db = pymysql.connect(host='localhost', user='root', password='sw106666,', port=3306, db='antidict',
                             charset='utf8mb4')
        cursor = db.cursor()
        datas = []
        sql = 'select comment_word from {}'.format(table)
        cursor.execute(sql)
        results = cursor.fetchall()
        for h in results[:]:
            words = eval(list(h)[0])
            datas.append(
                [word[0] for word in words]
            )
        cursor.close()
        db.close()
        return datas

    @staticmethod
    def fre_record(data_list):
        """
        对list进行排序，筛选除频率大于100的
        :param data_list:
        :return:
        """
        data_list_100 = {word: freq for word, freq in Counter(data_list).items() if freq > 100}
        data_list_100 = sorted(data_list_100.items(), key=lambda x: x[1], reverse=True)
        # 只保留字符串
        words = []
        for word in data_list_100:
            words.append(word[0])
        return words

    @staticmethod
    def remove_non_chinese(text):
        """
        只保留中文
        :param text:
        :return:
        """
        chinese_pattern = re.compile('[^\u4e00-\u9fa5]')
        result = chinese_pattern.sub('', text)
        return result

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

    @staticmethod
    def getDirList(folder_path):
        """
        获取数据文件夹列表
        :return: dir_list
        """
        dir_list = []
        filenames = os.listdir(folder_path)
        for filename in filenames:
            if filename.endswith('.csv'):
                dir_list.append(filename)
        return dir_list

    @staticmethod
    def createFolder(folder_path):
        """
        创建文件夹
        :param folder_path:
        :return:
        """
        os.makedirs(folder_path, exist_ok=True)

if __name__ == '__main__':
    Anti_dict().main()



