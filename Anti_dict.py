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
from nltk.text import TextCollection
from LLM_answer import LLMClassification
warnings.filterwarnings("ignore")

class Anti_dict:
    def __init__(self):
        # 数据库
        self.db = pymysql.connect(host='localhost', user='root', password='###,', port=3306, db='###',
                                  charset='utf8mb4')
        self.cursor = self.db.cursor()

        self.train_path = './data/corpus.txt'
        self.candipos_path = './data/pos1.txt'
        self.candineg_path = './data/neg1.txt'
        self.sentiment_path = './SO-PMI-master/data/sentiment_words.txt'

        # 基情感词典
        self.positive_dict = self.ReadTxt('./CandidateWord/all/LinkSoulChinese-Llama-2-7b-4bit/pos_adj.txt', encoding='gbk')
        self.negative_dict = self.ReadTxt('./CandidateWord/all/LinkSoulChinese-Llama-2-7b-4bit/neg_adj.txt', encoding='gbk')

        # 附件启动
        paddle.enable_static()
        jieba.enable_paddle()
        self.llm = LLMClassification()

    def main(self):
        # 已分完词，读取分词后的结果
        # 种子词挑选
        current_index = 'all'
        self.createFolder('./CandidateWord/{}'.format(current_index))
        data = []
        for data_dir in self.getDirList('./WeiboData')[:]:
            current_sql = data_dir.split('.csv')[0]
            print(current_sql)
            current_csv = 'epidemic_{}_jieba'.format(current_sql)
            sw = self.checkSql(current_csv)
            data += sw
        adv_list, adj_list, adv_adj_list = self.basicSentimentWord(data)
        adj_list_path = './CandidateWord/{}/{}_adj.txt'.format(current_index, current_index)
        adv_list_path = './CandidateWord/{}/{}_adv.txt'.format(current_index, current_index)
        adv_adj_list_path = './CandidateWord/{}/{}_adv_adj.txt'.format(current_index, current_index)
        self.WriteTxt(adj_list_path, adj_list)
        self.WriteTxt(adv_list_path, adv_list)
        self.WriteTxt(adv_adj_list_path, adv_adj_list)

        current_index = 'all'
        self.createFolder('./CandidateWord/{}'.format(current_index))
        data = []
        for data_dir in self.getDirList('./WeiboData')[:]:
            current_sql = data_dir.split('.csv')[0]
            print(current_sql)
            current_csv = 'epidemic_{}_jieba'.format(current_sql)
            sw = self.checkSqlALLWord(current_csv)
            data += sw
        all_list_path = './CandidateWord/{}/{}_word.txt'.format(current_index, current_index)
        self.WriteTxt(all_list_path, list(set(data)))

        # TF-IDF筛选
        adj_list = self.ReadTxt('./CandidateWord/all/all_adj.txt', encoding='gbk')
        cut_word = []
        for data_dir in self.getDirList('./WeiboData')[:]:
            current_sql = data_dir.split('.csv')[0]
            print(current_sql)
            current_csv = 'epidemic_{}_jieba'.format(current_sql)
            sw = self.checkSqlWord(current_csv)
            cut_word += sw

        # 分词
        corpus = TextCollection(cut_word)
        adj_list_tfidf = {}
        i = 1
        for adj_word in adj_list[:]:
            print('status:{}/{}'.format(str(i), str(len(adj_list))))
            adj_list_tfidf[adj_word] = corpus.tf_idf(adj_word, corpus)
            i += 1

        sorted_dict = sorted(adj_list_tfidf.items(), key=lambda x: x[1], reverse=True)
        top_5_keys = [key for key, _ in sorted_dict[:500]]
        top_5_keyValue = [key+' '+str(_) for key, _ in sorted_dict[:500]]
        self.WriteTxt('./CandidateWord/all/seed_adj.txt', top_5_keys)
        self.WriteTxt('./CandidateWord/all/seed_adj_value.txt', top_5_keyValue)

        # LLM判断
        self.llm.main('all')

        # 计算自适应情感词典
        all_data = self.ReadTxt('./CandidateWord/all/all_word.txt', encoding='gbk')
        seed_data = all_data
        seg_word = []
        for data_dir in self.getDirList('./WeiboData')[:]:
            current_sql = data_dir.split('.csv')[0]
            print(current_sql)
            current_csv = 'epidemic_{}_jieba'.format(current_sql)
            sw = self.checkSqlWord(current_csv)
            seg_word += sw
        cowords_list = self.collect_cowords(seed_data, seg_word[:])
        AntiDict = self.collect_candiwords(seg_word, cowords_list)
        self.save_candiwords(AntiDict,
                             './AdaptiveDict/{}_pos.txt'.format('sw'),
                             './AdaptiveDict/{}_neg.txt'.format('sw'),
                             './AdaptiveDict/{}_All_a.txt'.format('sw'))

    def collect_candiwords(self, seg_data, cowords_list):
        """
        计算so-pmi值
        :param seg_data:
        :param cowords_list:
        :return:
        """
        def compute_sopmi(candi_words, pos_words, neg_words, word_dict, co_dict, all):
            pmi_dict = dict()
            for candi_word in set(candi_words):
                pos_sum = 0.0
                neg_sum = 0.0
                for pos_word in pos_words:
                    p1 = word_dict[pos_word] / all
                    p2 = word_dict[candi_word] / all
                    pair = pos_word + '@' + candi_word
                    if pair not in co_dict:
                        continue
                    p12 = co_dict[pair] / all
                    pos_sum += self.compute_mi(p1, p2, p12)

                for neg_word in neg_words:
                    p1 = word_dict[neg_word] / all
                    p2 = word_dict[candi_word] / all
                    pair = neg_word + '@' + candi_word
                    if pair not in co_dict:
                        continue
                    p12 = co_dict[pair] / all
                    neg_sum += self.compute_mi(p1, p2, p12)

                so_pmi = pos_sum - neg_sum
                pmi_dict[candi_word] = so_pmi
            return pmi_dict

        word_dict, all = self.collect_worddict(seg_data)
        co_dict, candi_words = self.collect_cowordsdict(cowords_list)
        pos_words, neg_words = self.collect_sentiwords(word_dict)
        pmi_dict = compute_sopmi(candi_words, pos_words, neg_words, word_dict, co_dict, all)
        return pmi_dict


    def collect_cowords(self, sentiment_words, seg_data):
        """
        统计搭配次数
        """
        def check_words(sent):
            if set(sentiment_words).intersection(set(sent)):
                return True
            else:
                return False

        cowords_list = list()
        window_size = 3
        count = 0
        for sent in seg_data:
            count += 1
            print('Now status:{}/{}'.format(str(count), str(len(seg_data))))
            if check_words(sent):
                for index, word in enumerate(sent):
                    if index < window_size:
                        left = sent[:index]
                    else:
                        left = sent[index - window_size: index]
                    if index + window_size > len(sent):
                        right = sent[index + 1:]
                    else:
                        right = sent[index: index + window_size + 1]
                    context = left + right + [word]
                    if check_words(context):
                        for index_pre in range(0, len(context)):
                            if check_words([context[index_pre]]):
                                for index_post in range(index_pre + 1, len(context)):
                                    cowords_list.append(context[index_pre] + '@' + context[index_post])
        return cowords_list

    def wordCut(self, train_data):
        """
        分割词汇，再readCsv中已完成分词
        :param train_data:
        :return:
        """
        seg_data = list()
        for content in train_data:
            cut_words = pseg.lcut(content, use_paddle=True)
            tag_words = [[self.remove_non_chinese(list(item)[0]), list(item)[1]] for item in cut_words if list(item)[1] not in ['q', 'p', 'm', 'u', 'w', 'xc'] and self.remove_non_chinese(list(item)[0]) != '']
            seg_data.append(tag_words)
        return seg_data

    def basicSentimentWord(self, data_list):
        """
        候选情感词典
        :return:
        """
        # 副词
        adv_list = list()
        # 形容词
        adj_list = list()
        # 副词—形容词
        adv_adj_list = list()
        sum_num = 1
        for tag_list in data_list:
            print('Now status: {}/{}'.format(str(sum_num), str(len(data_list))))
            before_part = ['', '']
            for tag in tag_list:
                if tag[-1] == 'd':
                    adv_list.append(tag[0])
                elif tag[-1] in ('a', 'ad', 'an'):
                    adj_list.append(tag[0])
                elif tag[-1] in ('a', 'ad', 'an') and before_part[-1] == 'd':
                    adv_adj_list.append(before_part[0] + tag[0])
                before_part = tag
            sum_num += 1
        return list(set(adv_list)), list(set(adj_list)), list(set(adv_adj_list))

    def collect_sentiwords(self, word_dict):
        pos_words = set(self.positive_dict).intersection(set(word_dict.keys()))
        neg_words = set(self.negative_dict).intersection(set(word_dict.keys()))
        return pos_words, neg_words

    def save_candiwords(self, pmi_dict, candipos_path, candineg_path, all_path):
        """
        保存自适应情感词典
        :param pmi_dict:
        :param candipos_path:
        :param candineg_path:
        :return:
        """
        pos_dict = dict()
        neg_dict = dict()
        all_dict = dict()
        f_neg = open(candineg_path, 'w+')
        f_pos = open(candipos_path, 'w+')
        f_all = open(all_path, 'w+')

        for word, word_score in pmi_dict.items():
            all_dict[word] = word_score
            if word_score > 0:
                pos_dict[word] = word_score
            elif word_score < 0:
                neg_dict[word] = word_score

        for word, pmi in sorted(pos_dict.items(), key=lambda asd:asd[1], reverse=True):
            f_pos.write(word+'\n')  # 保存词
        for word, pmi in sorted(neg_dict.items(), key=lambda asd:asd[1], reverse=True):
            f_neg.write(word+'\n')  # 保存词
        for word, pmi in sorted(all_dict.items(), key=lambda asd:asd[1], reverse=True):
            f_all.write(word+' ' + str(pmi) + '\n')  # 保存词
        f_neg.close()
        f_pos.close()
        f_all.close()
        return

    @staticmethod
    def compute_mi(p1, p2, p12):
        """
        互信息计算
        :param p1:
        :param p2:
        :param p12:
        :return:
        """
        return math.log2(p12) - math.log2(p1) - math.log2(p2)
    @staticmethod
    def collect_cowordsdict(cowords_list):
        """
        统计词共现次数
        :param cowords_list:
        :return:
        """
        co_dict = dict()
        candi_words = list()
        for co_words in cowords_list:
            candi_words.extend(co_words.split('@'))
            if co_words not in co_dict:
                co_dict[co_words] = 1
            else:
                co_dict[co_words] += 1
        return co_dict, candi_words

    @staticmethod
    def collect_worddict(seg_data):
        """
        统计词频
        :param seg_data:
        :return: word_dict: 【】
                all: sum
        """
        word_dict = dict()
        all = 0
        for line in seg_data:
            for word in line:
                if word not in word_dict:
                    word_dict[word] = 1
                else:
                    word_dict[word] += 1
        all = sum(word_dict.values())
        return word_dict, all

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
    def checkSqlALLWord(table):
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
            datas += [word[0] for word in words]
        cursor.close()
        db.close()
        return list(set(datas))

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



