import pymysql
import os
import numpy as np
class shortDis:
    def __init__(self):
        self.db = pymysql.connect(host='localhost', user='root', password='sw106666,', port=3306, db='antidict',
                                  charset='utf8mb4')
        self.cursor = self.db.cursor()

    def main(self):
        # 示例文本和情感词
        sentiment_words = self.ReadTxt('./CandidateWord/all/all_adj.txt', encoding='gbk')
        cut_word = []
        for data_dir in self.getDirList('./WeiboData')[:]:
            current_sql = data_dir.split('.csv')[0]
            print(current_sql)
            current_csv = 'epidemic_{}_jieba'.format(current_sql)
            sw = self.checkSqlWord(current_csv)
            cut_word += sw

        # 计算每个词到其最近的情感词的距离
        distances_to_sentiment_word = self.calculate_distance_to_nearest_sentiment_word(cut_word, sentiment_words)
        average = np.mean(distances_to_sentiment_word)
        print(average)

    # 最短距离
    def calculate_distance_to_nearest_sentiment_word(self, sentences, sentiment_words):
        distances = []
        index = 1
        for sentence in sentences[:]:
            print('{}/{}'.format(str(index), str(len(sentences))))
            for i, word in enumerate(sentence):
                nearest_distance = 100
                for sentiment_word in sentiment_words:
                    if sentiment_word in sentence:
                        distance = abs(sentence.index(sentiment_word) - i)
                        nearest_distance = min(nearest_distance, distance)
                if nearest_distance == 100:
                    nearest_distance = 0
                distances.append(nearest_distance)
            index += 1
        return distances

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

if __name__ == '__main__':
    shortDis().main()

