import jieba
import jieba.posseg as pseg
import pandas
import os
import re
import pymysql
import paddle
class GetData:
    def __init__(self):
        self.db = pymysql.connect(host='localhost', user='root', password='###', port=3306, db='###',
                             charset='utf8mb4')
        self.cursor = self.db.cursor()
        self.data_list = []
        self.table = ''
        self.stop_word = self.ReadTxt('./BasicEmotionalDict/stopword.txt')

        paddle.enable_static()
        jieba.enable_paddle()

    def main(self):
        """
        Process all the data and save in sql
        """
        for data_dir in self.getDirList('./WeiboData')[:]:
            current_csv = pandas.read_csv('./WeiboData/{}'.format(data_dir)).loc[:, :]
            self.table = 'epidemic_{}_jieba'.format(data_dir.split('.csv')[0])
            print(self.table)
            for i in range(len(current_csv['_id'])):
                print('Now status: ./WeiboData/{} {}/{}'.format(data_dir, str(i + 1), str(len(current_csv['_id']))))
                self.channelData(current_csv, i, data_dir.split('.csv')[0])
            self.SaveSql()

        self.cursor.close()
        self.db.close()

    def channelData(self, data, i, time):
        """
        parse data
        :param data: sql table
        :param i: index
        :param time: comment post time
        :return:
        """
        param = dict()
        param['_id'] = str(i)
        param['created_time'] = time
        param['content'] = data['comment'][i]
        try:
            param['comment_word'] = self.coreTest(data['comment'][i])
        except Exception as e:
            print(data['comment'][i])
            param['comment_word'] = '[]'
        param_data = (param['_id'], param['created_time'], param['content'], param['comment_word'])
        self.data_list.append(param_data)

    def coreTest(self, content):
        """
        双向GRU分词
        :param content: comment content
        :return:
        """
        cut_words = pseg.lcut(content, use_paddle=True)
        tag_words = [[self.remove_non_chinese(list(item)[0]), list(item)[1]] for item in cut_words if
                     list(item)[1] not in ['q', 'p', 'm', 'u', 'w', 'xc', 'r', 'c', 'f', 's', 't', 'nr', 'ns', 'nt', 'nw', 'nz'] and self.remove_non_chinese(
                         list(item)[0]) is not '' and list(item)[1] not in self.stop_word]

        return str(tag_words)

    @staticmethod
    def ReadTxt(file_name, encoding='utf-8'):
        """
        Read a txt file
        :param file_name: txt文件地址
        :param encoding: 解码格式
        :return: list
        """
        dict_list = []
        with open(file_name, 'r', encoding=encoding) as file:
            for line in file.readlines():
                dict_list.append(line.strip('\n').strip(' '))
            return list(set(dict_list))

    def SaveSql(self):
        """
        Save data into Mysql
        """
        insert_query = "INSERT INTO {} (_id, comment_time, comment, comment_word) VALUES (%s, %s, %s, %s)" \
            .format(self.table)
        try:
            self.cursor.executemany(insert_query, self.data_list)
            self.db.commit()
            print("Save Success")
        except Exception as e:
            self.db.rollback()
            print(f"Error：{e}")
        self.data_list = []

    @staticmethod
    def remove_non_chinese(text):
        """
        Retain Chinese only
        :param text:
        :return:
        """
        chinese_pattern = re.compile('[^\u4e00-\u9fa5]')
        result = chinese_pattern.sub('', text)
        return result

    @staticmethod
    def getDirList(folder_path):
        """
        Get the list of data folders
        :return: dir_list
        """
        dir_list = []
        filenames = os.listdir(folder_path)
        for filename in filenames:
            if filename.endswith('.csv'):
                dir_list.append(filename)
        return dir_list

if __name__ == '__main__':
    sw = GetData()
    sw.main()






