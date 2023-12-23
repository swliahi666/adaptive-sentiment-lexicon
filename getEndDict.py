class sw:
    def main(self):
        # 示例数据: 一组so-PMI的数值列表
        so_pmi_list = self.ReadTxtAll('./AdaptiveDict/sw_All.txt', encoding='gbk')
        stop_word = self.ReadTxtNor('./BasicEmotionalDict/stopword.txt')
        # 分离正负得分
        negative_scores = [[word, score] for word, score in so_pmi_list if score < 6.8]
        positive_scores = [[word, score] for word, score in so_pmi_list if score > 6.8]

        # 负向词典
        neg_data = sorted(negative_scores, key=lambda x: x[1], reverse=False)
        selected_count = int(0.7 * len(neg_data))
        neg_list = neg_data[:selected_count]

        # 正向词典
        pos_data = sorted(positive_scores, key=lambda x: x[1], reverse=True)
        selected_count = int(0.7 * len(pos_data))
        pos_list = pos_data[:selected_count]

        # 只保留字段
        neg_list_full = [word_list[0] for word_list in neg_list if word_list[0] not in stop_word]
        pos_list_full = [word_list[0] for word_list in pos_list if word_list[0] not in stop_word]
        #
        self.WriteTxt('./AdaptiveDict/neg.txt', neg_list_full)
        self.WriteTxt('./AdaptiveDict/pos.txt', pos_list_full)

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

if __name__ == '__main__':
    sw().main()
