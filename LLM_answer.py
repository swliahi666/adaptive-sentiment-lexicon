import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import warnings
warnings.filterwarnings("ignore")

class LLMClassification:
    def __init__(self):
        model_path = "./LLM/LinkSoulChinese-Llama-2-7b-4bit"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, legacy=True)
        if model_path.endswith("4bit"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_4bit=True,
                torch_dtype=torch.float32,
                device_map='auto',
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path).half().cuda()
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def main(self, folder_path):
        adj_list = self.ReadTxt('./CandidateWord/{}/seed_adj.txt'.format(folder_path, folder_path), encoding='gbk')
        adj_pos_list = []
        adj_neg_list = []
        adj_else_list = []
        index = 1
        for adj_word in adj_list[:]:
            print('LLM 判断进程：{}/ {}'.format(str(index), str(len(adj_list))))
            index += 1
            answer = self.JudgeEmotion(adj_word)
            if '积极' in answer:
                adj_pos_list.append(adj_word)
            elif '消极' in answer:
                adj_neg_list.append(adj_word)
            else:
                adj_else_list.append(adj_word)
        # 种子情感词典-分开正负级开始存放
        pos_adj_path = './CandidateWord/{}/pos_adj.txt'.format(folder_path)
        neg_adj_path = './CandidateWord/{}/neg_adj.txt'.format(folder_path)
        else_adj_path = './CandidateWord/{}/else_adj.txt'.format(folder_path)
        self.WriteTxt(pos_adj_path, adj_pos_list)
        self.WriteTxt(neg_adj_path, adj_neg_list)
        self.WriteTxt(else_adj_path, adj_else_list)
        print(adj_pos_list)
        print(adj_neg_list)
        print(adj_else_list)

    def emotion_main(self):
        # save csv
        comment_list = []
        emotion_list = []
        comment_data = pd.read_csv('./Data/2020_05.csv').iloc[:100, :]
        for i in range(len(comment_data['content'])):
            print('LLM 判断进程：{}/ {}'.format(str(i+1), str(len(comment_data['content']))))
            current_comment = comment_data['content'][i]
            instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

                                                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""
            instruction = """[INST] <<SYS>>\n回答我的问题.\n<</SYS>>\n\n{} [/INST]"""
            prompt_text = "我会给你一句话，请判断他是积极的还是消极的，我只希望你回答两个字是'积极' 还是 '消极',不清楚就回答'不知'，判断句子：{}，请记住，只需回答两个字，一定不需要说多余的话".format(
                current_comment)
            prompt = instruction.format(prompt_text)
            generate_ids = self.model.generate(self.tokenizer(prompt, return_tensors='pt').input_ids.cuda(),
                                               max_new_tokens=4096, streamer=self.streamer)

            # 将生成的 IDs 转换为文本
            generated_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
            answer = generated_text
            comment_list.append(current_comment)
            if '积极' in answer:
                emotion_list.append(0)
            elif '消极' in answer:
                emotion_list.append(1)
            else:
                emotion_list.append(-1)

        data = {'comment': comment_list, 'emotion': emotion_list}
        df = pd.DataFrame(data)
        df.to_csv('./LLM/emotion_text.csv', index=False)


    def JudgeEmotion(self, word):
        instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

                            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""
        instruction = """[INST] <<SYS>>\n回答我的问题.\n<</SYS>>\n\n{} [/INST]"""
        prompt_text = "我会给你一个词，请判断他是积极的还是消极的，我只希望你回答两个字是'积极' 还是 '消极'，判断词汇：{}，请记住，只需回答两个字".format(word)
        print(prompt_text)
        prompt = instruction.format(prompt_text)
        generate_ids = self.model.generate(self.tokenizer(prompt, return_tensors='pt').input_ids.cuda(),
                                           max_new_tokens=4096, streamer=self.streamer)

        # 将生成的 IDs 转换为文本
        generated_text = self.tokenizer.decode(generate_ids[0], skip_special_tokens=True)
        answer = generated_text.split('[/INST]')[-1]
        print('##')
        print(word + ' 是 ' + answer)
        return answer

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
    LLMClassification().main('all')
