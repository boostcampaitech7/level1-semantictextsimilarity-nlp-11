import pandas as pd
from konlpy.tag import *
import seaborn as sns
import matplotlib.pyplot as plt
import os

class EDA_common_pos:

    def __init__(self, data_path):
        self.tokenizer = Kkma()
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.data_type = data_path.split("/")[-1].replace('.csv', '')
        self.folder_path = self.check_dir()
        
        common_pos_list = None
        if 'integer-label' not in self.df.columns:
            self.get_integer_label()

        if 'num-of-common-noun' not in self.df.columns:
            if common_pos_list is None: common_pos_list = self.get_common_pos()
            self.get_num_of_common_specific_pos(
                common_pos_list, 'noun', ["NNG", "NNP"]
            )
        if 'num-of-common-verb' not in self.df.columns:
            if common_pos_list is None: common_pos_list = self.get_common_pos()
            self.get_num_of_common_specific_pos(
                common_pos_list, 'verb', ["VV", "VA", "XR"]
            )
        if 'num-of-common-adj' not in self.df.columns:
            if common_pos_list is None: common_pos_list = self.get_common_pos()
            self.get_num_of_common_specific_pos(
                common_pos_list, 'adj', ["MM", "MAG"]
            )

    def get_int_label_barplot(self, ):
        self.get_barplot('label')

    def get_label_barplot(self, ):
        self.get_barplot('integer-label')
 
    def get_int_label_noun_boxplot(self, ):
        self.get_noun_boxplot('label')

    def get_label_noun_boxplot(self, ):
        self.get_noun_boxplot('integer-label')

    def get_barplot(self, label_type):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        new_df = self.df.query("source in ['slack-sampled', 'slack-rtt']")
        grouped_data_noun = new_df.groupby([label_type, 'source'])['num-of-common-noun'].mean().reset_index()
        sns.barplot(x=label_type, y='num-of-common-noun', hue='source', \
                    hue_order=['slack-sampled', 'slack-rtt'], data=grouped_data_noun, ax=axes[0])

        new_df = self.df.query("source in ['petition-sampled', 'petition-rtt']")
        grouped_data_noun = new_df.groupby([label_type, 'source'])['num-of-common-noun'].mean().reset_index()
        sns.barplot(x=label_type, y='num-of-common-noun', hue='source', \
                    hue_order=['petition-sampled', 'petition-rtt'], data=grouped_data_noun, ax=axes[1])

        new_df = self.df.query("source in ['nsmc-sampled', 'nsmc-rtt']")
        grouped_data_noun = new_df.groupby([label_type, 'source'])['num-of-common-noun'].mean().reset_index()
        sns.barplot(x=label_type, y='num-of-common-noun', hue='source', \
                    hue_order=['nsmc-sampled', 'nsmc-rtt'], data=grouped_data_noun, ax=axes[2])


        plt.savefig(f'{self.folder_path}/barplot_{label_type}_{self.data_type}.png')

    def get_noun_boxplot(self, label_type):
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(7, 1, figsize=(10, 60))
        sns.boxenplot(x=label_type, y='num-of-common-noun', data=self.df, ax=axes[0])
        axes[0].set_title('total')
        sns.boxenplot(x=label_type, y='num-of-common-noun', data=self.df.query('source == "slack-sampled"'), ax=axes[1])
        axes[1].set_title('slack-sampled')
        sns.boxenplot(x=label_type, y='num-of-common-noun', data=self.df.query('source == "slack-rtt"'), ax=axes[2])
        axes[2].set_title('slack-rtt')
        sns.boxenplot(x=label_type, y='num-of-common-noun', data=self.df.query('source == "petition-sampled"'), ax=axes[3])
        axes[3].set_title('petition-sampled')
        sns.boxenplot(x=label_type, y='num-of-common-noun', data=self.df.query('source == "petition-rtt"'), ax=axes[4])
        axes[4].set_title('petition-rtt')
        sns.boxenplot(x=label_type, y='num-of-common-noun', data=self.df.query('source == "nsmc-sampled"'), ax=axes[5])
        axes[5].set_title('nsmc-sampled')
        sns.boxenplot(x=label_type, y='num-of-common-noun', data=self.df.query('source == "nsmc-rtt"'), ax=axes[6])
        axes[6].set_title('nsmc-rtt')

        plt.savefig(f'{self.folder_path}/boxplot_{label_type}_{self.data_type}.png')

    def get_integer_label(self, ):
        label_list = self.df['label'].tolist()
        integer_label_list = list()
        for label in label_list:
            integer_label_list.append(int(label))

        self.df["integer-label"] = integer_label_list
        self.df.to_csv(self.data_path, index=False)

    def get_num_of_common_specific_pos(self, common_pos_list, col_name, pos_list):
        num_of_common_token_list = list()
        if pos_list is None:
            num_of_common_token_list = [len(common_pos) for common_pos in common_pos_list]

        else:
            for common_pos in common_pos_list:
                num_of_common_token = 0
                for cw in common_pos:
                    if cw[1] in pos_list:
                        num_of_common_token += 1
                num_of_common_token_list.append(num_of_common_token)

        self.df["num-of-common-{col_name}"] = num_of_common_token_list
        self.df.to_csv(self.data_path, index=False)

    def get_common_pos(self, ):
        sentences1 = self.df['sentence_1'].tolist()
        sentences2 = self.df['sentence_2'].tolist()
        common_pos_list = list()

        for sentence1, sentence2 in zip(sentences1, sentences2):
            pos1 = self.tokenizer.pos(self.remove_emojis(sentence1))
            pos2 = self.tokenizer.pos(self.remove_emojis(sentence2))
            common_pos_list.append(self.find_common_words(pos1, pos2))

        return common_pos_list

    def find_common_words(self, text1, text2):
        words1 = set(text1)
        words2 = set(text2)
        common_words = words1.intersection(words2)

        return list(common_words)

    def remove_emojis(self, text):
        return ''.join(c for c in text if not (0x1F600 <= ord(c) <= 0x1F64F or 0x1F300 <= ord(c) <= 0x1F5FF))

    def check_dir(self, ):
        if not os.path.exists('graph/'):
            os.makedirs('graph/')
        folder_path = f"graph/common_pos"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        return folder_path

if __name__=="__main__":
    data_paths = ['/Users/gj/Documents/study/STS/data/dev.csv', '/Users/gj/Documents/study/STS/data/train.csv']
    for data_path in data_paths:
        common_eda = EDA_common_pos(data_path)
        common_eda.get_label_barplot()
        common_eda.get_int_label_barplot()
        common_eda.get_label_noun_boxplot()
        common_eda.get_int_label_noun_boxplot()