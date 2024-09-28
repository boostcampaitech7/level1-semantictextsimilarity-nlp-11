import pandas as pd
import transformers
import os

class DataAugmentation: 

    def __init__(self, data_path):
        self.data_path = data_path

    def label_a_to_b(self, a, b):
        df = pd.read_csv(f'{self.data_path}/train.csv')
        label_1_to_3 = df.query(f'{a} <= label <= {b}')
        sentence_1 = label_1_to_3['sentence_1']
        sentence_2 = label_1_to_3['sentence_2']
        label_1_to_3['sentence_1'] = sentence_2
        label_1_to_3['sentence_2'] = sentence_1

        if 'py-ko-spacing_sentence_1' in label_1_to_3.columns:
            sentence_1 = label_1_to_3['py-ko-spacing_sentence_1']
            sentence_2 = label_1_to_3['py-ko-spacing_sentence_2']
            label_1_to_3['py-ko-spacing_sentence_1'] = sentence_2
            label_1_to_3['py-ko-spacing_sentence_2'] = sentence_1
        
        if 'py-ko-spacing_hanspell_sentence_1' in label_1_to_3.columns:
            sentence_1 = label_1_to_3['py-ko-spacing_hanspell_sentence_1']
            sentence_2 = label_1_to_3['py-ko-spacing_hanspell_sentence_2']
            label_1_to_3['py-ko-spacing_hanspell_sentence_1'] = sentence_2
            label_1_to_3['py-ko-spacing_hanspell_sentence_2'] = sentence_1

        if 'hanspell_sentence_1' in label_1_to_3.columns:
            sentence_1 = label_1_to_3['hanspell_sentence_1']
            sentence_2 = label_1_to_3['hanspell_sentence_2']
            label_1_to_3['hanspell_sentence_1'] = sentence_2
            label_1_to_3['hanspell_sentence_2'] = sentence_1

        df = pd.concat([df, label_1_to_3], ignore_index=True)
        df.to_csv(f'{self.data_path}/aug_{a}-to-{b}_train.csv', index=False)

    def max_len(self, col_name, tokenizer):
        max_len = 0
        max_sentence = ""
        for data_type in ['train', 'dev', 'test']:
            df = pd.read_csv(f'{self.data_path}/{data_type}.csv')
            for text in df[f'{col_name}sentence_1']:
                ids = tokenizer(text)['input_ids']
                if max_len < len(ids):
                    max_len = len(ids)
                    max_sentence = text
            for text in df[f'{col_name}sentence_2']:
                ids = tokenizer(text)['input_ids']
                if max_len < len(ids):
                    max_len = len(ids)
                    max_sentence = text

        return max_len

    def split_random(self, ):
        pass

    def split_uniform1(self, ):
        pass

    def split_uniform2(self, ):
        pass

    def split_uniform3(self, ):
        pass

    def rtt_only(self, ):
        pass
        
    def rtt_v2(self, ):
        n = 1000
        total_train_df = pd.read_csv(f'{self.data_path}/train.csv')
        total_dev_df = pd.read_csv(f'{self.data_path}/dev.csv')
        rtt_only_train_df = total_train_df.query("source == 'slack-rtt' or source == 'petition-rtt' or source == 'nsmc-rtt'")
        rtt_only_dev_df = total_dev_df.query("source == 'slack-rtt' or source == 'petition-rtt' or source == 'nsmc-rtt'")

        sampled_only_train_df = total_train_df.query("source == 'slack-sampled' or source == 'petition-sampled' or source == 'nsmc-sampled'")
        sampled_1_df = sampled_only_train_df.query('1 <= label < 2').sample(n=n, random_state=42, replace=True)
        sampled_2_df = sampled_only_train_df.query('2 <= label < 3').sample(n=n, random_state=42, replace=True)
        sampled_3_df = sampled_only_train_df.query('3 <= label < 4').sample(n=n, random_state=42, replace=True)
        sampled_4_df = sampled_only_train_df.query('4 <= label <= 5').sample(n=n, random_state=42, replace=True)

        rtt_train_df = pd.concat([rtt_only_train_df, sampled_1_df, sampled_2_df, sampled_3_df, sampled_4_df])
        rtt_train_df.loc[rtt_train_df['source'] == 'slack-sampled', 'source'] = 'slack-rtt'
        rtt_train_df.loc[rtt_train_df['source'] == 'petition-sampled', 'source'] = 'petition-rtt'
        rtt_train_df.loc[rtt_train_df['source'] == 'nsmc-sampled', 'source'] = 'nsmc-rtt'

        if not os.path.exists(f"{self.data_path}/rtt-v2"):
            os.makedirs(f"{data_path}/rtt-v2")
        rtt_train_df.to_csv(f'{data_path}/rtt-v2/train.csv', index=False)
        rtt_only_dev_df.to_csv(f'{data_path}/rtt-v2/dev.csv', index=False)
    
    def simplify(self, ):
        cols = ['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label', 'target']
        for file_name in ['dev_output.csv', 'train_output.csv']:
            df = pd.read_csv(f"{self.data_path}/{file_name}")
            new_df = df[cols]
            new_df.to_csv(f"{self.data_path}/{file_name}")


if __name__=='__main__':
    data_path = '/data/ephemeral/home/gj/pytorch-template/output/aug=0.5-to-3.5_plm=kakaobank-kf-deberta-base_val-pearson=0.9309012293815613'
    data_path = '/data/ephemeral/home/gj/pytorch-template/output/aug=0.5-to-3.5_plm=monologg-koelectra-base-v3-discriminator_val-pearson=0.931106686592102'
    data_path = '/data/ephemeral/home/gj/pytorch-template/output/aug=0.5-to-3.5_plm=snunlp-KR-ELECTRA-discriminator_val-pearson=0.9379042387008667'
    #tokenizer = transformers.AutoTokenizer.from_pretrained('snunlp/KR-ELECTRA-discriminator')
    data_aug = DataAugmentation(data_path)
    data_aug.simplify()
    #data_aug.rtt_v2()
    #data_aug.label_a_to_b(0.5, 3.5)
    #data_aug.max_len('py-ko-spacing_hanspell_', tokenizer=tokenizer)