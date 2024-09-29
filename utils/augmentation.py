import pandas as pd
import transformers
import os

class DataAugmentation: 

    def label_a_to_b(self, data_path, a, b):
        df = pd.read_csv(f'{data_path}/train.csv')
        a_to_b_df = df.query(f'{a} <= label <= {b}')
        sentence_1 = a_to_b_df['sentence_1']
        sentence_2 = a_to_b_df['sentence_2']
        a_to_b_df['sentence_1'] = sentence_2
        a_to_b_df['sentence_2'] = sentence_1

        if 'py-ko-spacing_sentence_1' in a_to_b_df.columns:
            sentence_1 = a_to_b_df['py-ko-spacing_sentence_1']
            sentence_2 = a_to_b_df['py-ko-spacing_sentence_2']
            a_to_b_df['py-ko-spacing_sentence_1'] = sentence_2
            a_to_b_df['py-ko-spacing_sentence_2'] = sentence_1
        
        if 'py-ko-spacing_hanspell_sentence_1' in a_to_b_df.columns:
            sentence_1 = a_to_b_df['py-ko-spacing_hanspell_sentence_1']
            sentence_2 = a_to_b_df['py-ko-spacing_hanspell_sentence_2']
            a_to_b_df['py-ko-spacing_hanspell_sentence_1'] = sentence_2
            a_to_b_df['py-ko-spacing_hanspell_sentence_2'] = sentence_1

        if 'hanspell_sentence_1' in a_to_b_df.columns:
            sentence_1 = a_to_b_df['hanspell_sentence_1']
            sentence_2 = a_to_b_df['hanspell_sentence_2']
            a_to_b_df['hanspell_sentence_1'] = sentence_2
            a_to_b_df['hanspell_sentence_2'] = sentence_1

        df = pd.concat([df, a_to_b_df], ignore_index=True)
        df.to_csv(f'{self.data_path}/aug_{a}-to-{b}_train.csv', index=False)

    def simplify(self, data_path):
        cols = ['id', 'source', 'sentence_1', 'sentence_2', 'label', 'binary-label']
        for file_name in ['dev_output.csv', 'train_output.csv']:
            df = pd.read_csv(f"{data_path}/{file_name}")
            new_df = df[cols]
            new_df.to_csv(f"{data_path}/{file_name}")


if __name__=='__main__':
    data_path = None
    data_aug = DataAugmentation()
    data_aug.label_a_to_b(data_path)
    data_aug.simplify(data_path)