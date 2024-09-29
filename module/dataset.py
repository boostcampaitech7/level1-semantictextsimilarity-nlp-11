import torch
import re
import pandas as pd
from abc import *
from base.base_dataset import *
from itertools import accumulate
from tqdm.auto import tqdm
from py_hanspell_master.hanspell import spell_checker
from pykospacing import Spacing
from soynlp.normalizer import *


class DefaultDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = data[self.col_info['label']].values.tolist()
        except:
            targets = []

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(data)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[text_column] for text_column in self.col_info['input']])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)

            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})        
        return data


class SegmentDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = data[self.col_info['label']].values.tolist()
        except:
            targets = []

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(data)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            sentences = [item[text_column] for text_column in self.col_info['input']]
            outputs = self.tokenizer(sentences[0], sentences[1], \
                                     add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        return data


class SourceDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = data[self.col_info['label']].values.tolist()
        except:
            targets = []

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(data)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        sep_token = self.tokenizer.sep_token
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            outputs = self.tokenizer(item['source'] + sep_token + item['sentence_1'] + sep_token + item['sentence_2'], \
                                     add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
            
            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        return data


class SourceSegmentDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = data[self.col_info['label']].values.tolist()
        except:
            targets = []

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(data)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        sep_token = self.tokenizer.sep_token
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            outputs = self.tokenizer(item['source'], item['sentence_1'] + sep_token + item['sentence_2'], \
                                     add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)

            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        return data


class SourceSegmentAlphaDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = data[self.col_info['label']].values.tolist()
        except:
            targets = []

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(data)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        sep_token, sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            outputs = self.tokenizer(item['source'], item['sentence_1'] + sep_token + item['sentence_2'], \
                                     add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
            outputs['token_type_ids'] = [0 for _ in range(self.max_length)]

            idx, sep_count = 0, 0
            for i in range(1, self.max_length):
                outputs['token_type_ids'][i] = idx
                if outputs['input_ids'][i] == sep_token_id:
                    idx += 1
                    sep_count += 1
                if sep_count == 3:
                    break
            
            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        return data
    

class PyKoSpacingDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, df):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = df[self.col_info['label']].values.tolist()
        except:
            targets = []

        # check PyKoSpacing
        if f"py-ko-spacing_{self.col_info['input'][0]}" not in df.columns \
            or f"py-ko-spacing_{self.col_info['input'][1]}" not in df.columns:
            self.py_ko_spacing(df)

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(df)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[f'py-ko-spacing_{text_column}'] for text_column in self.col_info['input']])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True,  max_length=self.max_length)
            
            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        return data

    def py_ko_spacing(self, df):
        spacing = Spacing()

        for input_col in self.col_info['input']:
            spacing_list = list()
            for sentence in tqdm(df[input_col], total=len(df[input_col])):
                spacing_list.append(spacing(sentence))
            df[f'py-ko-spacing_{input_col}'] = spacing_list
        df.to_csv(self.data_path, index=False)


class HanspellDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, df):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = df[self.col_info['label']].values.tolist()
        except:
            targets = []

        # check HanSpell
        if f"hanspell_{self.col_info['input'][0]}" not in df.columns \
            or f"hanspell_{self.col_info['input'][1]}" not in df.columns:
            self.hanspell(df)

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(df)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([item[f'hanspell_{text_column}'] for text_column in self.col_info['input']])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True,  max_length=self.max_length)

            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        return data

    def hanspell(self, df):
        for input_col in self.col_info['input']:
            hanspell_list = list()
            for sentence in tqdm(df[input_col], total=len(df[input_col])):
                hanspell_list.append(spell_checker.check(sentence.replace('&', ' &amp; ')).as_dict()['checked'])
            df[f'hanspell_{input_col}'] = hanspell_list
        df.to_csv(self.data_path, index=False)

class HanspellPyKoSpacingDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, df):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = df[self.col_info['label']].values.tolist()
        except:
            targets = []

        # check PyKoSpacing
        if f"py-ko-spacing_{self.col_info['input'][0]}" not in df.columns \
            or f"py-ko-spacing_{self.col_info['input'][1]}" not in df.columns:
            self.py_ko_spacing(df)

        # check HanSpell
        if f"py-ko-spacing_hanspell_{self.col_info['input'][0]}" not in df.columns \
            or f"py-ko-spacing_hanspell_{self.col_info['input'][1]}" not in df.columns:
            self.hanspell(df)

        # tokenizing by pre-train tokenizer
        inputs = self.tokenizing(df)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            sentences = [item[f'py-ko-spacing_hanspell_{text_column}'] for text_column in self.col_info['input']]
            outputs = self.tokenizer(sentences[0], sentences[1], \
                                     add_special_tokens=True, padding='max_length', truncation=True, max_length=self.max_length)
            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        return data

    def py_ko_spacing(self, df):
        spacing = Spacing()

        for input_col in self.col_info['input']:
            spacing_list = list()
            for sentence in tqdm(df[input_col], total=len(df[input_col])):
                spacing_list.append(spacing(sentence))
            df[f'py-ko-spacing_{input_col}'] = spacing_list
        df.to_csv(self.data_path, index=False)

    def hanspell(self, df):

        for input_col in self.col_info['input']:
            hanspell_list = list()
            for sentence in tqdm(df[f'py-ko-spacing_{input_col}'], total=len(df[f'py-ko-spacing_{input_col}'])):
                hanspell_list.append(spell_checker.check(sentence.replace('&', ' &amp; ')).as_dict()['checked'])
            df[f'py-ko-spacing_hanspell_{input_col}'] = hanspell_list
        df.to_csv(self.data_path, index=False)


class KFDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, col_info, max_length):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.col_info = col_info
        self.max_length = max_length
        self.inputs, self.targets = self.preprocessing(self.data)

    def __getitem__(self, idx):
        inputs = {k: torch.tensor(v) for k, v in self.inputs[idx].items()}
        if len(self.targets) == 0:
            return inputs
        else:
            targets = torch.tensor(self.targets[idx])
            return inputs, targets

    def __len__(self):
        return len(self.inputs)

    def preprocessing(self, data):
        try:
            targets = data[self.col_info['label']].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            sentences = [item[text_column] for text_column in self.col_info['input']]
            outputs = self.tokenizer(sentences[0], sentences[1], 
                                     add_special_tokens=True, padding='max_length',
                                     max_length=self.max_length, truncation=True)

            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})
        
        return data
    

class STSWithBinClsDataset(BaseDataset):
    """
    1단계: 3점 미만, 3점 이상으로 이진 분류, 2단계: 점수 예측하는 STSWithMinClsModel을 위해 label과 binary label을 같이 반환하는 데이터셋
    """
    def __init__(self, data_path, tokenizer, col_info, max_length):
        super().__init__(data_path, tokenizer, col_info, max_length)

    def preprocessing(self, data):
        try:
            labels = data[self.col_info['label']].values.tolist()
        except:
            labels = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)
        bin_labels = list(map(lambda x: 0 if x < 3 else 1, labels))
        targets = []
        for i in range(len(labels)):
            targets.append({'bin': bin_labels[i], 'label': labels[i]})

        return inputs, targets

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            sep_token, sep_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
            outputs = self.tokenizer(item['source'], item['sentence_1'] + sep_token + item['sentence_2'], 
                                     add_special_tokens=True, padding='max_length',
                                     max_length=self.max_length, truncation=True)

            data.append({'input_ids': outputs['input_ids'], 
                         'token_type_ids': outputs['token_type_ids'],
                         'attention_mask': outputs['attention_mask']})

        return data
    
    def __getitem__(self, idx):
        inputs = {k: torch.tensor(v) for k, v in self.inputs[idx].items()}
        if len(self.targets) == 0:
            return inputs
        else:
            targets = {k: torch.tensor(v) for k, v in self.targets[idx].items()}
            return inputs, targets