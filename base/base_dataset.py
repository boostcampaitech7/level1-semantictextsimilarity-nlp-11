import torch
import pandas as pd
from abc import *

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tokenizer, col_info, max_length):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
            max_length (int): max length of input
        """
        super().__init__()
        data = pd.read_csv(data_path)
        self.max_length = max_length
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.col_info = col_info
        self.inputs, self.targets = self.preprocessing(data)

    def __getitem__(self, idx):
        inputs = {k: torch.tensor(v) for k, v in self.inputs[idx].items()}

        if len(self.targets) == 0:
            return inputs
        else:
            return inputs, torch.tensor(self.targets[idx])

    def __len__(self):
        return len(self.inputs)

    @abstractmethod
    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): DataFrame 객체로 구성된 dataset
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression) or int(classification))
        """
        pass
