import transformers
import torch
from module import dataset


class DataModule:
    def __init__(self, dataset_name, plm_name, batch_size, shuffle, train_path, dev_path, test_path, col_info, max_length, add_special_token):
        super().__init__()
        self.plm_name = plm_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.train_dataset = None
        self.dev_dataset = None
        self.test_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(plm_name)
        if len(add_special_token) != 0:
             special_token = {
                  'additional_special_tokens': add_special_token
             }
             self.tokenizer.add_special_tokens(special_token)
        self.col_info = col_info
        self.setup(max_length)

    def setup(self, max_length):
            self.train_dataset = getattr(dataset, self.dataset_name)(self.train_path, self.tokenizer, self.col_info, max_length)
            self.dev_dataset = getattr(dataset, self.dataset_name)(self.dev_path, self.tokenizer, self.col_info, max_length)
            self.test_dataset = getattr(dataset, self.dataset_name)(self.test_path, self.tokenizer, self.col_info, max_length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def dev_dataloader(self):
        return torch.utils.data.DataLoader(self.dev_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)