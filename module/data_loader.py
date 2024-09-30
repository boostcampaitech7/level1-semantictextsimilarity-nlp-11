import transformers
import torch
from module import dataset


class KFoldDataLoader:
    def __init__(self, plm_name, dataset_name, batch_size, shuffle, train_data, dev_data, predict_data, col_info, max_length, add_special_token):
        super().__init__()
        self.plm_name = plm_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = dev_data
        self.predict_data = predict_data

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        if plm_name == "snunlp/KR-ELECTRA-discriminator":
            self.tokenizer = transformers.ElectraTokenizer.from_pretrained(plm_name)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(plm_name)
            
        if len(add_special_token) != 0:
             special_token = {
                  'additional_special_tokens': add_special_token
             }
             self.tokenizer.add_special_tokens(special_token)

        self.col_info = col_info
        self.max_length = max_length
        self.setup()

    def setup(self):
            self.train_dataset = getattr(dataset, self.dataset_name)(self.train_data, self.tokenizer, self.col_info, self.max_length)
            self.val_dataset = getattr(dataset, self.dataset_name)(self.dev_data, self.tokenizer, self.col_info, self.max_length)
            self.test_dataset = getattr(dataset, self.dataset_name)(self.test_data, self.tokenizer, self.col_info, self.max_length)
            self.predict_dataset = getattr(dataset, self.dataset_name)(self.predict_data, self.tokenizer, self.col_info, self.max_length)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size)