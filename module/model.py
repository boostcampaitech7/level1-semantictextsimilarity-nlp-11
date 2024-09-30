import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


class DefaultModel(nn.Module):
    def __init__(self, plm_name, add_special_token):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )
        if len(add_special_token) != 0:
            self.plm.resize_token_embeddings(self.plm.config.vocab_size+len(add_special_token))

    def forward(self, inputs):
        x = self.plm(**inputs)["logits"]
        return x
    

class ResizedTokenTypeEmbedsModel(nn.Module):
    def __init__(self, plm_name, add_special_token):
        super().__init__()
        self.plm_name = plm_name
        
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )

        """
        source token을 추가하여 token_type_ids를 0, 1이 아닌 0, 1, 2로 구성할 경우, token_type_embeddings의 차원을 그에 맞게 확장해주는 코드
        그런데 self.plm = transformers.{modelclass}.
        
        from_pretrained(..., type_vocab_size=3, ignore_mismathched_sizes=True)로 설정할 경우 임베딩 레이어가 무작위로 초기화되면서 학습 초기에 불안정한 경우가 발생하기 때문에,
        사전 학습 모델의 token_type_embeddings을 그대로 불러온 다음, 차원을 확장한 임베딩에 그 값을 옮겨 붙이는 방식으로 token_type_embeddings을 수정
        
        아래 코드는 plm을 Electra로 불러왔을 때 기준이므로 모델 클래스에 따라 적절히 수정해야함
        """
        old_emb1 = self.plm.electra.embeddings.token_type_embeddings.weight.data
        new_emb1 = torch.nn.Embedding(3, old_emb1.size(1))
        new_emb1.weight.data[1:] = old_emb1
        new_emb1.weight.data[0] = old_emb1[0]
        self.plm.config.type_vocab_size = 3
        self.plm.electra.embeddings.token_type_embeddings.weight.data = new_emb1.weight.data

        if len(add_special_token) != 0:
            self.plm.resize_token_embeddings(self.plm.config.vocab_size+len(add_special_token))

    def forward(self, inputs):
        x = self.plm(**inputs)["logits"]
        return x
    

class STSWithBinClsModel(nn.Module):
    """
    1단계: 3점 미만, 3점 이상으로 이진 분류, 2단계: 점수 예측하는 모델
    이 모델 사용시 반드시 STSWithBinClsDataset, CE_l2_loss와 함께 사용하고, trainer.py의 주석 확인
    """
    def __init__(self, plm_name, add_special_token):
        super().__init__()
        self.plm_name = plm_name
        # 0(0<=label<3), 1(3<=label<=5)인 binary label을 이진 분류하는 레이어
        self.bin_cls_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=2, use_auth_token=True
        )
        # 0(3<=label)인 데이터의 유사도 점수(label)를 예측하는 레이어
        self.sts_under3_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )
        # 1(3<=label)인 데이터의 유사도 점수(label)를 예측하는 레이어
        self.sts_over3_model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, num_labels=1, use_auth_token=True
        )

        if len(add_special_token) != 0:
            self.plm.resize_token_embeddings(self.plm.config.vocab_size+len(add_special_token))

    def forward(self, x):
        bin_logits = self.bin_cls_model(input_ids=x["input_ids"], token_type_ids=x["token_type_ids"], attention_mask=x["attention_mask"])["logits"]
        bin_probs = torch.softmax(bin_logits, dim=1)
        bin_preds = torch.argmax(bin_probs, dim=1)

        
        logits_under3 = self.sts_under3_model(input_ids=x["input_ids"], token_type_ids=x["token_type_ids"], attention_mask=x["attention_mask"])["logits"].squeeze()
        logits_over3 = self.sts_over3_model(input_ids=x["input_ids"], token_type_ids=x["token_type_ids"], attention_mask=x["attention_mask"])["logits"].squeeze()
        logits = torch.where(bin_preds == 0, logits_under3, logits_over3)

        return bin_logits, logits