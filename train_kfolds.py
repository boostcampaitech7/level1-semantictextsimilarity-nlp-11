import os
import torch
import numpy as np
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
from module.trainer import KFoldTrainer
from omegaconf import OmegaConf
from utils.utils import *
import hydra
import wandb
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from module.data_loader import KFoldDataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):

    # 0. DictConfig to dict
    cfg.pwd = os.getcwd()
    config = OmegaConf.to_container(cfg, resolve=True)

    if config['wandb']['enable']:
        wandb.init(
            project=config["wandb"]["project_name"],
            name=f"plm-name={config['arch']['args']['plm_name']}+{config['arch']['type']}_aug=copy_swap-over0.5"
        )
        config['optimizer']['args']['lr'] = wandb.config['lr']
        config['data_module']['args']['batch_size'] = wandb.config['batch_size']
        config['run_name'] = \
        f"plm-name={config['arch']['args']['plm_name']}"
        wandb.run.name = config['run_name']
        wandb.run.save()


    # 1. K-fold Cross Validation 설정
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    train_df = pd.read_csv(config["data_module"]["args"]["train_path"])
    valid_df = pd.read_csv(config["data_module"]["args"]["dev_path"])
    test_df = pd.read_csv(config["data_module"]["args"]["predict_path"])
    df = pd.concat([train_df, valid_df])
    bin_labels = df['binary-label']

    # 2. K-fold 교차 검증 수행
    test_outputs = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, bin_labels)):
        if fold % 2: continue
        
        # 3. Train/Validation 데이터 분리
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # 4. set model(=nn.Module class)
        model = init_obj(config["arch"]["type"], config["arch"]["args"], module_arch)

        # 5. set deivce(cpu or gpu)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model = model.to(device)

        # 6. set loss function & matrics
        criterion = getattr(module_loss, config["loss"])
        metrics = [getattr(module_metric, met) for met in config["metrics"]]

        # 7. set optimizer & learning scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = init_obj(
            config["optimizer"]["type"],
            config["optimizer"]["args"],
            torch.optim,
            trainable_params,
        )
        lr_scheduler = init_obj(
            config["lr_scheduler"]["type"],
            config["lr_scheduler"]["args"],
            torch.optim.lr_scheduler,
            optimizer,
        )

        # 8. 데이터 로더 설정
        data_module = KFoldDataLoader(
            plm_name=config['arch']['args']['plm_name'],
            dataset_name=config['data_module']['args']['dataset_name'],
            batch_size=config['data_module']['args']['batch_size'],
            shuffle=config['data_module']['args']['shuffle'],
            train_data=train_df,
            dev_data=val_df,
            predict_data=test_df,
            col_info=config['data_module']['args']['col_info'],
            max_length=config['data_module']['args']['max_length']
        )

        # 9. 위에서 설정한 내용들을 trainer에 넣는다.
        trainer = KFoldTrainer(
            model,
            criterion,
            metrics,
            optimizer,
            config=config,
            device=device,
            data_module=data_module,
            lr_scheduler=lr_scheduler,
        )

        # 10. train
        output = trainer.train()
        if output is not None:
            test_outputs.append(output)   

    # 11. 테스트 아웃풋 저장
    avg_test_output = torch.mean(torch.stack(test_outputs), dim=0)
    
    pwd = os.getcwd()
    
    test_output_df = pd.read_csv(f'{pwd}/data/sample_submission.csv')
    test_output_df['target'] = avg_test_output.tolist()

    if not os.path.exists(f'{pwd}/output/'):
        os.makedirs(f'{pwd}/output/')
    folder_name = 'kfolds_test'
    folder_path = f'{pwd}/output/{folder_name}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    test_output_df.to_csv(folder_path + 'test_output.csv', index=False)



if __name__ == "__main__":
    # sweep_configuration = {
    #     "method" : "bayes",
    #     "metric" : {"goal" : "maximize", "name" : "pearson"},
    #     "parameters" : {
    #         "batch_size" : {"values": [16, 32]},
    #         "lr" : {"max": 0.0001, "min": 0.00001},
    #     },
    # }

    # sweep_id = wandb.sweep(sweep=sweep_configuration, project="")
    # wandb.agent(sweep_id, function=main, count=8)
    main()
