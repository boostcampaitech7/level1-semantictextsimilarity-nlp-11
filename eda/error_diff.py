import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def error_diff(baseline_output, custom_output, data):
    baseline = pd.read_csv(baseline_output)['target']
    custom = pd.read_csv(custom_output)['target']
    label = pd.read_csv(data)['label']

    ### base 모델의 dev, swapped dev에 대한 예측값 분포 확인
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.subplots_adjust(wspace=0.22)


    # 그림1: base모델의 dev에 대한 예측값 분포
    x, y= label, baseline
    axes[0].scatter(x=x, y=y, s=4)
    axes[0].plot([min(x), max(x)], [min(x), max(x)], color='grey')
    axes[0].set_xlabel("Label")
    axes[0].set_ylabel("Prediction")
    axes[0].set_title("Baseline Model Prediction")

    # 그림2: base모델의 swapped dev에 대한 예측값 분포
    x, y= label, custom
    axes[1].scatter(x=x, y=y, s=4)
    axes[1].plot([min(x), max(x)], [min(x), max(x)], color='grey')
    axes[1].set_xlabel("Label")
    axes[1].set_ylabel("Prediction")
    axes[1].set_title("Custom Model Prediction")

    # 베이스 모델의 dev 데이터에 대한 예측값과 실제 label 사이의 오차
    error_base = (baseline - label).abs()
    # swap 증강 모델의 dev 데이터에 대한 예측값과 실제 label 사이의 오차
    error_swap = (custom - label).abs()

    # 오차 간의 차이
    error_gap = error_base - error_swap

    x1 = label.iloc[error_gap[(error_gap >= 0)].index]
    y1 = error_gap[(error_gap >= 0)]
    axes[2].scatter(x=x1, y=y1, s=4)

    x2 = label.iloc[error_gap[(error_gap < 0)].index]
    y2 = error_gap[(error_gap < 0)]
    axes[2].scatter(x=x2, y=y2, s=4)

    axes[2].plot([min(label), max(label)], [0, 0], color='grey', label='y=x Line')

    axes[2].set_ylim([-1, 1])
    # 구간 설정 및 각 구간별 데이터 갯수 계산
    bins = np.linspace(0, 5, 6)  # 0부터 5까지 5개의 구간으로 나누기
    positive_counts = np.histogram(x1, bins=bins)[0]
    negative_counts = np.histogram(x2, bins=bins)[0]

    # 각 구간의 중간 위치에 양수/음수 갯수 표시
    for i in range(len(bins)-1):
        bin_center = (bins[i] + bins[i+1]) / 2
        axes[2].text(bin_center, 0.9, f'+: {positive_counts[i]}', ha='center', fontsize=8, color='blue')
        axes[2].text(bin_center, -0.9, f'-: {negative_counts[i]}', ha='center', fontsize=8, color='red')

    axes[2].set_xlabel("Label")
    axes[2].set_ylabel("Error Difference")
    axes[2].set_title("Error Difference Between Baseline & Custom Model")

    plt.show()


if __name__=="__main__":
    baseline_output = '/Users/gj/Downloads/level1-semantictextsimilarity-nlp-11/output/plm=klue-roberta-small_val-pearson=0.85345/dev_output.csv'
    custom_output = '/Users/gj/Downloads/level1-semantictextsimilarity-nlp-11/output/plm=klue-roberta-small_val-pearson=0.85868/dev_output.csv'
    data_path = '/Users/gj/Downloads/level1-semantictextsimilarity-nlp-11/data/dev.csv'
    error_diff(baseline_output, custom_output, data_path)