# **문장 간 유사도 측정**

## 개요

1. 과제 소개

   본 프로젝트의 목표는 <span style="color:red; font-weight:bold">STS(Semantic Textual Similarity, 문맥적 유사도 측정)</span> 태스크를 수행하는 모델을 개발하는 것입니다.

   STS란 <U>두 문장이 담고 있는 의미가 얼마나 유사한지</U> 평가하는 작업으로, 본 프로젝트에서는 0 ~ 5 사이의 값으로 점수를 부여합니다. 예를 들어, 아래와 같이 두 문장이 서로 거의 비슷한 의미를 담고 있으면 높은 점수를 부여합니다.

    <div>

   | 문장 1                                                                                    | 문장 2                                                                                 | 유사도 점수 |
   | :---------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- | :---------- |
   | 진짜 최고의 명작이다                                                                      | 역시 여전히 진짜 명작이다.                                                             | 4.0         |
   | 소년법 폐지 절실합니다.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 소년법 폐지 강력하게 청원합니다.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; | 4.0         |

    </div>

   반면 아래와 같이 두 문장이 서로 다른 의미를 담고 있으면 낮은 점수를 부여합니다. 특히 일부 단어나 문장 구조가 비슷하다고 해도 문장의 의미가 다르면 유사도 점수는 낮게 평가됩니다.

    <div>

   | 문장 1                       | 문장 2                                 | 유사도 점수 |
   | :--------------------------- | :------------------------------------- | :---------- |
   | 그렇게 재미있진 않네요.      | 그런데 많이 재미있었습니다.            | 0.8         |
   | 유치하고 촌스러운 애니메이션 | 유일하게 질리지 않는 (영화)애니메이션. | 0.8         |

    </div>

   이러한 유사도 측정 작업은 번역 품질 평가(원본 문장 - 번역 문장의 유사도 평가), 질의 응답(입력 질문과 기존 질문 리스트 중 유사한 것을 찾아 적절한 답변 제공), 중복 문서 제거 등 자연어 처리의 다양한 분야에 응용될 수 있습니다.
   <br><br>

2. 평가 지표

   $$
   (\text{Pearson Correlation Coefficient})= {{\sum (y - \bar y)(t - \bar t)}\over \sqrt{(\sum{(y - \bar y)^2}} \sqrt{\sum{(t - \bar t)^2}}}
   $$

   본 대회에서는 <span style="color:red; font-weight:bold">피어슨 상관 계수</span>를 평가 지표로 사용합니다. 피어슨 상관 계수란, 두 변수의 선형 상관 관계를 계량화한 수치로, +1과 -1 사이의 값을 갖습니다. **두 변수가 강한 양의 선형 상관 관계를 가질수록 피어슨 상관 계수는 +1에 가까워지고**, 강한 음의 상관 관계를 가질수록 상관 계수는 -1에 가까워집니다. 본 프로젝트에서는 유사도 점수의 실제 값(label)과 모델의 예측값(target)이 양의 상관 관계를 가질수록 좋은(유사도 예측의 경향이 실제 값의 경향과 일치하는) 모델이 되는 것이므로, 피어슨 상관 계수를 +1에 가깝게 만드는 것이 목표가 됩니다.

   모델A가 모델B보다 피어슨 상관 계수가 높다는 것은, 모델B보다 <U>모델A의 예측값들의 전체적인 경향(증감율)이 실제 값들의 경향(증감율)과 더 비슷하다</U>는 것을 의미합니다. 단, 이것이 모델A가 모델B보다 실제 값에 가깝게 예측하는 것을 보장하지는 않으므로 해석에 주의가 필요합니다. (실제 값에 가깝게 예측하는지도 평가하고자 한다면 MSE 등 다른 평가 지표를 함께 사용하는 것이 좋습니다.)

## 프로젝트 진행

- 간단한 스케줄 표
- 계획
- 역할
  - AI 프로젝트 전체 구조에 대한 적응과 학습을 위해 팀원 전원이 end-to-end로 프로젝트 전 과정에 참여함
  - - 정완: 템플릿 제작

## 프로젝트 템플릿

- 템플릿 소개(소개글 요약)

## 데이터 분석

- 전체적인 특징(문장 스타일, 맞춤법, 띄어 쓰기 등)
- label별 / 소스별 분포
- train / dev 데이터 셋의 분포 비교
- label 별 데이터 특징
- …

## 학습 설계 및 실험 결과 분석

(볼드 항목은 학습 결과와 dev set 대상 예측 결과 분석 상세하게, 기타 항목은 간단하게 세줄 요약 정도만)

- 학습 결과 : val_pearson, dev data의 label-prediction 산점도
- **예측 결과 분석(데이터 증강은 따로)**
  1. [\*\*베이스라인 모델의 예측값](https://www.notion.so/Baseline-a313bf8efe7449c4b5e69643bcb5c4bf?pvs=21) vs 방법론(자신이 맡은 전처리 or 증강 방식) 적용 모델의 예측값 비교\*\*
     - x축: 레이블, y축: 각 모델의 예측값(색상 분리) 산점도
     - 예측 값 사이의 차이가 큰 데이터들에 대해,
       - 실제 데이터 행 표시(소스, 문장1, 문장2, label)
       - 각 모델의 해당 데이터에 대한 토크나이징 결과를 비교 분석하여 방법론이 실제로 어떤 영향을 미쳤는지 분석
       - 차이가 큰 데이터가 많다 ⇒ 방법론이 예측 값 변화에 영향을 많이 미쳤다
     1. 베이스라인 모델 예측값의 분포
     2. 방법론 적용 모델 예측값의 분포
     3. ~~예측값 사이의 차이를 그래프~~
  2. **베이스라인 모델의 예측값(pred1)과 참값(label)의 차이(err1) vs 방법론 적용 모델의 예측값(pred2)과 참값(label)의 차이(err2) 비교**
     - x축: 레이블, y축: |err1| - |err2| (=err_diff) 산점도
     - err_diff
       - err_diff > 0이면 pred2가 pred1보다 label에 가까워졌다(클수록 가까워진 정도가 큼) ⇒ 방법론 적용 시 성능 향상의 요인
       - err_diff < 0이면 pred2가 pred1보다 label에서 멀어졌다(작을수록 멀어진 정도가 큼) ⇒ 방법론 적용 시 성능 하락의 요인

### 사전 학습 모델 선택

- snunlp/KR-ELECTRA-discriminator
- monologg/koelectra-base-v3-discriminator
- kakaobank/kf-deberta-base

### 데이터 전처리

- 맞춤법 교정
- 띄어 쓰기 교정
- 영어 → 한글 변환
- 기타
  - 특수문자 제거

### 데이터 증강

- train dataset 전체 대상 swap 증강
  - 베이스라인 모델(Base), 증강 데이터로 학습한 모델(Swap)
  - 원본 dev set(dev_original), 원본을 스왑한 dev set(dev_swap)
  - Base의 dev_original에 대한 예측(pred_bo), Base의 dev_swap에 대한 예측(pred_bs),
  - Swap의 dev_original에 대한 예측(pred_so), Swap의 dev_swap에 대한 예측(pred_ss) 네 가지를 비교 분석
    1. pred_bo 대비 pred_bs의 분포가
       1. 비슷하다 ⇒ 스왑 증강한 데이터를 학습하지 않아도 문장 순서에 관계 없이 유사도 점수를 비슷하게 예측한다
       2. 다르다 ⇒ 스왑 증강한 데이터를 학습하지 않으면 문장 순서에 따라 점수 예측값이 달라질 수 있다
    2. pred_so 대비 pred_ss의 분포가
       1. 비슷하다 ⇒ 스왑 증강한 데이터를 학습하면 문장 순서에 관계 없이 유사도 점수를 비슷하게 예측한다
       2. 다르다 ⇒ 스왑 증강한 데이터를 학습해도 문장 순서에 따라 점수 예측값이 달라질 수 있다
    3. 기대하는 결과는 1-b & 2-a ⇒ 스왑 증강 데이터를 학습해서 문장 순서에 관계 없이 점수를 예측하는 모델이라면, 어떻게 나올지 모르는 임의의 테스트 데이터 셋에 대해서 안정적으로 점수를 예측해서 성능이 향상될 수 있을 것이다
       1. 실제 결과와 비교해서 기대 효과가 잘 반영되었는지, 아닌지를 판단
- train dataset의 label 0.5~3.5인 데이터 대상 swap 증강
- 기타
  - 0.5~5 swap + 0~3 중 1000개 뽑아서 5점 데이터 만들기

### 모델링

- 스페셜 토큰 추가
- 기타
  - segment embedding
  - 소스별 특화 모델 학습 후 결과 병합 (rtt, sampled / nsmc, petition, slack)
  - 이진 분류(0~3→0 / 3~5→1) 후 점수 예측 모델

## 최종 학습 결과(모델 선택)

- 위 실험 분석 결과, 방법론을 적절히 조합하여 리더 보드 제출에 사용할 모델 선정

- 앙상블

- 리더보드 제출 결과
