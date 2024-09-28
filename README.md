# **문장 간 유사도 측정**

## 개요

1. 과제 소개

   본 프로젝트의 목표는 **STS(Semantic Textual Similarity, 문맥적 유사도 측정)** 태스크를 수행하는 모델을 개발하는 것입니다.

   STS란 <ins>두 문장이 담고 있는 의미가 얼마나 유사한지</ins> 평가하는 작업으로, 본 프로젝트에서는 0 ~ 5 사이의 값으로 점수를 부여합니다. 예를 들어, 아래와 같이 두 문장이 서로 거의 비슷한 의미를 담고 있으면 높은 점수를 부여합니다.

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

   본 대회에서는 **피어슨 상관 계수**를 평가 지표로 사용합니다. 피어슨 상관 계수란, 두 변수의 선형 상관 관계를 계량화한 수치로, +1과 -1 사이의 값을 갖습니다. **두 변수가 강한 양의 선형 상관 관계를 가질수록 피어슨 상관 계수는 +1에 가까워지고**, 강한 음의 상관 관계를 가질수록 상관 계수는 -1에 가까워집니다. 본 프로젝트에서는 유사도 점수의 실제 값(label)과 모델의 예측값(target)이 양의 상관 관계를 가질수록 좋은(유사도 예측의 경향이 실제 값의 경향과 일치하는) 모델이 되는 것이므로, 피어슨 상관 계수를 +1에 가깝게 만드는 것이 목표가 됩니다.

   모델A가 모델B보다 피어슨 상관 계수가 높다는 것은, 모델B보다 <ins>모델A의 예측값들의 전체적인 경향(증감율)이 실제 값들의 경향(증감율)과 더 비슷하다</ins>는 것을 의미합니다. 단, 이것이 모델A가 모델B보다 실제 값에 가깝게 예측하는 것을 보장하지는 않으므로 해석에 주의가 필요합니다. (실제 값에 가깝게 예측하는지도 평가하고자 한다면 MSE 등 다른 평가 지표를 함께 사용하는 것이 좋습니다.)

## 프로젝트 진행
<img width="900" alt="스크린샷 2024-09-27 13 15 37" src="https://github.com/user-attachments/assets/68c8d05c-281b-47f0-bcc8-5a180ff47440" />


## 프로젝트 템플릿
프로젝트를 진행할 때 여러 가지 모델, 전처리 방식, 하이퍼 파라미터 등을 조합하여 다양하게 실험해보는 것이 중요하다고 생각했습니다. [해당 탬플릿](https://github.com/GJ98/pytorch-template)은 모델, 데이터셋, 손실함수 등 학습 구성 요소를 별도 파일로 두어 코드 관리를 용이하게 해주고, 설정 파일 수정만으로 구현 클래스와 하이퍼파라미터를 쉽게 변경할 수 있도록 하여 다양한 실험을 효율적으로 수행할 수 있게 해줍니다.

## 데이터 분석
### 1. 데이터 라벨링 기준

**공식 라벨 분류 기준**

- *0점 : 두 문장의 핵심 내용이 동등하지 않고, 부가적인 내용에서도 공통점이 없음*
- *1점 : 두 문장의 핵심 내용은 동등하지 않지만, 비슷한 주제를 다루고 있음*
- *2점 : 두 문장의 핵심 내용은 동등하지 않지만, 몇 가지 부가적인 내용을 공유함)*
- *3점 : 두 문장의 핵심 내용은 대략적으로 동등하지만, 부가적인 내용에 무시하기 어려운 차이가 있음*
- *4점 : 두 문장의 핵심 내용이 동등하며, 부가적인 내용에서는 미미한 차이가 있음*
- *5점 : 두 문장의 핵심 내용이 동일하며, 부가적인 내용들도 동일함*

**라벨 분류 기준 재해석**

- 0점: 중요 단어/문구가 전혀 공유되어 있지 않음
- 1점: 중요 단어/문구 한개만 공유하고 나머지 중요 단어/문구는 서로 무관하거나 상반됨
- 2점: 중요 단어/문구 한개를 공유하고, 나머지 중요 단어/문구는 연관성이 있음
- 3점: 여러 중요 단어/문구를 공유하지만, 부사/부사구가 다름
- 4점: 여러 중요 단어/문구를 공유하고, 부사/부사구까지 동일함
- 5점: 감탄사 및 세부 표현까지 모두 동일함

중요 단어/문구는 **연관성**이 있음: 의미가 비슷한 중요 단어/문구를 가지고 있음. (e.g., 휴가/주말, 삭감/미지급, *즐길만함/재밌었던*)  
*중요 단어/문구를 **공유**함: 의미가 (거의) 동일한 중요 단어/문구를 서로 가지고 있음. (e.g., 봬요/만나요,  일회용품/일회용컵, 빵빵 터진다/진짜 재밌다)*  
분석: 0점과 5점은 학습이 수월해 보인다. 반면에, 2점과 3점 학습이 어려워 보인다. 연관성이 있는 단어(재밌다/좋다)와 의미가 동일한 단어(봬다/만나다)의 차이를 명확히 알아야 구분할 수 있기 때문이다.  
ㅇ  
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

**snunlp/KR-ELECTRA-discriminator**

**선택 이유:**

KR-ELECTRA는 34GB의 다양한 한국어 텍스트(위키피디아 문서, 뉴스 기사, 법률 텍스트, 제품 리뷰 등)를 기반으로 사전 학습되었습니다. 특히, 이 모델은 공식적이고 구조화된 텍스트뿐만 아니라 비공식적이고 다양한 문체의 텍스트에도 좋은 성능을 발휘합니다.

- **petition (국민청원 게시판 제목 데이터):** NSMC나 slack과 비교해 문법적으로 구조화된 문장들을 포함하고 있어 KR-ELECTRA의 사전 학습 데이터와 높은 연관성을 가집니다.
- **NSMC (네이버 영화 감성 분석 코퍼스):** 제품 리뷰와 유사한 데이터로, KR-ELECTRA가 학습한 리뷰 데이터가 영화 리뷰의 문장 간 의미 관계를 이해하는 데 도움이 된다고 판단하였습니다.

**성능 지표:**

- KorSTS (Spearman Correlation): **85.41**

**monologg/koelectra-base-v3-discriminator**

**선택 이유:**

KoELECTRA-base-v3는 20GB의 확장된 한국어 텍스트(신문, 위키, 나무위키, 메신저, 웹 등)를 포함하여 사전 학습되었습니다. 특히, v3 버전은 다양한 형태의 텍스트를 추가로 학습하여 비공식적이고 구어체 문장 처리에 강점을 가지고 있습니다.

- **petition (국민청원 게시판 제목 데이터):** 신문과 위키 등의 공식적인 문서가 포함된 사전 학습 데이터는 청원 제목의 구조화된 유사도 평가에 유리합니다.
- **NSMC (네이버 영화 감성 분석 코퍼스):** KoELECTRA-base-v3 사전 학습에 사용된 메신저, 웹 데이터는 영화 리뷰와 같이 다양한 주제와 문제 처리에 강점을 가집니다.
- **slack (업스테이지 슬랙 데이터):** 메신저와 웹 데이터를 포함한 비공식적이고 대화체 텍스트 학습 덕분에 슬랙 데이터의 대화체 문장 유사도 평가에 높은 성능을 기대할 수 있습니다.

**성능 지표:**

- KorSTS (Spearman Correlation): **85.53**

**kakaobank/kf-deberta-base**

**선택 이유:**

KF-DeBERTa는 DeBERTa-v2 아키텍처를 기반으로 하며, 범용 도메인 말뭉치와 금융 도메인 말뭉치를 함께 학습한 특화된 언어 모델입니다. 이 모델은 금융 관련 문장에서 높은 정확도를 가지지만, 범용 도메인에서도 금융 도메인과 마찬가지로 우수한 성능을 보이기 때문에 선택하게 되었습니다.

- **petition (국민청원 게시판 제목 데이터):** 청원 데이터는 비교적 문법에 따라 작성된 데이터로 범용 도메인 말뭉치 중 문법적으로 구조화된 데이터와 유사하여 좋은 성능을 기대할 수 있다고 생각해 선택하게 되었습니다.
- **NSMC (네이버 영화 감성 분석 코퍼스):** 범용 도메인 작업에서도 우수한 성능을 발휘하는 KF-DeBERTa는 영화 리뷰 문장 사이의 유사도 평가에 효과적이라고 보았습니다.
- **slack (업스테이지 슬랙 데이터):** 슬랙의 비공식적이고 대화체 문장은 범용 도메인 말뭉치와의 연관성이 높아, 다양한 문체 처리 성능이 슬랙 데이터의 유사도 평가에 기여할 것이라고 보았습니다.

**성능 지표:**

- KorSTS (Spearman Correlation): **85.99**

**team-lucid/deberta-v3-xlarge-korean**

**선택 이유:**
team-lucid/deberta-v3-xlarge-korean 모델은 현재 우리팀이 사용한 한국어 모델 중에서 가장 큰 사이즈(1.56 GB)를 가지고 있어서 복잡한 문장의 패턴과 미묘한 의미 차이를 포착하는 데 유리하다고 생각하였습니다. xlarge모델의 정확한 evaluation scores는 적혀있지 않아서 파악이 힘들지만 base 모델의 KorSTS 점수가 준수한 점을 들어서 모델을 선택하게 되었습니다.

성능 지표(DeBERTa-base):

- KorSTS (Spearman Correlation): 84.46

**요약:**

STS 과제의 데이터 소스(petition, NSMC, slack)와 각 모델에서 사전 학습 시 사용된 데이터가 높은 연관성을 가지며, KorSTS 평가에서도 높은 Spearman 상관 계수를 기록한 모델을 선택하였습니다.

- **snunlp/KR-ELECTRA-discriminator**는 다양한 공식 및 비공식 텍스트 처리에 강하며, 안정적인 성능을 보입니다.
- **monologg/koelectra-base-v3-discriminator**는 확장된 데이터로 비공식적 문장 처리에 유리하며, 유사한 높은 성능을 유지합니다.
- **kakaobank/kf-deberta-base**는 범용성과 금융 도메인 특화 능력을 겸비하여, KorSTS 평가에서 가장 높은 상관 계수를 기록함으로써 다양한 주제의 문장 쌍을 효과적으로 평가할 수 있습니다.
- **team-lucid/deberta-v3-xlarge-korean**은 가장 큰 모델 사이즈를 통해 복잡한 문장의 패턴과 미묘한 의미 차이를 효과적으로 포착할 수 있어, 문장 유사도 평가에서 높은 정확도와 정밀도를 기대할 수 있습니다.

### 데이터 전처리

1. **맞춤법 교정:** 한국어 맞춤법 교정 라이브러리인 `py-hanspell`을 사용하여 입력 데이터를 전처리 수행
    
    
    |  | **원본** | **py-hanspell 적용** |
    | --- | --- | --- |
    | 띄어쓰기 | 스릴도있고 반전도 있고 여느 한국영화 쓰레기들하고는 차원이 다르네요~ | 스릴도 있고 반전도 있고 여느 한국 영화 쓰레기들하고는 차원이 다르네요~ |
    | 맞춤법 교정 | 그 책부터 언능 꺼내봐야 겠어요! | 그 책부터 얼른 꺼내봐야겠어요! |
    | 오류 | 이건 진짜 대박임ㅇㅇ | 이건 진짜 대박 임용 |
    |  | 스우파 리정 vs 시미즈에서 리정 배틀곡이기도합니다 | 스투파 리 전 vs 시미즈에서 리 전 배를 곡이기도 합니다 |
    |  | 보육교사대 아동비율수 조정 청원합니다. | 보육교사다 아동 비율 수 조정 청원합니다. |
    |  | 결론은 완전 노잼. | 결론은 완전 나 잼. |
    |  | 와아아아안전 좋아요오오 | 와아아아 안전 좋아요 오 오 |
    |  | 추가로 스우파-스걸파-쇼미를 잇는 너무 즐거운 취향 공유 시간도 너무 즐거웠네요. | 추가로 스 우파-그걸 파-쇼미를 잇는 너무 즐거운 취향 공유 시간도 너무 즐거웠네요. |
    |  | 자유한국당 퇴출 시킨시다. | 자유한국당 퇴출 시킨 시다. |
    
    전반적으로 맞춤법 교정이 잘되고 있으나, 몇가지 케이스에서 원본 데이터 의미의 왜곡이 관찰되었다.
    
    - 실제 실험 결과(그래프)
        - snunlp/KR-ELECTRA-discriminator
        
        ![image](https://github.com/user-attachments/assets/903972b8-6d3f-4756-8e6f-d9a91c1b288b)
        
        - monologg/koelectra-base-v3-discriminator
        
        ![image](https://github.com/user-attachments/assets/72e91c80-fc35-44b8-a716-00d3043c8136)

        
        - kakaobank/kf-deberta-base
        
        ![image](https://github.com/user-attachments/assets/dd1502ab-39b9-4af7-b74b-25e4750cc7c8)

        
        - 해석
            
            그래프를 통해 맞춤법 교정이 일부 영역에서는 성능 향상을 가져오지만, **모든 경우에 긍정적인 결과를 주는 것은 아님**을 알 수 있습니다. 특히 **이모티콘(ㅇㅇ), 고유명사, 신조어, 비표준어, 구어체** 등에서 원본 문장의 의미를 왜곡하는 한계를 보였습니다. 이것은 인터넷에서 수집된 데이터의 특성을 처리하기에 맞춤법 교정이 한계가 있다는 의미입니다. 따라서 데이터의 특성을 고려하지 않은 일률적인 맞춤법 교정은 오히려 모델의 성능을 저해할 수 있습니다. 만약 맞춤법 교정을 사용해야 한다면 맞춤법 교정을 통해 전처리된 데이터를 한번 더 전처리하여 데이터의 특성을 더 잘 표현할 수 있도록 해야합니다.
          
- 띄어 쓰기 교정
- **영어 → 한글 변환**: `hangulize`를 이용해 영단어를 한글 발음으로 변환하는 작업을 수행하였습니다.
    - 예시
    
       | **원본** | **hangulize 적용** |
       | --- | --- |
       | 제가 있는 회의실의 **jabra** 마이크/스피커 들고 가신분?? | 제가 있는 회의실의 자바라 마이크/스피커 들고 가신분?? |
    - 기대 효과
        
        이 작업으로 기대한 바는 실제로 의미가 유사한 문장들의 유사도가 영단어와 한글 단어로 인해 생기는 차이를 줄여 문장들의 유사도를 좀 더 올바르게 판별해 모델 정확도가 높아지는 것입니다.
    - 실제 실험 결과

        ![image](https://github.com/user-attachments/assets/7e98125d-8f9f-4a6a-b48a-cbd7f7d5ed3e)

        ![image](https://github.com/user-attachments/assets/4c344600-c7c5-42fb-8e03-dc10cf3663ec)

        ![image](https://github.com/user-attachments/assets/b1822634-4787-408a-b19b-2796b3ba4982)

    
    - 해석
        
        베이스 라인 모델의 경우 데이터 포인트들이 대체로 대각선(1:1 선)을 따라 분포하고 있어, 예측값이 실제 값과 가까운 것을 보여줍니다. 방법론 적용 모델의 그래프 역시 데이터 포인트가 대각선 근처에 분포하고 있지만, 약간의 차이가 보입니다. 방법론 적용을 통해 예측 성능에 긍정적인 영향을 미친 것으로 볼 수 있었습니다. 두 모델의 오차를 비교한 그래프의 경우(각 행의 세번째 그래프), 대체로 파란색 점이 주황색 점보다 많지만 전반적으로 점들의 분포가 고르게 퍼져 있습니다. 이는 방법론이 베이스 라인 모델에 비해 긍정적인 영향을 미친 경우가 많지만, 특정 레이블에서는 여전히 성능 저하가 발생할 수 있음을 보여줍니다. 
        
        - 성능 저하 원인 데이터 예시
          
            | **원본** | **hangulize 적용** |
            | --- | --- |
            | **“github co-pilot**은 제가 깜빡하기 쉬운 **param**을 넣어주거나 **if** 같은것의 조건들을 다 체크해줘서 편한것 같습니다.”, “와우 **notion** 참 필요하다 싶은 기능들을 잘 제공해주면서 성장하는 것 같네요… 다른 기능들을 보니,,!” | **“기시브 코필로트**은 제가 깜빡하기 쉬운 **파람**을 넣어주거나 **이브** 같은것의 조건들을 다 체크해줘서 편한것 같습니다.”, “와우 **노티온** 참 필요하다 싶은 기능들을 잘 제공해주면서 성장하는 것 같네요… 다른 기능들을 보니,,!”|
- 기타
  - 특수문자 제거

### 데이터 증강

- train dataset 전체 대상 swap 증강
  - 기대 효과
    - `문장1, 문장2, 점수` 와 `문장2, 문장1, 점수` 데이터를 모두 학습함으로써 문장 순서에 관계 없이 유사도를 예측할 수 있는 방향으로 학습이 되기를 기대하였습니다.
- 실험 결과
    - snunlp/KR-ELECTRA-discriminator
    
    ![fig_snunlp_aug=fullswap](https://github.com/user-attachments/assets/6c37356a-b136-4d3d-89e4-e62f2ae20d62)
    
    - monologg/koelectra-base-v3-discriminator
    
    ![fig_monologg_aug=fullswap](https://github.com/user-attachments/assets/9aad08ce-c753-43b1-a21d-5e8b287bcac9)
    
    - kakaobank/kf-deberta-base
    
    ![fig_kakaobank_aug=fullswap](https://github.com/user-attachments/assets/fbdbad2d-b153-4c9f-8fef-87615ac21e9b)
    

- 해석
    - 세 모델 모두 swap 증강을 적용했을 때 예측 값 분포가 극적으로 개선되지는 않았으나, `val_pearson`은 소폭 상승하였습니다. 각 모델의 3번 그래프에서 예측이 label에 더 가까워진 데이터(파란색)가 멀어진 데이터(주황색)에 비해 약간 더 많은 것을 확인할 수 있습니다. 이는 swap 증강을 적용하여 문장 순서에 관계 없이 유사도를 측정함으로써 성능이 향상된 정도가 swap 증강의 부작용으로 같은 문장이 두 번씩 학습됨으로써 학습 데이터에 과적합 되어 성능이 하락한 정도보다 조금 더 크다고 해석하였습니다.
    실제로 아래 그래프와 같이 원본 dev dataset과 문장 순서를 바꾼 dataset에 대한 예측 값 차이를 살펴보면, 베이스 모델에서보다 증강 데이터로 학습한 모델에서 분산이 줄어들었음을 확인할 수 있습니다. (dev dataset의 문장 순서를 바꾼 데이터는 두 모델 모두 학습에 사용하지 않음) 즉, 증강 데이터로 학습한 모델은 문장 순서가 바뀌어도 비슷한 점수를 예측할 가능성이 높아졌을 것이라고 판단했습니다.
        
        ![pred_diff_dist](https://github.com/user-attachments/assets/9c2dbf7e-042e-40f5-b408-6f8cbec29aa1)
        
- train dataset의 label 0.5~3.5인 데이터 대상 swap 증강
    - 기대 효과
       - 데이터 분석 결과, 2\~3점 구간의 데이터 학습이 어려울 것이라 판단했고, 실제로 해당 구간에서 예측이 안되는 모습을 관찰할 수 있습니다. 이를 개선하기 위해 label 0.5\~3.5점 구간의 데이터를 swap 증강을 하여 학습하면 2~3점 구간에서의 예측 오차를 감소시킬 수 있다고 판단했습니다.
- 실험 결과
     - snunlp/KR-ELECTRA-discriminator
  
     ![2_11](https://github.com/user-attachments/assets/901fe092-c356-4e2a-bdca-2a80157dd52c)
        
     - monologg/koelectra-base-v3-discriminator
  
     ![1_11](https://github.com/user-attachments/assets/f1d681f0-7ec3-42e7-8ee0-cc1f94db8e49)
        
     - kakaobank/kf-deberta-base
  
     ![0_11](https://github.com/user-attachments/assets/62324464-31d5-4c59-9312-bf26232a2dce)
        
- 해석
  - label 0.5\~3.5점 구간의 데이터 swap 증강을 적용해본 결과, 세 모델 모두 label 2\~3점 구간의 데이터에서 예측값이 라벨값에서 멀어진 경우(주황색)보다 가까워진 경우(파란색)가 더 많음을 알 수 있습니다. 예측값에 더 가까워진 이유는 더 많은 데이터로 해당 구간을 학습했기 때문입니다. 반대로, 예측값에 더 멀어진 이유는 증강 데이터가 기존 데이터와 거의 동일한 데이터이기 때문에 모델이 baseline 모델보다 학습 데이터에 과적합 되어 발생한 것이라고 생각합니다. 결론, label 0.5\~3.5 구간의 데이터 swap 증강을 적용하면 학습 데이터의 특정 부분에서 과적합이 발생할 수 있지만, 기존에 학습이 어려웠던 부분인 label 2\~3점 구간에서의 성능 향상을 더 크게 기대할 수 있습니다.

|  | snunlp | monologg | kakaobank |
| --- | --- | --- | --- |
| 가까워진 경우 | 70(+20) | 70(+20) | 60(+10) |
| 멀어진 경우 | 40 | 40 | 50 |

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
