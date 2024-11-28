# **1. Overview**

**🚩 Data-Centric NLP (Topic Classification) Project**

뉴스의 `헤드라인`으로 뉴스가 어떤 `topic`을 갖는지 `분류`하는 Project입니다.

> 🔥 Caution
> 
> - Data-Centric의 취지에 맞게, 베이스라인 모델은 수정 불가
> - Only 데이터의 수정으로만 성능 향상을 이끌어내야 함

**데이터 구성**

![스크린샷 2024-11-28 오후 7.35.00.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/9c69e9f7-0e56-42a8-9a97-bf59a84d1cf6/1d5ee642-ec60-42d8-8e4e-178c74b7de7a/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-28_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.35.00.png)

| 가능한 방법 | 불가능한 방법 |
| --- | --- |
| - 공개된 생성 모델 (T5 등)을 통한 Synthetic Data 생성
- 각종 Data Augmentation 기법 적용- Data sampling 기법- Negative sampling 등
- Data Quality Control- Data labeling error detection, Data Cleaning, Data Filtering 등
- 다양한 Self-training 기법 (코드가 아닌 데이터 변경)- 모델의 변경이 아닌 베이스라인 코드의 변경 | - 유료 버전의 비공개 생성 모델을 활용하는 모든 방법(GPT-4, ChatGPT (GPT-3.5), api key 결제 일체 포함)
- 베이스라인 코드 내 모델의 변경을 야기하는 모든 방법들
- 테스트셋 정보를 활용한 모든 방법
- 외부 데이터셋 사용- Active learning, Curriculum learning 등의 데이터 수정이아닌 모델을 업데이트 하는 방법 |

**모델 성능 평가 지표(Metrics)**

- Accuracy(정확도) : TP+TN / TP+FP+TN+FN
- F1 score : 2 * Precision * Recall / Precision + Recall

# **2. 프로젝트 구성**

**⏰ 개발 기간**

- 2024년 10월 28일(월) 10:00 ~ 2024년 11월 7일(목) 19:00
- 부스트캠프 AI Tech NLP 트랙 11-12주차
    
    
    | Title | Period | Days | Description |
    | --- | --- | --- | --- |
    | 필요 지식 학습 | 10.28 ~ 10.31 | 4 days | Data-centric NLP에 대해 이해 |
    | 데이터 분석 및 EDA | 10.31 ~ 11.01 | 2 days | 데이터 구조 및 프로젝트 내용 분석 |
    | 데이터 노이즈 제거 | 11.01 ~ 11.03 | 3 days | 데이터 노이즈 제거 방법 개선 |
    | 데이터 증강 및 고도화 | 11.03 ~ 11.07 | 4 days | 실험 및 성능 개선 |

**✨ 분석 환경**

- Upstage AI Stages 제공 V100 GPU Server 활용
- OS : Linux
- Language : Python
- Libraries(mainly used) : Pytorch, Hugging Face, Wandb etc.

**💾 Data**

- 결측치와 중복 값이 없는 Train, Test 데이터

|  | Samples | Description |
| --- | --- | --- |
| Train | 2,800 | ID, text, target으로 이루어진 학습 데이터 |
| Test | 30,000 | ID, text로 이루어진 평가 데이터 |

# **3. 수행 내용**

**1. 데이터 분리 Task**

1. Text 내 영어, 숫자, 특수문자를 기준으로 Noise 데이터 1차 분류
2. Rule Base로 Noise 데이터 2차 분류

**Noise Patterns**

```
abnormal_patterns = [
        r'[A-Za-z]{3,}[0-9]{3,}', # 영어 3자 이상 + 숫자 3자 이상 (너무 짧은 패턴 제외)
        r'[0-9]{3,}[A-Za-z]{3,}', # 숫자 3자 이상 + 영어 3자 이상
        r'[A-Za-z]{2,}[^A-Za-z0-9가-힣\s]{2,}', # 영어 2자 이상 + 특수문자 2자 이상
        r'[^A-Za-z0-9가-힣\s]{2,}[A-Za-z]{2,}', # 특수문자 2자 이상 + 영어 2자 이상
        r'[A-Za-z]{2,}-[0-9]+-[A-Za-z]{2,}', # 영어 2자 이상 - 숫자 - 영어 2자 이상
        r'[^A-Za-z0-9가-힣\s]{3,}',    # 연속된 특수 문자 3자 이상
        r'^(?=[^{}\(\*\[\]]*[}\(\*\[][^{}\(\*\[\]]*$)',# 특정 기호가 하나만 포함된 문자열 필터링
        r'[A-Za-z]{1,2}[^A-Za-z0-9가-힣\s]+[가-힣]',  # 영어 1~2자 + 특수문자 + 한글
        r'[가-힣]+[^A-Za-z0-9가-힣\s]+[A-Za-z]{1,2}',  # 한글 + 특수문자 + 영어 1~2자
        r'[가-힣]+[^A-Za-z0-9가-힣\s]+[가-힣]+[A-Za-z]',  # 한글 + 특수문자 + 한글 + 영어  
        r'[^A-Za-z0-9가-힣\s]+[0-9]+[^A-Za-z0-9가-힣\s]+' # 특수문자 + 숫자 + 특수문자
      ]
```

**2. 데이터 노이즈 복원**

- 정상 데이터를 랜덤하게 임의의 ASCII코드로 대체한 데이터 구성
- LLama 3.1에 ASCII로 오염된 데이터를 정상 데이터로 복구하도록 Finetuning
훈련된 모델을 데이터 복구에 활용
- 복구되는 과정에서 분류가 잘못될 수 있음 (ex. 야구팀 롯데를 회사 롯데로 판단하여 복구해 스포츠에서 경제 기사로 변경)

**3. 데이터 증강**

- BackTranslation (중간 언어 : 영어, 일본어, 독일어 등)
- LLM을 활용한 문장 재구성, 오염된 문장 복원, Topic Clustering을 통해 추출한 Keyword 기반 문장 생성 (Llama 3.1 활용)

**4. 데이터 순환구조 구축(Data Flywheel)**

1. 데이터가 주어졌을 때 Regular Expression을 통해 Text가 오염된 데이터와 오염되지 않은 데이터를 분류
2. Text가 오염된 데이터는 LLM을 통해 복구 후 Label에 이상이 있는 데이터와 합쳐 Train을 수행
3. 학습된 모델로 전체 데이터셋을 Cleanlab을 통해 라벨 이상치가 있는 데이터를 분류
4. Label issue로 색출된 데이터는 주어진 Label이 아닌 모델이 예측한 Label로 수정
5. 모델의 최대 Logit값이 낮은 (확신이 없는)데이터는 LLM을 활용하여 비슷한 문장을 추가 생성
6. 텍스트가 정상으로 분류된 데이터는 LLM을 활용한 문장 재구성 및 역번역으로 데이터 증강
7. Label 분포를 일정하게 형성하기 위해 증강된 데이터들 중 Label 수가 많은 데이터는 삭제

## ✓ 4.  결과 (최종 Leader Board 6위)

|  | Accuracy | F1 |
| --- | --- | --- |
| Baseline | 0.5937 | 0.5621 |
| DataFlywheel 4 Cycle | 0.8523 | 0.8481 |
| LLM Pretrained | 0.8519 | 0.8480 |
- 오염된 문장을 Pretrained LLM 보다, 직접 정상-오염 Pair dataset 구축하여 Finetuning하였을 때 훨씬 잘 복구되었음
- Data Flywheel을 통해 점진적으로 성능이 개선되었지만, 일정횟수의 Cycle을 반복하면 더 개선되지 않았음

## 5. Team
<table>
    <tbody>
        <tr>
            <td align="center">
                <a href="https://github.com/Kimyongari">
                    <img src="https://github.com/Kimyongari.png" width="100px;" alt=""/><br />
                    <sub><b>Kimyongari</b></sub>
                </a><br />
                <sub>김용준</sub>
            </td>
            <td align="center">
                <a href="https://github.com/son0179">
                    <img src="https://github.com/son0179.png" width="100px;" alt=""/><br />
                    <sub><b>son0179</b></sub>
                </a><br />
                <sub>손익준</sub>
            </td>
            <td align="center">
                <a href="https://github.com/P-oong">
                    <img src="https://github.com/P-oong.png" width="100px;" alt=""/><br />
                    <sub><b>P-oong</b></sub>
                </a><br />
                <sub>이현풍</sub>
            </td>
            <td align="center">
                <a href="https://github.com/Aitoast">
                    <img src="https://github.com/Aitoast.png" width="100px;" alt=""/><br />
                    <sub><b>Aitoast</b></sub>
                </a><br />
                <sub>정석현</sub>
            </td>
            <td align="center">
                <a href="https://github.com/uzlnee">
                    <img src="https://github.com/uzlnee.png" width="100px;" alt=""/><br />
                    <sub><b>uzlnee</b></sub>
                </a><br />
                <sub>정유진</sub>
            </td>
            <td align="center">
                <a href="https://github.com/hayoung180">
                    <img src="https://github.com/hayoung180.png" width="100px;" alt=""/><br />
                    <sub><b>hayoung180</b></sub>
                </a><br />
                <sub>정하영</sub>
            </td>
        </tr>
    </tbody>
</table>

<br><br>

---

<br>
