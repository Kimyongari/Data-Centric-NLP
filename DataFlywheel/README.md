## Data Flywheel
> 데이터의 질을 올려 모델의 성능을 향상시키고, 향상된 성능의 모델을 통해 데이터의 질을 향상시키는 선순환 파이프라인 입니다.
위 과정 사이사이에 데이터 증강을 수행하여 데이터의 다양성을 증가시키고 라벨의 분포를 일정하게 유지합니다.   
   
## How to use

```python
from flywheel import flywheel

flywheel = flywheel() # flywheel 객체를 생성합니다. 
# 이 과정에서 데이터를 불러오고 Text abnormal data를 선별합니다.
flywheel.inference_and_cleanlab() # Text abnormal data가 있다면 Text 오염을 LLM을 통해 복구한 뒤 훈련합니다.
# Train data에 대한 Label issue를 찾아 데이터의 Label을 모델의 예측값으로 바꿉니다.
flywheel.make_new_data() # 주어진 데이터의 Label 분포를 분석하여 분포가 일정하도록 역번역, 문장 재구성, 키워드를 통한 문장생성 등의 증강을 수행합니다.
```

