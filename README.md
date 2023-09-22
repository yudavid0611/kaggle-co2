# [Kaggle Competition] Predict CO2 Emissions in Rwanda
## 소개
- [Competition Link](https://www.kaggle.com/competitions/playground-series-s3e20)
- **기간**: 23.08.02 - 23.08.12
- **팀 구성**: 1인
- **목적**: Kaggle Learn에서 배운 내용을 활용하여 작업하기
- **특징**
    1. 프로세스: 결측값 처리 -> Feature engineering -> Feature selection -> Modeling -> 예측 -> 완료
    2. 각 프로세스별 최선의 선택지를 `score_dataset` 함수를 통해 선택(ex. 결측값 처리 시 평균으로 대체할지, 선형보간법을 사용할지)
    3. EDA에서 target 변수의 값이 계절에 따라 다르게 움직이는 것을 확인하여 모델 또한 계절별로 제작
- **결과**
    1. 업로드한 코드로 `브론즈 메달` 획득([링크](https://www.kaggle.com/code/yudavid/eng-kor-from-preprocessing-to-modeling))
    2. 공식 기록 13.66073(1위 기록: 9.33307) -> `상위 18%`

## Main 코드
- [코드 바로가기](https://github.com/yudavid0611/carbon/blob/master/main.ipynb)