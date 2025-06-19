# TFT_MK

Temporal Fusion Transformer를 활용한 MK2000 금융지수 시계열 예측
한국 금융시장 MK2000 지수에 대해 Temporal Fusion Transformer(TFT) 딥러닝 모델을 적용하여 다중 시계열 예측(Multi-horizon forecasting) 및 변수 중요도 해석을 수행

1. 개요
분석 목적:

최신 딥러닝(TFT) 기반 시계열 예측 모형의 금융지수 예측 성능 평가

각종 외생변수(금리, 환율, 변동성 등)와 시계열 변수들의 예측력 및 중요도 해석

데이터:

MK2000 지수. DataGuide 제공

일간/월간 종가, 변동성, 금리, 환율 등

입력: 시계열 특성변수, 외생변수, 달력/계절성 변수 등

모델:

PyTorch Forecasting 기반 Temporal Fusion Transformer (TFT)

Encoder-Decoder 구조, Multi-horizon output, Attention/Variable Importance 지원

2. 코드 구조 및 주요 실행 흐름
(1) Temporal_fusion_Transformer.ipynb
데이터 로딩/전처리:

MK2000(또는 예제) 지수 불러오기

결측값 처리, 시계열 feature engineering(이동평균, 변동성 등), 계절성 변수 생성(day_of_week, month 등)

TFT용 Dataset 구축:

TimeSeriesDataSet 객체 정의

target, known/unknown, real/categorical feature 구분

max_encoder_length, prediction_length 등 하이퍼파라미터 설정

모델 학습/예측:

TemporalFusionTransformer 모델 선언, hyperparameter 지정

Trainer 기반 모델 학습, validation/test 예측

예측 결과 시각화(실제값 vs 예측값, 구간별 plot), MSE/MAE 등 metric 산출

해석:

Variable Importance(encoder/decoder별), Attention 분포

계절성, exogenous variable 효과 등 시각화 및 해석

(2) TFT_with_MK2000.ipynb
데이터:

MK2000 실제 데이터(은행, 보험, 증권, 기타, macro/exogenous 포함)

분석 구조:

위와 동일한 TFT 프로세스 적용

다양한 예측 구간/target, 변수 조합 테스트

예측 성능, 변수별 영향, attention 등 실증적 해석

결과:

실제 금융지수 예측에서 TFT의 성능 확인

주요 변수(자기지수, 이동평균, 변동성, 금리/환율 등) 효과 계량

구간별/섹터별 예측력 차이 실증

3. 주요 결과 요약
예측 성능

MK2000 하위지수(은행, 보험, 증권 등) 장·단기 예측에서 TFT가 계절성, 추세, 외생변수 반영에 강점

급격한 구조변동/충격 구간 제외 시, MSE/MAE 등 오류율 우수

변수 중요도(Variable Importance)

자기지수, 이동평균, 변동성 등 시계열 주요 feature가 가장 큰 영향

외생변수(금리, 환율) 영향력도 시계열적 상황에 따라 유의미

Encoder/Decoder, Attention 분포로 시계열별/구간별 영향 차이 분석

해석/활용

딥러닝 기반 모델로도 해석가능성 확보(Variable Importance, Attention 등)

금융시장 예측, 리스크관리, 투자전략 시나리오 분석 등 실무 적용 가능

4. 실행 환경
Python 3.8+

PyTorch, pytorch-forecasting, pandas, numpy, matplotlib 등

Colab 환경에서 실행

5. 참고
PyTorch Forecasting 공식문서

DataGuide/FnGuide MK2000 지수

예측모델(SCINet, MTST)과 성능 비교는 별도 branch/노트북 참고

참고 문헌: Lim, Bryan, et al. "Temporal fusion transformers for interpretable multi-horizon time series forecasting." International Journal of Forecasting 37.4 (2021): 1748-1764.
