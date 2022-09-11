# CardiVu-D 행태 데이터 추출

- 스마트폰을 활용하여 사용자의 행태 데이터를 수집한다.
- 환자군/대조군을 모집하여 실험을 진행했고, 2주간의 데이터로 우울증 환자를 판별하는 모델을 구성하고자 했다.
- Life_logging.py 는 행태 데이터를 활용해 Location_Variance, Energy, Circadian_Movement, Total Distance, Transition Time을 추출한다.

=> 추출된 행태 데이터와 CardiVu를 통해 얻어진 홍채 데이터를 통해 우울증 환자를 판별하는 모델을 만들고자 했으나, 다양한 환자군 모집에 어려움이 있어 진행 중이지만 데이터 수집이 완료되지 않아서 모델링을 진행하고 있지 못한 상태이다.

- 참조논문
1) SAEB, Sohrab, et al. Mobile phone sensor correlates of depressive symptom severity in daily-life behavior: an exploratory study. Journal of medical Internet research, 2015, 17.7: e4273.
2) SAEB, Sohrab, et al. The relationship between mobile phone location sensor data and depressive symptom severity. PeerJ, 4, e2537. 2016.
