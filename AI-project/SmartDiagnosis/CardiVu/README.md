# CardiVu 연구과정

### CardiVu 성능 개선 연구
1) 학습 모델 개선
2) 신호 데이터 resampling & preprocessing
3) 홍채 영역 재정의
4) OpenCV를 이용한 홍채 미세 근육 데이터 추출 알고리즘 비교 연구
5) IR 카메라를 이용한 홍채 미세 근육 움직임 데이터와 심박수 및 심박변이도에 상관성을 띄는 변수 재정의
6) 재정의된 변수로 구성된 데이터 preprocessing & resampling

### 식약처에 제출한 연구 결과 자료 제출
<span><맥박수 측정 범위에 따른 결과></span>
<img width="918" alt="스크린샷 2022-09-09 오후 5 01 10" src="https://user-images.githubusercontent.com/48307173/189302159-9b62080f-60e2-4595-9fdd-357a964f704d.png">
    <br><span><전체 맥박수 구간 데이터의 Bland-Altman plot></span><br>
    <img width="400" alt="스크린샷 2022-09-09 오후 5 03 44" src="https://user-images.githubusercontent.com/48307173/189302445-85cbc236-0be0-4467-852b-e2724c385414.png">
    <br><span><맥박수 90BPM 이상 데이터의 Bland-Altman Plot></span><br>
    <img width="400" alt="스크린샷 2022-09-09 오후 5 06 18" src="https://user-images.githubusercontent.com/48307173/189302880-0dc0633c-fc19-4422-b2c6-0ed247a6e34c.png">
    <br><span><맥박수 65BPM 이상 90BPM 미만 데이터의 Bland-Altman Plot></span><br>
    <img width="400" alt="스크린샷 2022-09-09 오후 5 07 18" src="https://user-images.githubusercontent.com/48307173/189303053-f171e11a-302a-4dab-9afb-5652db0b43c6.png">
    <br><span><맥박수 65BPM 미만 데이터의 Bland-Altman Plot></span><br>
    <img width="400" alt="스크린샷 2022-09-09 오후 5 08 19" src="https://user-images.githubusercontent.com/48307173/189303228-b8bc9945-8610-4adf-ba65-d03ee39f2736.png">
<hr>
  <span><b>The Comparison for correlation(Sensor BPM - Cardivu BPM)</b>
    <br>
    : 센서를 통해 측정된 맥박수 BPM과 Cardivu를 통해 측정된 맥박수 BPM의 상관관계 
  </span>
<br><br><span><전체 맥박수 구간 데이터의 상관관계 비교></span><br>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/48307173/189303609-02ff78a7-a62c-4e8b-9995-3ba3c9a39411.png">
<br><span><맥박수 90BPM 이상 데이터의 상관관계 비교></span><br>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/48307173/189303855-c3bb3d93-0823-473c-bb7b-979e620b106c.png">
<br><span><맥박수 65BPM이상 90BPM 미만 데이터의 상관관계 비교></span><br>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/48307173/189303900-3632c076-35c5-4b33-915e-64390869353a.png">
<br><span><맥박수 65BPM 미만 데이터의 상관관계 비교></span><br>
<img width="400" alt="image" src="https://user-images.githubusercontent.com/48307173/189303961-2dc94220-0857-4c2f-9ac5-2d8c0352ebb0.png">
<br><span><맥박수 계산에 사용된 모델의 성능을 확인할 수 있는 근거자료(모델 학습 시 활용한 검증 데이터(validation data)를 통해 얻은 결과)></span><br>
<img width="695" alt="스크린샷 2022-09-09 오후 5 19 52" src="https://user-images.githubusercontent.com/48307173/189305380-bfb9ecaf-74ae-4b4a-b119-fa8f85d08858.png">
