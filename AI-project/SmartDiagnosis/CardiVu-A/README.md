# CardiVu-A 연구 과정 (연구 기간: 약 3개월)

### 알코올과 빛에 의한 동공 반응의 관계가 있다는 생물학적 근거를 기반으로 연구를 진행했다.

- 알코올이 눈에 미치는 영향(체온상승, 동공 크기 확장, 경련, 건조, 빛에 대한 반응속도 감소)
- 동공 반응이 명확하게 나타나지 않아, 눈 깜박임이라는 조건을 추가하여 실험
- 동공 반응에 의한 홍채 근육 움직임을 가장 잘 표현하는 Optical Flow 함수를 선정.<br>

https://user-images.githubusercontent.com/48307173/189474498-082688a7-7bc1-41a2-8b61-85888276b21c.mp4

- 움직임을 가장 잘 표현한 DeepFlow 함수를 알코올 섭취 전, 후 빛에 대해 홍채 근육의 적응 시간을 보고자 실험을 진행했다.
![DeepFlow Eyeblinking](https://user-images.githubusercontent.com/48307173/189474780-525cf5eb-7330-4497-8bd0-05e7261affde.png)

[피험자 2명에 대한 알코올 섭취 여부에 따른 실험결과]
- 음주 전, 후 왼쪽 눈과 오른쪽 눈의 홍채 움직임을 평균, 표준편차, 분산, 수축 및 회복시간(Frame)을 지표로 평가했다.
- 동공 사이즈에 대한 평균, 분산, 표준편차에서는 알코올 여부에 따른 큰 차이를 보이는 특징은 보이지 않았다.
- 회복시간에서 차이를 보였다.
(알코올 전 LE: 평균 1.45초 RE: 평균 1.46초 / 알코올 후 LE: 평균 1.92초 RE: 평균 1.68초)
 => 0.2sec ~ 0.5sec 차이(60fps기준 12~30Frame의 차이를 보임)
![알코올 섭취 여부에 따른 동공 반응 시간 측정 결과](https://user-images.githubusercontent.com/48307173/189481358-39947466-a88b-4d9a-9cea-1582f584f7ee.png)


<img width="1000" alt="스크린샷 2022-09-10 오후 9 00 30" src="https://user-images.githubusercontent.com/48307173/189482338-0adfba48-ff02-4296-96bb-e8f6197e1616.png">
<img width="1000" alt="스크린샷 2022-09-10 오후 9 01 56" src="https://user-images.githubusercontent.com/48307173/189482385-d1c2b5c8-fd9c-4326-b3c9-618f4d077921.png">
<img width="1000" alt="스크린샷 2022-09-10 오후 9 04 18" src="https://user-images.githubusercontent.com/48307173/189482470-cf97b077-63ee-481f-ae94-fbf2029dd42e.png">
<img width="1000" alt="스크린샷 2022-09-10 오후 9 05 59" src="https://user-images.githubusercontent.com/48307173/189482524-df70f5b6-4782-46df-8076-faadb3b0c848.png">

