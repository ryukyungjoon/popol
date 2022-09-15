# 기존 Cardivu를 통해 얻어진 데이터 정제

- Noise Remove 폴더의 사진은 구간내로 예측된 움직임 데이터를 Reference로 사용하여 구간 외 예측된 데이터를 정제하고자 노력한 결과들이다.
- Super Resolution 폴더는 image_data 폴더의 사진과 영상 데이터를 통해 홍채 미세근육 움직임 데이터 값을 추출하는 알고리즘인 Optical Flow의 성능 검증을 위한 과정이다.

# CardiVu preprocessing project

- 홍채 미세근육 움직임 데이터를 얻기 위한 과정은 그림과 같다.
<br>
<img width="900" alt="홍채 미세근육 데이터 추출과정" src="https://user-images.githubusercontent.com/48307173/189474251-759df152-aace-4f95-8b75-d556e28e1337.png">
<br>
- main.py를 실행하면 전처리 프로젝트를 실행한다.<br>
- 영상 파일 선택, 영상처리 함수 Threshold 설정한다.<br>
- 해상도가 낮은 영상을 매 프레임마다 Super Resolution(초해상화) 과정을 통해 고해상도로 높인다.
<br>
[OpenCV 영상처리 보간법 함수를 이용해서 사이즈 늘린 사진]
<img width="334" alt="image" src="https://user-images.githubusercontent.com/48307173/189480697-b4886b7d-2d52-4a0e-a04e-e8af61789c07.png">
[Super Resolution을 이용해서 사이즈 늘린 사진]
<img width="334" alt="image" src="https://user-images.githubusercontent.com/48307173/189480701-b0b76678-8995-44b8-bb4c-3dfb7c4ed841.png">

[OpenCV 영상처리 보간법 함수를 이용해서 사이즈 늘린 사진]<br>
<img width="138" alt="image" src="https://user-images.githubusercontent.com/48307173/189480613-7745f65e-0762-4037-a77c-8bb9695471f9.png">
[Super Resolution을 이용해서 사이즈 늘린 사진]<br>
<img width="138" alt="image" src="https://user-images.githubusercontent.com/48307173/189480630-5bf55eba-aecb-4f4d-893a-47fd00a7f575.png">


- 각 프레임마다 영상처리하고 홍채미세근육 움직임 데이터를 추출한다.<br>
- 영상마다 움직임 데이터를 저장한다.<br>
