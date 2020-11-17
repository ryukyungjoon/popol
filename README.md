# popol

<h3>INDEX</h2>

* [Deep-Learning_Project]
  * [Vector Data]
	* [CICIDS 2018]
		* [Feature Selection]
		* [Feature Correlation]
		* [DNN]
	* [NSL-KDD]
		* [AE]
		* [CNN]
		* [DNN]
		* [LSTM]
		* [SIAMESE]
  * [Image Data]
	* [MNIST]
		* [AE]
		* [SIAMESE]
	* [Brain tumor]
		* [CNN]

  <h3>Deep Learning Project</h3>
<p>CICIDS 데이터와 NSL-KDD 데이터는 데이터 불균형 문제와 희소 클래스 문제를 지니고 있다.
기계 학습에 있어서 이 두 가지 문제는 성능 저하의 원인이 된다.
이를 개선하기 위해 데이터 정제 및 전처리부터 학습 알고리즘을 다양하게 적용하는 등 여러 실험을 진행했다.
또한, 이미지 처리에 대한 개념을 익히기 위해 대표적인 이미지 처리 라이브러리 openCV를 활용한 Brain tumor 데이터를 활용한 학습을 수행했다.
</p>

    Vector Data
    
      - CICIDS 2018
      CICIDS 2018 데이터 셋은 네트워크 정상 및 비정상 트래픽 데이터로 이루어진 데이터 셋이다. 
      정상(BENIGN)과 비정상(Attack)으로 구성되어있고, 아래 표에서 확인 할 수 있듯이 각 클래스의 인스턴스의 극명한 차이를 보이는 불균형 데이터 셋이다.

<div>
  <img width="200" src="https://user-images.githubusercontent.com/48307173/99228641-3e146f80-2830-11eb-87dd-fbd214f2bd9d.png">
</div>

CICIDS 2018 데이터 셋의 성능 개선을 위한 실험 구조도

* [데이터 입력]-[데이터 정제]-[특징선택]-[정규화]-[신경망 훈련]-[분류]

<div>
<p>
<img width="1000" height="300"  src="https://user-images.githubusercontent.com/48307173/99352408-ecc6b780-28e5-11eb-82ad-9993fedf8763.png">
</p>
</div>


     - NSL-KDD
      NSL-KDD 데이터 셋은 네트워크 정상 및 비정상 트래픽 데이터로 이루어진 데이터 셋이다.
      KDD99 데이터 셋의 단점을 보완해서 재구성한 데이터 셋으로, 믾은 연구자들 사이에서 벤치마크 데이터 셋으로활용되고 있다.
<div>
  <img width="250" height="200" src="https://user-images.githubusercontent.com/48307173/99346887-044b7380-28d9-11eb-8fdc-0d390d5109ae.png">
  <img width="350" height="200" src="https://user-images.githubusercontent.com/48307173/99347139-a8cdb580-28d9-11eb-95f1-bd65f188a6ec.png">
</div>

    Image Data

      - MNIST
      인공지능 연구의 권위자 Yann LeCun 교수가 만든 MNIST 데이터는 0~9까지의 손글씨 이미지 데이터 셋이다.
      60,000 개의 훈련 데이터와 10,000개의 테스트 데이터로 데이터 이미지는 모두 28x28 픽셀의 크기로 이루어져있다.

<div>
  <img width="250" height="200" src="https://user-images.githubusercontent.com/48307173/99346174-5be8df80-28d7-11eb-9f63-2a6b6687ff0e.png">
 </div>
            
      - Brain tumor
      Brain tumor 데이터 셋은 이진 분류(yes or no) 데이터이다. 각기 다른 크기로 만들어진 이미지이다.
      [yes - 155장 || no - 98장]으로 이루어져있고, 훈련 데이터와 테스트 데이터가 따로 구성되어있지 않아서, 임의로 학습하기 전에 분류하는 작업이 필요한 데이터 셋이다.
<div>
  <img width="250" height="200" src="https://user-images.githubusercontent.com/48307173/99346076-1af0cb00-28d7-11eb-9c51-cb36ec81ce1b.jpg">
</div>
<br></br>
<h3>DATASET URL</h3>

	NSL-KDD : https://www.unb.ca/cic/datasets/nsl.html

	CICIDS 2018 : https://www.unb.ca/cic/datasets/ids-2018.html
	
	Brain tumor : https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
