# TourismDataClassification

Main project1

<img src="https://media.discordapp.net/attachments/1002189622912221250/1060487598642044948/6b85bddd246b2388.JPG?width=868&height=676" width="680 " height="560">



목 록 | 설 명
------------|--------------------
[DataAugmentation.ipynb](https://github.com/Yeons2013/TourismDataClassification/blob/main/DataAugmentation.ipynb) | 데이터 증강 코드
[Image_RegNetY120.ipynb](https://github.com/Yeons2013/TourismDataClassification/blob/main/Image_RegNetY120.ipynb) | 이미지 모델
[Text_KlueRoBERTa_Large.ipynb](https://github.com/Yeons2013/TourismDataClassification/blob/main/Text_KlueRoBERTa_Large.ipynb) | 텍스트 딥러닝 모델
[Text_MechinLearning.ipynb](https://github.com/Yeons2013/TourismDataClassification/blob/main/Text_MechinLearning.ipynb) | 텍스트 ML 모델
[adjective.csv](https://github.com/Yeons2013/TourismDataClassification/blob/main/adjective.csv) | 증강용 형용사 사전
[adverb.csv](https://github.com/Yeons2013/TourismDataClassification/blob/main/adverb.csv) | 증강용 부사어 사전
[dacon.yaml](https://github.com/Yeons2013/TourismDataClassification/blob/main/dacon.yaml) | 사용한 가상 환경
[nsmc_stopwords.txt](https://github.com/Yeons2013/TourismDataClassification/blob/main/nsmc_stopwords.txt) | 불용어 사전
[sim_data.csv](https://github.com/Yeons2013/TourismDataClassification/blob/main/sim_data.csv) | 유의어 사전




##
### 2022 관광데이터 AI 경진대회(데이콘)

---

### 목   차
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065596852331880461/image.png?width=896&height=676" width="680 " height="500">


---
### 1. 프로젝트 기획

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065597597420621835/image.png?width=901&height=676" width="680 " height="500">

관광지 **이미지**와 설명 **텍스트**를 입력으로 넣어 어떤 관광지인지 **카테고리 예측** (소분류)

카테고리 분류를 인공지능의 힘으로 자동화 한다면, 더 적은 공공의 예산으로 더 많은 POI 데이터 생성 가능!


## 


<img src="https://media.discordapp.net/attachments/1022477080031666276/1065597663648677888/image.png?width=909&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065597722335379456/image.png?width=902&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065597783018578000/image.png?width=899&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065597855588417546/image.png?width=898&height=676" width="680 " height="500">


---
### 2. 전처리


<img src="https://media.discordapp.net/attachments/1022477080031666276/1065598628871278692/image.png?width=903&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065598715232014396/image.png?width=901&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065598784349941770/image.png?width=899&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065598848053018644/image.png?width=900&height=676" width="680 " height="500">

◆ 형용사 증강만 했을 때 성능이 향상되었던 이유는?
+ 한 문장내에 명사의 개수가 많기 때문에, 명사 앞에 랜덤으로 형용사를 삽입하는 경우 문장의 의미를 크게 손상시키지 않으면서, 유사도를 줄인채 증강을 할 수 있었기 때문으로 추정
+ 부사 증강의 경우 문장내에 동사가 적어 증강된 문장이 유사도가 너무 높고, 유의어나 Back Translation은 원래 문장의 의미를 훼손할 수 있기에 효과가 적은 것으로 판단

## 

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065598923097518080/image.png?width=901&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065598997349269635/image.png?width=900&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065599079704440842/image.png?width=902&height=676" width="680 " height="500">

---
### 3. 모델 학습

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065599932708421663/image.png?width=902&height=676" width="680 " height="500">

◆ Support Vector Mechine이 단일 ML 모델 중 가장 성능이 높았던 이유?
+ SVM은 범주를 예측하는데 사용이 가능하며, 오류 데이터의 영향이 적다. 또한 과적합 되는 경우가 적기 때문에 단일 모델 중에서 관광 데이터 분류와 잘 맞은 것으로 추정

## 

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065600034902642718/image.png?width=899&height=676" width="680 " height="500">

◆ RegNetY120?
+ 기존의 다양한 Networks 설정과 비교해서 성능이 뛰어나며, GPU 환경에서 빠른 속도를 보여주는 모델
+ 120층 모델을 사용한 것은 사용 가능한 GPU자원으로 감당 가능한 것이 120층이 최대였기 때문

## 

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065600155098808330/image.png?width=900&height=676" width="680 " height="500">

◆Klue/RoBERTa?
+ KLUE에서 학습시킨 BERT 계열 모델
+ 한국어로 사전학습이 매우 잘 되어 있어 여러 task에 fine-tuning하기 적합
+ 모델을 더 많은 데이터로 오래 그리고 더 큰 batch로 학습
+ 더 긴 문장들에 대해 학습
+ Mask를 dynamic하게 바꿔줌(epoch마다 중복되지 않도록)

## 

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065600237202325544/image.png?width=900&height=676" width="680 " height="500">

◆Multi-Modal은 데이콘의 HOJK님께서 공유해주신 코드를 참조함
+ [링크 연결](https://dacon.io/competitions/official/235978/codeshare/6861?page=1&dtype=recent)
+ Image Features와 Text Features를 횡방향으로 연결하는것이 종방향으로 연결할 때보다 좋은 성능을 보임

## 

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065600295402479666/image.png?width=901&height=676" width="680 " height="500">

---
### 4. 모델 핸들링 및 앙상블

<img src="https://media.discordapp.net/attachments/1022477080031666276/1065600827709997126/image.png?width=904&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065600907397582868/image.png?width=900&height=676" width="680 " height="500">


---
### 5. 결과 및 피드백
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065601222154919999/image.png?width=903&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065601274860540024/image.png?width=901&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065601347182923786/image.png?width=902&height=676" width="680 " height="500">
<img src="https://media.discordapp.net/attachments/1022477080031666276/1065601407018860655/image.png?width=900&height=676" width="680 " height="500">

