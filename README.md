# 玉山人工智慧挑戰賽2021夏季賽-中文手寫影像辨識

## 競賽說明

請參閱[T-Brain官方網站-玉山人工智慧挑戰賽2021夏季賽](https://tbrain.trendmicro.com.tw/Competitions/Details/14)

## 競賽成果

總排名為第20名，準確率為93.11%(第1名準確為97.12%)，總參賽隊伍數468隊，Top排名比率為4.27%

## 檔案說明

### 1. API Server部署檔案(esun_api_docker)

參考玉山官方提供的[API範例](https://github.com/Esun-DF/ai_competition_api_sharedoc)進行修改，並包裝成Docker方便快速部署在雲端

### 2. API Server測試檔案(test_api)

此程式碼主要模擬玉山發出的Request，藉此來測試API是否能夠順利運作，並紀錄API SeverResponse的速度

### 3. 模型訓練程式碼(esun_model_colab.ipynb)

此為模型訓練程式碼，主要以ResNet18模型為主架構進行模型訓練

* [[點我直接在Colab開啟]](https://colab.research.google.com/drive/1G0FEw-FXYtokg7RXzDruTNdHhFS8qY0v?usp=sharing)









