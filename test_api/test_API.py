# 測試API是否能夠正常運作
# 2021/05/06 蘇彥庭
import os
import time
import base64
import datetime
import requests

# API位置
apiUrl = 'http://127.0.0.1:8080/inference'  # Local端
# apiUrl = 'http://[API Server IP]:8080/inference'  # 雲端主機

# 整理測試照片路徑
testImgFilePath = './test_image'
images = [f'{testImgFilePath}/{elem}' for elem in os.listdir(testImgFilePath)]

# 讀取測試照片並轉為Base64格式
imagesBase64 = []
for image in images:
    with open(image, 'rb') as image_file:
        image_64_encoded = base64.b64encode(image_file.read()).decode("utf-8")
    imagesBase64.append(image_64_encoded)

# 迴圈傳出照片
requestTime = []  # 紀錄每次request時間
for image in imagesBase64:

    # 紀錄傳送時間
    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))

    # 模擬玉山request
    # 建立post參數資訊
    postData = {"esun_uuid": "399d69f7-8199-454e-bd79-c0c9571bc98b",
                "esun_timestamp": ts,
                "image": image,
                "retry": 1}

    # 執行request
    res = requests.post(url=apiUrl, json=postData)

    # 印出相關資訊
    print(f'request時間: {res.elapsed.total_seconds()}')
    print('回傳結果:')
    print(res.json())
    print('-' * 30)

    # 紀錄request時間
    requestTime.append(res.elapsed.total_seconds())

    # 暫停1秒
    time.sleep(1)

# 計算request時間
print(f'request回應平均時間: {sum(requestTime)/len(requestTime)}')
print(f'request回應最長時間: {max(requestTime)}')
print(f'request回應最短時間: {min(requestTime)}')

# 將request時間寫出csv檔案
import csv
with open('request_time.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for elem in requestTime:
        writer.writerow([elem])