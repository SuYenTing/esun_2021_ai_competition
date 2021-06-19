# 玉山人工智慧挑戰賽2021夏季賽API主程式
from argparse import ArgumentParser
import base64
import datetime
import hashlib
import time

import cv2
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np

# 自製模型推論函數
from predict import predict

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'team_leader@gmail.com'  # 隊長信箱
SALT = 'StarRingChild'  # 加密用字串 隨意輸入即可 此處輸入隊名
#########################################


# 產生uuid函數 讓主辦單位可以辨識
def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


# 照片base64編碼轉OpenCV格式
def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    # 接收玉山發來的Request
    data = request.get_json(force=True)

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    # 產生隊伍的UUID
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL)

    # 模型預測
    try:
        answer = predict(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e

    # 紀錄圖片資訊
    cv2.imencode('.jpg', image)[1].tofile(f'./image/{answer}_{data["esun_uuid"]}_{data["esun_timestamp"]}.jpg')

    # 紀錄API回傳時間
    server_timestamp = time.time()
  
    # 提交API預測結果
    return jsonify({'esun_uuid': data['esun_uuid'],  # 玉山傳給使用者的任務ID
                    'server_uuid': server_uuid,  # 隊伍的UUID
                    'answer': answer,  # 預測的文字
                    'server_timestamp': server_timestamp})  # API回傳時間


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    options = arg_parser.parse_args()

    app.run(debug=options.debug, port=options.port)
