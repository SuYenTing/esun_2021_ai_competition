# 玉山人工智慧挑戰賽2021夏季賽API推論函數
import cv2
from PIL import Image
import pickle
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from torchvision import models, transforms
from albumentations.pytorch import ToTensorV2


############### 前置作業(Yesting) ###############
# 讀取字對應編號資料
with open('vocab2idx.pkl', 'rb') as handle:
    vocab2idx = pickle.load(handle)
    wordClassDict = {k: v for k, v in vocab2idx.items()}

# 分類數量
num_classes = len(list(wordClassDict.keys()))

# 讀取模型架構
yesting_model = models.resnet18(pretrained=False)
num_ftrs = yesting_model.fc.in_features
yesting_model.fc = nn.Linear(num_ftrs, num_classes)

input_size = 224

# 讀取已訓練好的模型權重
modelWeightFilePath = './pytorch_yesting_model_weight'
yesting_model.load_state_dict(torch.load(modelWeightFilePath, map_location=torch.device('cpu')))
yesting_model.fc = nn.Sequential(yesting_model.fc, nn.LogSoftmax(dim=1))
yesting_model.eval()

# 模型預測函數
def yesting_model_predict(image):

    image = Image.fromarray(image)
    transforms_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transforms_test(image)  # transform照片
    image = image.unsqueeze(0)  # 增加一個batch維度

    with torch.no_grad():
        output = yesting_model(image)

    return torch.exp(output)


############### 前置作業(Hao) ###############
# 讀取字對應編號資料
with open('vocab2idx.pkl', 'rb') as handle:
    vocab2idx = pickle.load(handle)
    idx2vocab = {v: k for k, v in vocab2idx.items()}

# 分類數
NUM_CLASSES = len(vocab2idx)

# 設定pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HEIGHT, WIDTH = 67, 49
NORM_MEAN, NORM_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
_test_transform = A.Compose(
    [
        A.Resize(always_apply=True, height=HEIGHT, width=WIDTH),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2()
    ]
)


# transformer函數
def test_transform(img):
    return _test_transform(image=img)['image']


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# 預測一張RGB圖片
def predict_image(img, model, transform=None):
    """
        input: image of RGB format
        output: tuple(predicted_index, predicted_probability)
    """

    img = img[:, :, :3]
    h, w, c = img.shape
    # 圖片太窄，直接回傳 index=-1
    if w <= 5:
        return -1
    # 圖片不合理地寬，動點手腳
    if w/h >= 2:
        img_diff_hrz = img[:, 1:,:] - img[:, :-1,:]
        img_diff_hrz_pxl = ((np.abs(img_diff_hrz) < 6).sum(axis=2) == 3) # RGB 都相同
        hrz_sum = img_diff_hrz_pxl.sum(axis=0) / h
        window_size = 5
        threshold = 0.5
        hrz_sum_window = running_mean(hrz_sum, window_size)
        kept_idx_hrz = [i+window_size for i, x in enumerate(hrz_sum_window) if x < threshold]
        if kept_idx_hrz:
            img = img[:, list(range(window_size))+kept_idx_hrz, :]
    if transform is not None:
        img = transform(img)

    with torch.no_grad():
        output = model.to(device).forward(img.unsqueeze(0).to(device))
        # prob, pred = torch.max(torch.exp(out).to('cpu'), 1)

    return torch.exp(output)


# 建立pytorch模型框架
class EsunModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.width = WIDTH
        self.height = HEIGHT

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear((512 * 4 * 3), 2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, NUM_CLASSES),
            nn.LogSoftmax(dim=1)
        )
        self.flatten = nn.Flatten()

    def forward(self, img):
        conv = self.conv(img)
        out = self.fc(self.flatten(conv))
        return out


# 建立pytorch模型
model = EsunModel()
model.load_state_dict(torch.load("pytorch_hao_model_weight.pkl"))
model.eval()


############### 預測主程式 ###############
# 預測函數
def predict(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
        注意: 此處為cv2轉換的ndarray 格式為BGR
    @returns:
        prediction (str): a word.
    """

    # 將cv2格式照片轉換為PIL
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 將BGR轉為RGB

    # 兩模型預測結果
    haoModelOutput = predict_image(img=image, model=model, transform=test_transform)
    yestingModelOutput = yesting_model_predict(image)

    # 模型平均
    _, prediction = torch.max(haoModelOutput + yestingModelOutput, 1)
    prediction = idx2vocab[prediction.item()]

    # 檢查是否為字串
    if _check_datatype_to_string(prediction):
        return prediction


# 檢查predict結果是否為字串
def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')
