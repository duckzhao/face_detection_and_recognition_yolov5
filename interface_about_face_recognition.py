import os
import time
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import json
import uvicorn
import cv2
import numpy as np
import io
import base64
from iresnet import iresnet50

app = FastAPI()


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


device = select_device('')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
weights = './runs/train/exp/weights/best.pt'
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(640, s=stride)  # check img_size
if half:
    model.half()  # to FP16

def detect(source, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
    # img0 = cv2.imread(source)  # BGR
    img0 = source
    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Get names and colors---获取classnamelist
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    pred_list = []
    # print(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    # print(pred)
    if pred:
        for p in pred:
            pred_list.append(p.tolist())
    # print(f'Done. ({time.time() - t0:.3f}s)')
    return pred_list


def base64_to_image(base64_code):
    # base64解码
    img_data = base64.b64decode(base64_code)
    # 转换为np数组
    img_array = np.fromstring(img_data, np.uint8)
    # 转换成opencv可用格式
    img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)

    return img


class Image(BaseModel):
    img: str


net = iresnet50()
net.load_state_dict(torch.load('./backbone.pth'))
net.eval()
# 获取人脸特征
@torch.no_grad()
def inference(img):
    # img 是人脸的剪切图像
    img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    feat = net(img).numpy().flatten()
    # print(feat)
    return feat

# 加载存储的人脸特征
def load_face_feature(dir):
    # 将人脸文件夹中的人脸特征都存到字典里，方便比对
    face_list = os.listdir(dir)
    # print(face_list)
    face_feature_dict = {}
    for face in face_list:
        img0 = cv2.imread(os.path.join(dir, face))
        img0_feature = inference(img0)
        face_feature_dict[face.replace('.jpg', '')] = img0_feature
    return face_feature_dict
face_feature_dict = load_face_feature('./face_img_database')

# 人脸特征对比
def cosin_metric(x1, x2):
    # single feature vs. single feature
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


# 对比人脸特征返回比对名称，不存在返回none
def compare_face(face_img0, face_feature_dict):
    face_img0_feature = inference(face_img0)
    # print(face_img0_feature)
    max_prob = 0
    max_name = ''
    for name in face_feature_dict.keys():
        face_img1_feature = face_feature_dict[name]
        prob = cosin_metric(face_img0_feature, face_img1_feature)
        if prob > max_prob:
            max_prob = prob
            max_name = name
    # print(max_name, max_prob)

    if max_prob > 0.4:
        return max_name
    else:
        face_name_list = os.listdir('./face_img_database')
        index = 1
        while True:
            if 'unkonw{}.jpg'.format(index) in face_name_list:
                index += 1
            else:
                cv2.imwrite('./face_img_database/{}'.format('unkonw{}.jpg'.format(index)), face_img0)
                face_feature_dict['unkonw{}'.format(index)] = face_img0_feature
                return 'unkonw{}'.format(index)


@app.post('/detect')
def detect_fun(image: Image):
    img = base64_to_image(image.img)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred_list = detect(source=img)
    # print(pred_list[0])
    if len(pred_list[0]) != 0:
        # 有人脸的话就把人脸截取出来
        for index, x in enumerate(pred_list[0]):
            face = img[int(x[1]): int(x[3]), int(x[0]): int(x[2])]
            name = compare_face(face, face_feature_dict)
            pred_list[0][index][-1] = name
    # cv2.imwrite('xxx.jpg', img)
    return {'state': 'success', 'answer': pred_list}


if __name__ == '__main__':
    # fast:app 中的 fast=运行的文件名,如果修改了记得这里别忘记改
    uvicorn.run("interface_about_face_recognition:app", host="0.0.0.0", port=8000, reload=True)
    # pred_list = detect(weights='./runs/train/exp8/weights/best.pt', source="/home/zk/git_projects/hand_pose/hand_pose_yolov5_5.0/hand_pose/images/four_fingers10.jpg")
    # print(pred_list)