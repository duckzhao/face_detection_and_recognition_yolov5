import requests
import cv2
import base64
import json
import random

def run():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print('cap error,system exit')
            break
        # 反转图像
        frame = cv2.flip(frame, 1)
        # 检测手位置
        hands_pos = detect(frame)
        # 在复制的图层进行表情绘制
        draw_frame = frame.copy()
        # 如果检测的人脸位置不为空才进来预测表情
        if len(hands_pos) != 0:
            # 预测人脸的表情
            draw_hands(draw_frame, hands_pos)

        # 展示视频画面
        cv2.imshow('video capture', draw_frame)
        key = cv2.waitKey(10) & 0xff
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

def image_to_base64(image_np):
    image = cv2.imencode('.jpg', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]
    return image_code

def detect(frame):
    files = {'img': image_to_base64(frame)}
    # print(files)
    response = requests.post('http://192.168.0.101:8000/detect', json.dumps(files))
    # print(response.json())
    return response.json()['answer'][0]

def draw_hands(draw_frame, hands_pos):
    classes = ['person']
    # print(hands_pos)
    for hand_pos in hands_pos:
        plot_one_box(hand_pos[:4], draw_frame, [255,0,0], label=hand_pos[-1], confidence=hand_pos[-2])


# x是 预测结果前4个
def plot_one_box(x, img, color=None, label=None, line_thickness=3, confidence=0):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label+'{}'.format(format(confidence, '.3f')), (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255],
                    thickness=tf, lineType=cv2.LINE_AA)

if __name__ == '__main__':
    run()