from copy import deepcopy
import shutil
import cv2
import tqdm

def read_file(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_data = f.read().splitlines()
    # print(data[0:128])
    img_info_list = []
    # 当前处理的img信息行号所处位置
    img_info__index = 1
    first_flag = True
    # 存放每个图片信息的dict
    temp_dict = {}
    # 存放每个图片 bbox xy wh 的list
    temp_list = []
    for row_data in tqdm.tqdm(full_data):
        # 如果第一次出现某图片名，则将其加入字典中，新建一个新的字典
        if '.jpg' in row_data:
            if first_flag:
                first_flag = False
            else:
                # 将上一个图片相关的信息都存到list中
                temp_dict['coord_list'] = temp_list
                temp_dict['processed_num'] = len(temp_list)
                img_info_list.append(deepcopy(temp_dict))
            # 清空temp_dict中的内容
            temp_dict.clear()
            temp_list = []
            temp_dict['filename'] = row_data
            # 如果进到了 jpg这一行，下一行就是数量行了
            img_info__index = 2
            continue
        # 则进入了数量行
        if img_info__index == 2:
            temp_dict['ori_face_num'] = int(row_data)
            img_info__index = 3
            continue
        # 则进入了坐标行
        if img_info__index >= 3:
            info = row_data.split(' ')[:10]
            # 设置过滤条件
            # blur：是模糊度，分三档：0，清晰；1：一般般；2：人鬼难分,过滤2
            # express：表达（什么鬼也没弄明白，反正我训这个用不着）
            # illumination：曝光，分正常和过曝
            # occlusion：遮挡，分三档。0，无遮挡；1，小遮挡；2，大遮挡；
            # invalid：（没弄明白）
            # pose：（疑似姿态？分典型和非典型姿态）
            if info[4] == '2' or info[7] == '2':
                continue
            else:
                # 取坐标
                coord = info[:4]
                temp_list.append(coord)
    # 最后一个图片的信息不会进入循环保存，手动保存下
    temp_dict['coord_list'] = temp_list
    temp_dict['processed_num'] = len(temp_list)
    img_info_list.append(deepcopy(temp_dict))
    return img_info_list

'''
[{'filename': '0--Parade/0_Parade_marchingband_1_465.jpg', 'ori_face_num': 126,
'coord_list': [['331', '126', '3', '3'], ['221', '128', '4', '5']], 'processed_num': 42}, ...]
'''
def process_img_info_list(img_info_list, type):
    for img_info in tqdm.tqdm(img_info_list):
        # 仅处理人物数量大于10的图片---对processed_num进行筛选即可
        pass
        file_name = img_info['filename']
        # 根据filename解析出要复制前后的文件路径
        if type == 0:
            ori_file_path = train_img_ori_path + file_name
            dst_file_path = train_img_save_path + file_name.replace('/', '')
            # 解析labels的保存地址
            dst_labels_path = train_img_save_path.replace('images', 'labels') + file_name.replace('/', '').replace('.jpg', '.txt')
        else:
            ori_file_path = val_img_ori_path + file_name
            dst_file_path = val_img_save_path + file_name.replace('/', '')
            dst_labels_path = val_img_save_path.replace('images', 'labels') + file_name.replace('/', '').replace('.jpg', '.txt')

        # 复制图片文件
        shutil.copyfile(ori_file_path, dst_file_path)

        # 生成labels文件并写入
        lines = []
        label = 0
        img = cv2.imread(ori_file_path)
        if not img.data:
            print('xxxxxxxxxx', ori_file_path)
            continue
        height = img.shape[0]
        width = img.shape[1]
        coord_list = img_info['coord_list']
        for coord in coord_list:
            x = float(coord[0])
            y = float(coord[1])
            x2 = x + float(coord[2])
            y2 = y + float(coord[3])
            cx = (x2 + x) * 0.5 / width
            cy = (y2 + y) * 0.5 / height
            w = (x2 - x) * 1. / width
            h = (y2 - y) * 1. / height
            line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
            lines.append(line)
        with open(dst_labels_path, 'w')as f:
            f.writelines(lines)


if __name__ == '__main__':
    train_txt_path = './wider_face_train_bbx_gt.txt'
    val_txt_path = './wider_face_val_bbx_gt.txt'
    train_img_save_path = './wider_dataset/images/train/'
    val_img_save_path = './wider_dataset/images/valid/'
    train_img_ori_path = './wider/WIDER_train/images/'
    val_img_ori_path = './wider/WIDER_valid/images/'

    img_info_list = read_file(val_txt_path)
    # print(img_info_list)
    process_img_info_list(img_info_list, 1)

    img_info_list = read_file(train_txt_path)
    # print(img_info_list)
    process_img_info_list(img_info_list, 0)