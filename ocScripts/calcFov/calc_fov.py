import os
import math
import cv2
import numpy as np

# 设定相机参数
fx = 762.77632319717236
fy = 762.77632319717236

anno_dir = 'data-qianqian-body/annotation'
img_dir = 'data-qianqian-body/images'
images_txt = 'data-qianqian-body/images.txt'
output_file = 'data-qianqian-body/fov_results_qianqian_body.txt'

# 1. 解析 images.txt，获得有序图片名列表
image_order = []
with open(images_txt, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith('#') or line == '' or 'IMAGE_ID' in line:
            continue
        # 只处理图片信息行（即每两行的第一行，且最后一个字段是图片名）
        parts = line.split()
        if len(parts) >= 10 and parts[-1].endswith('.png'):
            image_order.append(parts[-1])

# 2. 按照 image_order 顺序处理
with open(output_file, 'w', encoding='utf-8') as fout:
    for idx, img_name in enumerate(image_order):
        # 读取图片真实宽高
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"图片读取失败: {img_path}")
            continue
        height, width = img.shape[:2]

        anno_name = img_name.replace('.png', '.txt')
        anno_path = os.path.join(anno_dir, anno_name)
        if not os.path.exists(anno_path):
            continue
        with open(anno_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                parts = line.strip().split(',')
                if len(parts) != 5:
                    continue
                cls, x1, y1, x2, y2 = parts
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])
                box_width = x2 - x1
                box_height = y2 - y1
                x_img = (x1 + x2) / 2
                y_img = (y1 + y2) / 2
                Xo = -(x_img - width / 2)
                Yo = -(y_img - height / 2)

                Xo=Xo
                Yo=Yo

                W = box_width
                H = box_height

                W = W
                H = H
                fovx_r = math.atan((Xo + W/2) / fx)
                fovx_l = math.atan((Xo - W/2) / fx)
                fovy_t = math.atan((Yo + H/2) / fy)
                fovy_b = math.atan((Yo - H/2) / fy)

                #import pdb;pdb.set_trace()
                fout.write(f'{idx},{img_name},{fovx_r:.6f},{fovx_l:.6f},{fovy_t:.6f},{fovy_b:.6f},{W:.2f},{H:.2f}\n')
print('处理完成，结果保存在', output_file)