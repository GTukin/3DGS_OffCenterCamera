import os
import math
import cv2
import numpy as np

# 默认相机参数（当无法在 cameras.txt 中找到时兜底）
DEFAULT_FX = 762.77632319717236
DEFAULT_FY = 762.77632319717236

anno_dir = 'scripts_SR_annotator\dataset_03\HR_annotation'
img_dir = 'scripts_SR_annotator\dataset_03\HR_src'

# 读取 COLMAP 导出的相机与图像元信息
cameras_txt = 'scripts_SR_annotator\dataset_03\data_qianqian_body_new\\txt\cameras.txt'
images_txt = 'scripts_SR_annotator\dataset_03\data_qianqian_body_new\\txt\images.txt'

output_file = 'scripts_SR_annotator\dataset_03\\fov_results_HR_src.txt'

# 1. 解析 cameras.txt，建立 CAMERA_ID -> (fx, fy) 映射
camera_id_to_intrinsics = {}
if os.path.exists(cameras_txt):
    with open(cameras_txt, 'r', encoding='utf-8') as fcam:
        for line in fcam:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # 格式：CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
            # 例如：0 PINHOLE 526 938 770.763 770.763 263 469
            if len(parts) >= 8:
                try:
                    cam_id = int(parts[0])
                    fx_val = float(parts[4])
                    fy_val = float(parts[5])
                    camera_id_to_intrinsics[cam_id] = (fx_val, fy_val)
                except Exception:
                    continue

# 2. 解析 images.txt，获得有序图片名列表与对应 CAMERA_ID
image_order = []
image_name_to_camera_id = {}
if os.path.exists(images_txt):
    with open(images_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith('#') or line == '' or 'IMAGE_ID' in line:
                continue
            parts = line.split()
            # COLMAP images.txt 的图片行：IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME
            # 该文件每两行一组，这里只处理图片行（有 >=10 个字段，最后一个为文件名）
            if len(parts) >= 10:
                image_name = parts[-1]
                try:
                    camera_id = int(parts[-2])
                except Exception:
                    # 若无法解析则跳过本行
                    continue
                image_order.append(image_name)
                image_name_to_camera_id[image_name] = camera_id

# 3. 按照 image_order 顺序处理
with open(output_file, 'w', encoding='utf-8') as fout:
    for idx, img_name in enumerate(image_order):
        # 读取图片真实宽高
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"图片读取失败: {img_path}")
            continue
        height, width = img.shape[:2]

        # 获取对应相机的 fx, fy
        camera_id = image_name_to_camera_id.get(img_name, None)
        if camera_id is not None and camera_id in camera_id_to_intrinsics:
            fx, fy = camera_id_to_intrinsics[camera_id]
        else:
            fx, fy = DEFAULT_FX, DEFAULT_FY

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

                W = box_width
                H = box_height

                fovx_r = math.atan((Xo + W/2) / fx)
                fovx_l = math.atan((Xo - W/2) / fx)
                fovy_t = math.atan((Yo + H/2) / fy)
                fovy_b = math.atan((Yo - H/2) / fy)

                fout.write(f'{idx},{img_name},{fovx_r:.6f},{fovx_l:.6f},{fovy_t:.6f},{fovy_b:.6f},{W:.2f},{H:.2f}\n')
print('处理完成，结果保存在', output_file)
