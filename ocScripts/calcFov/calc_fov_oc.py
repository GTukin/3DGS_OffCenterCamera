import os
import math
import cv2
import numpy as np
import shutil

# 默认相机参数（当无法在 cameras.txt 中找到时兜底）
DEFAULT_FX = 762.77632319717236
DEFAULT_FY = 762.77632319717236

# 相机参数缩放倍数（用于处理不同分辨率的图像）
# 当图像分辨率变化时，相机内参 fx、fy 需要按相同比例缩放
# scale < 1.0 表示缩小（图像分辨率变小，相机参数也缩小）
# scale > 1.0 表示放大（图像分辨率变大，相机参数也放大）
# 例如：scale = 0.25 表示图像分辨率缩小到原来的 1/4，相机参数也缩小到原来的 1/4
scale = 1/2

# ========== 配置参数 ==========
# 偏心相机的输入目录
oc_anno_dir = r'G:\3dgs\DataSet_3D\360_v2\kitchen\01_annotation_ocrename'
# OC图像的输入目录（这里的图像会被重命名添加后缀）
oc_img_input_dir = r'G:\3dgs\DataSet_3D\360_v2\kitchen\01_HR_SRC_cropped_ocrename'
# 原图目录（src图像，用于参考，不会被重命名）
oc_img_dir = r'G:\3dgs\DataSet_3D\360_v2\kitchen\00_org_images\images_2'

# 原始相机参数文件路径
cameras_txt = r'G:\3dgs\DataSet_3D\360_v2\kitchen\00_org_images\sparse_txt\cameras.txt'
images_txt = r'G:\3dgs\DataSet_3D\360_v2\kitchen\00_org_images\sparse_txt\images.txt'

# 输出目录（用于保存新的 cameras.txt 和 images.txt）
output_sparse_dir = r'G:\3dgs\DataSet_3D\360_v2\kitchen\00_org_images\sparse_txt_oc'

# 图像文件名后缀（例如 "_oc"）
image_suffix = "_oc"

# 输出文件
output_file = r'G:\3dgs\DataSet_3D\360_v2\kitchen\00_org_images\fov_results_HR_oc.txt'

# ========== 步骤1: 重命名OC图像文件，添加后缀 ==========
print("=" * 60)
print("步骤1: 重命名OC图像文件，添加后缀")
print("=" * 60)

if not os.path.exists(oc_img_input_dir):
    print(f"错误: OC图像输入目录不存在: {oc_img_input_dir}")
    exit(1)

renamed_count = 0
image_name_mapping = {}  # 原始名称 -> 新名称的映射

# 获取所有图像文件
image_extensions = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']
for filename in os.listdir(oc_img_input_dir):
    file_path = os.path.join(oc_img_input_dir, filename)
    if not os.path.isfile(file_path):
        continue
    
    # 检查是否是图像文件
    _, ext = os.path.splitext(filename)
    if ext not in image_extensions:
        continue
    
    # 检查是否已经有后缀
    name_without_ext, ext = os.path.splitext(filename)
    if name_without_ext.endswith(image_suffix):
        print(f"  跳过（已有后缀）: {filename}")
        new_name = filename
    else:
        # 添加后缀
        new_name = name_without_ext + image_suffix + ext
        new_path = os.path.join(oc_img_input_dir, new_name)
        
        # 如果新文件名已存在，跳过
        if os.path.exists(new_path):
            print(f"  警告: 目标文件已存在，跳过: {new_name}")
            new_name = filename
        else:
            os.rename(file_path, new_path)
            print(f"  重命名: {filename} -> {new_name}")
            renamed_count += 1
    
    image_name_mapping[filename] = new_name

print(f"完成: 重命名了 {renamed_count} 个图像文件")

# ========== 步骤1.5: 重命名标注文件，添加后缀 ==========
print("\n" + "=" * 60)
print("步骤1.5: 重命名标注文件，添加后缀")
print("=" * 60)

if not os.path.exists(oc_anno_dir):
    print(f"警告: 标注目录不存在: {oc_anno_dir}")
    print("将跳过标注文件重命名步骤")
else:
    anno_renamed_count = 0
    
    # 获取所有标注文件
    for filename in os.listdir(oc_anno_dir):
        file_path = os.path.join(oc_anno_dir, filename)
        if not os.path.isfile(file_path):
            continue
        
        # 检查是否是标注文件（.txt）
        if not filename.lower().endswith('.txt'):
            continue
        
        # 获取不带扩展名的文件名
        name_without_ext, ext = os.path.splitext(filename)
        
        # 检查是否已经有后缀
        if name_without_ext.endswith(image_suffix):
            print(f"  跳过（已有后缀）: {filename}")
            continue
        
        # 添加后缀
        new_name = name_without_ext + image_suffix + ext
        new_path = os.path.join(oc_anno_dir, new_name)
        
        # 如果新文件名已存在，跳过
        if os.path.exists(new_path):
            print(f"  警告: 目标文件已存在，跳过: {new_name}")
            continue
        
        os.rename(file_path, new_path)
        print(f"  重命名: {filename} -> {new_name}")
        anno_renamed_count += 1
    
    print(f"完成: 重命名了 {anno_renamed_count} 个标注文件")

# ========== 步骤2: 复制并修改 cameras.txt ==========
print("\n" + "=" * 60)
print("步骤2: 复制 cameras.txt")
print("=" * 60)

if not os.path.exists(cameras_txt):
    print(f"错误: cameras.txt 文件不存在: {cameras_txt}")
    exit(1)

# 创建输出目录
os.makedirs(output_sparse_dir, exist_ok=True)
output_cameras_txt = os.path.join(output_sparse_dir, 'cameras.txt')

# 直接复制 cameras.txt（相机参数保持不变）
shutil.copy2(cameras_txt, output_cameras_txt)
print(f"已复制 cameras.txt 到: {output_cameras_txt}")

# ========== 步骤3: 复制 images.txt 并添加 oc 图片条目 ==========
print("\n" + "=" * 60)
print("步骤3: 复制 images.txt 并添加 oc 图片条目")
print("=" * 60)

if not os.path.exists(images_txt):
    print(f"错误: images.txt 文件不存在: {images_txt}")
    exit(1)

output_images_txt = os.path.join(output_sparse_dir, 'images.txt')

# 先读取原始 images.txt，获取最大 IMAGE_ID
max_image_id = 0
original_entries = []  # 存储原始条目信息

with open(images_txt, 'r', encoding='utf-8') as fin:
    lines = fin.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 保留注释和空行
        if line.strip().startswith('#') or line.strip() == '':
            original_entries.append((line, None))
            i += 1
            continue
        
        parts = line.strip().split()
        # COLMAP images.txt 格式：每两行一组
        # 第一行：IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        if len(parts) == 10:
            try:
                image_id = int(parts[0])
                max_image_id = max(max_image_id, image_id)
                
                # 保存原始条目（两行）
                points2d_line = lines[i + 1] if i + 1 < len(lines) else ""
                original_entries.append((line, points2d_line))
                i += 2
            except Exception as e:
                print(f"  警告: 解析图像ID时出错: {line.strip()}, 错误: {e}")
                original_entries.append((line, None))
                i += 1
        else:
            original_entries.append((line, None))
            i += 1

# 写入新的 images.txt：先写入原始内容，再添加 oc 条目
added_oc_count = 0
new_image_id = max_image_id + 1

with open(output_images_txt, 'w', encoding='utf-8') as fout:
    # 第一步：写入原始 images.txt 的所有内容（保持不变）
    print("  写入原始 images.txt 内容...")
    for entry_line, points2d_line in original_entries:
        fout.write(entry_line)
        if points2d_line:
            fout.write(points2d_line)
    
    # 第二步：为 oc 图片添加新条目
    print("  添加 oc 图片条目...")
    for entry_line, points2d_line in original_entries:
        # 跳过注释和空行
        if entry_line.strip().startswith('#') or entry_line.strip() == '':
            continue
        
        parts = entry_line.strip().split()
        if len(parts) == 10:
            try:
                original_image_id = int(parts[0])
                camera_id = int(parts[8])
                image_name = parts[9]
                
                # 查找对应的 oc 图片名
                if image_name in image_name_mapping:
                    oc_image_name = image_name_mapping[image_name]
                else:
                    # 如果映射中没有，尝试添加后缀
                    name_without_ext, ext = os.path.splitext(image_name)
                    if not name_without_ext.endswith(image_suffix):
                        oc_image_name = name_without_ext + image_suffix + ext
                    else:
                        oc_image_name = image_name
                
                # 检查 oc 图片文件是否存在（在OC图像输入目录中查找重命名后的文件）
                oc_img_path = os.path.join(oc_img_input_dir, oc_image_name)
                if os.path.exists(oc_img_path):
                    # 创建新的条目，使用新的 IMAGE_ID，但保持相同的相机参数和位姿
                    new_parts = parts.copy()
                    new_parts[0] = str(new_image_id)  # 新的 IMAGE_ID
                    new_parts[9] = oc_image_name  # oc 图片名
                    new_line = ' '.join(new_parts) + '\n'
                    fout.write(new_line)
                    
                    # 写入对应的 POINTS2D 数据行（如果有的话）
                    if points2d_line:
                        fout.write(points2d_line)
                    else:
                        fout.write('\n')  # 如果没有 POINTS2D 数据，写入空行
                    
                    print(f"  添加 oc 条目: IMAGE_ID={new_image_id}, {image_name} -> {oc_image_name}")
                    new_image_id += 1
                    added_oc_count += 1
                    
            except Exception as e:
                print(f"  警告: 处理 oc 条目时出错: {entry_line.strip()}, 错误: {e}")
                continue

# 统计原始图像条目数量
original_image_count = 0
for entry_line, _ in original_entries:
    parts = entry_line.strip().split()
    if len(parts) == 10:
        original_image_count += 1

print(f"完成: 保留了原始 {original_image_count} 个图像条目")
print(f"完成: 添加了 {added_oc_count} 个 oc 图像条目")

# ========== 步骤4: 读取新的相机参数文件 ==========
print("\n" + "=" * 60)
print("步骤4: 读取新的相机参数文件")
print("=" * 60)

camera_id_to_intrinsics = {}
print(f"正在读取相机参数文件: {output_cameras_txt}")
if os.path.exists(output_cameras_txt):
    with open(output_cameras_txt, 'r', encoding='utf-8') as fcam:
        lines = fcam.readlines()
        print(f"cameras.txt 文件存在，共 {len(lines)} 行")
        
        valid_camera_count = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # 格式：CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
            if len(parts) >= 8:
                try:
                    cam_id = int(parts[0])
                    fx_val = float(parts[4]) * scale
                    fy_val = float(parts[5]) * scale
                    camera_id_to_intrinsics[cam_id] = (fx_val, fy_val)
                    valid_camera_count += 1
                    print(f"  第{line_num}行: Camera ID {cam_id}, fx={fx_val:.6f}, fy={fy_val:.6f} (已应用 scale={scale})")
                except Exception as e:
                    print(f"  第{line_num}行解析失败: {line}, 错误: {e}")
                    continue
        
        print(f"成功解析 {valid_camera_count} 个相机参数")
        if valid_camera_count == 0:
            print("警告: cameras.txt 中没有找到有效的相机参数，将使用默认值")
else:
    print(f"错误: cameras.txt 文件不存在: {output_cameras_txt}")
    print("将使用默认相机参数")

# ========== 步骤5: 读取新的图像信息文件 ==========
print("\n" + "=" * 60)
print("步骤5: 读取新的图像信息文件")
print("=" * 60)

image_order = []
image_name_to_camera_id = {}
print(f"正在读取图像信息文件: {output_images_txt}")
if os.path.exists(output_images_txt):
    with open(output_images_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(f"images.txt 文件存在，共 {len(lines)} 行")
        
        valid_image_count = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('#') or line == '':
                i += 1
                continue
            
            parts = line.split()
            # 检查是否是图像信息行（应该恰好有10个字段）
            if len(parts) == 10:
                try:
                    image_id = int(parts[0])
                    camera_id = int(parts[8])  # CAMERA_ID 在第9个位置（索引8）
                    image_name = parts[9]      # NAME 在第10个位置（索引9）
                    
                    # 只处理 oc 图像（带后缀的）
                    name_without_ext, ext = os.path.splitext(image_name)
                    if name_without_ext.endswith(image_suffix):
                        image_order.append(image_name)
                        image_name_to_camera_id[image_name] = camera_id
                        valid_image_count += 1
                        print(f"  第{valid_image_count}个 oc 图像: {image_name}, Camera ID: {camera_id}")
                    
                    # 跳过下一行（POINTS2D数据行）
                    i += 2
                except Exception as e:
                    print(f"解析图像行时出错: {line}, 错误: {e}")
                    i += 1
            else:
                i += 1
        
        print(f"成功解析 {valid_image_count} 个图像信息")
        if valid_image_count == 0:
            print("警告: images.txt 中没有找到有效的图像信息")
else:
    print(f"错误: images.txt 文件不存在: {output_images_txt}")

# ========== 步骤6: 计算偏心相机的FOV参数 ==========
print("\n" + "=" * 60)
print("步骤6: 计算偏心相机的FOV参数")
print("=" * 60)
print(f"开始处理图像，共 {len(image_order)} 个图像")
print(f"输出文件: {output_file}")

processed_count = 0
skipped_count = 0
total_annotations = 0

with open(output_file, 'w', encoding='utf-8') as fout:
    for idx, img_name in enumerate(image_order):
        print(f"\n处理第 {idx+1}/{len(image_order)} 个图像: {img_name}")
        
        # 将带"_oc"后缀的图像名转换为原图名称（用于从oc_img_dir读取）
        name_without_ext, ext = os.path.splitext(img_name)
        if name_without_ext.endswith(image_suffix):
            original_img_name = name_without_ext[:-len(image_suffix)] + ext
        else:
            original_img_name = img_name
        
        # 读取图片真实宽高（从oc_img_dir读取原图，用于FOV计算）
        img_path = os.path.join(oc_img_dir, original_img_name)
        if not os.path.exists(img_path):
            print(f"  警告: 图像文件不存在: {img_path}")
            skipped_count += 1
            continue
            
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  错误: 图片读取失败: {img_path}")
            skipped_count += 1
            continue
            
        height, width = img.shape[:2]
        print(f"  图像尺寸: {width} x {height}")

        # 获取对应相机的 fx, fy
        camera_id = image_name_to_camera_id.get(img_name, None)
        if camera_id is not None and camera_id in camera_id_to_intrinsics:
            fx, fy = camera_id_to_intrinsics[camera_id]
            print(f"  使用相机参数: Camera ID {camera_id}, fx={fx:.6f}, fy={fy:.6f} (已应用 scale={scale})")
        else:
            fx, fy = DEFAULT_FX * scale, DEFAULT_FY * scale
            print(f"  使用默认相机参数: fx={fx:.6f}, fy={fy:.6f} (已应用 scale={scale})")

        # 检查标注文件
        # 标注文件名应该对应重命名后的图像名
        anno_name = img_name.replace('.JPG', '.txt').replace('.jpg', '.txt').replace('.png', '.txt')
        anno_path = os.path.join(oc_anno_dir, anno_name)
        if not os.path.exists(anno_path):
            print(f"  警告: 标注文件不存在: {anno_path}")
            skipped_count += 1
            continue
        
        # 处理标注文件
        annotation_count = 0
        with open(anno_path, 'r', encoding='utf-8') as fin:
            for line_num, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                if len(parts) != 5:
                    print(f"  警告: 第{line_num}行格式不正确: {line}")
                    continue
                    
                try:
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
                    annotation_count += 1
                    total_annotations += 1
                    
                except Exception as e:
                    print(f"  错误: 第{line_num}行处理失败: {line}, 错误: {e}")
                    continue
        
        print(f"  处理了 {annotation_count} 个标注")
        processed_count += 1

print(f'\n' + "=" * 60)
print(f'处理完成!')
print(f'成功处理: {processed_count} 个图像')
print(f'跳过: {skipped_count} 个图像')
print(f'总标注数: {total_annotations} 个')
print(f'结果保存在: {output_file}')
print(f'新的 cameras.txt 保存在: {output_cameras_txt}')
print(f'新的 images.txt 保存在: {output_images_txt}')
print("=" * 60)

