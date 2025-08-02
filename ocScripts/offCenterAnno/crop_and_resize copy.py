import os
import cv2
import numpy as np
from tqdm import tqdm
import glob

def load_class_mapping(class_file):
    """
    加载类别映射文件
    
    Args:
        class_file: 类别映射文件路径
        
    Returns:
        dict: 类别名称到编码的映射字典
    """
    class_mapping = {}
    try:
        with open(class_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    index, code, class_name = parts[0], parts[1], parts[2]
                    class_mapping[class_name] = code
    except Exception as e:
        print(f"警告: 读取类别映射文件时出错: {str(e)}")
    return class_mapping

def process_single_image(image_file, annotation_file, output_dir, class_mapping, scale_factor=2):
    """
    处理单个图片：读取标注文件，裁剪图片区域，根据缩放倍率调整大小
    
    Args:
        image_file: 原始图片文件路径
        annotation_file: 标注文件路径
        output_dir: 输出目录
        class_mapping: 类别映射字典
        scale_factor: 缩放倍率，例如2表示将尺寸缩小为原来的1/2
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取原始图片文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(image_file))[0]
    
    # 读取图片
    try:
        image = cv2.imdecode(np.fromfile(image_file, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"警告: 无法读取图片 {image_file}")
            return
    except Exception as e:
        print(f"警告: 读取图片 {image_file} 时出错: {str(e)}")
        return
        
    # 读取标注文件
    try:
        with open(annotation_file, 'r') as f:
            annotations = f.readlines()
    except Exception as e:
        print(f"警告: 读取标注文件 {annotation_file} 时出错: {str(e)}")
        return
        
    # 处理每个标注
    for i, anno in enumerate(annotations):
        try:
            # 解析标注信息
            parts = anno.strip().split(',')
            if len(parts) != 5:
                print(f"警告: 标注格式错误 {anno}")
                continue
                
            class_name, x1, y1, x2, y2 = parts
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # 获取类别编码
            class_code = class_mapping.get(class_name)
            if class_code is None:
                print(f"警告: 找不到类别 {class_name} 的编码")
                continue
            
            # 确保坐标在图片范围内
            height, width = image.shape[:2]
            
            # 计算目标尺寸（基于标注文件）
            target_width = x2 - x1
            target_height = y2 - y1
            
            # 创建黑色背景
            cropped = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # 计算实际可用的裁剪区域
            valid_x1 = max(0, x1)
            valid_y1 = max(0, y1)
            valid_x2 = min(width, x2)
            valid_y2 = min(height, y2)
            
            # 计算在黑色背景上的偏移量
            offset_x = valid_x1 - x1
            offset_y = valid_y1 - y1
            
            # 将有效区域复制到黑色背景上
            valid_region = image[valid_y1:valid_y2, valid_x1:valid_x2]
            if valid_region.size > 0:  # 确保有效区域不为空
                cropped[offset_y:offset_y+valid_region.shape[0], 
                       offset_x:offset_x+valid_region.shape[1]] = valid_region
            
            # 计算缩放后的尺寸
            target_width = cropped.shape[1] // scale_factor
            target_height = cropped.shape[0] // scale_factor
            
            # 调整大小
            resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # 生成输出文件名（保持原始文件名）
            output_name = f"{base_name}.jpg"
            output_path = os.path.join(output_dir, output_name)
            cv2.imwrite(output_path, resized)
            
            # 只处理第一个标注（如果有多个标注，只保存第一个）
            break
            
        except Exception as e:
            print(f"警告: 处理标注 {anno} 时出错: {str(e)}")
            continue

def process_folder(image_dir, annotation_dir, output_dir, class_file, scale_factor=2):
    """
    处理文件夹中的所有图片
    
    Args:
        image_dir: 图片文件夹路径
        annotation_dir: 标注文件夹路径
        output_dir: 输出文件夹路径
        class_file: 类别映射文件路径
        scale_factor: 缩放倍率
    """
    # 加载类别映射
    class_mapping = load_class_mapping(class_file)
    if not class_mapping:
        print("错误: 无法加载类别映射文件")
        return
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"错误: 在 {image_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 处理每个图片文件
    for image_file in tqdm(image_files, desc="处理图片"):
        # 获取图片文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        
        # 构建对应的标注文件路径
        annotation_file = os.path.join(annotation_dir, f"{base_name}.txt")
        
        # 检查标注文件是否存在
        if not os.path.exists(annotation_file):
            print(f"警告: 找不到对应的标注文件 {annotation_file}")
            continue
        
        # 处理单个图片
        process_single_image(image_file, annotation_file, output_dir, class_mapping, scale_factor)
    
    print("所有图片处理完成！")

def main():
    # 设置文件夹路径
    image_dir = r"D:\PCdata\project\HR_DatasetBuild\dataset_person_008\000092"  # 图片文件夹路径
    annotation_dir = r"D:\PCdata\project\HR_DatasetBuild\dataset_person_008\annotation"  # 标注文件夹路径
    output_dir = r"D:\PCdata\project\HR_DatasetBuild\dataset_person_008\cropped_images"  # 输出文件夹路径
    class_file = r"D:\PCdata\project\HR_DatasetBuild\dataset_person_007\class.txt"  # 类别映射文件路径
    
    # 设置缩放倍率
    scale_factor = 1  # 设置为2表示将尺寸缩小为原来的1/2
    
    # 检查文件夹是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图片文件夹不存在: {image_dir}")
        return
        
    if not os.path.exists(annotation_dir):
        print(f"错误: 标注文件夹不存在: {annotation_dir}")
        return
    
    # 处理文件夹中的所有图片
    process_folder(image_dir, annotation_dir, output_dir, class_file, scale_factor)

if __name__ == "__main__":
    main() 