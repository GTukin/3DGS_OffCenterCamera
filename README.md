# 偏心投影功能更新说明

## 概述
项目新增了偏心投影（Off-center Projection）功能，用于处理非中心投影的3D高斯渲染。

## 主要新增功能

### 1. 偏心投影计算模块
- **文件位置**: `ocScripts/calcFov/calc_fov_oc.py`
- **功能**: 计算偏心投影的FOV参数（r, l, b, t），和camera.txt+images.txt编码对应
- **核心特性**:
  - 支持偏心投影矩阵计算
  - 支持FOV边界参数（fovx_r, fovx_l, fovy_t, fovy_b）
  - 自动生成对应的 cameras.txt 和 images.txt 文件
  - 支持图像重命名和参数缩放

### 2. 图像标注工具
- **文件位置**: `ocScripts/offCenterAnno/image_annotator.py`
- **功能**: 交互式图像标注工具
- **特性**:
  - 基于PyQt5的GUI界面
  - 支持矩形框标注
  - 可调整选框大小
  - 支持批量图片处理
  - 快捷键操作（A/D键切换图片）

### 3. 图像裁剪和缩放工具
- **文件位置**: `ocScripts/offCenterAnno/crop_and_resize copy.py`
- **功能**: 根据标注信息裁剪和缩放图像
- **特性**:
  - 支持类别映射
  - 自动创建黑色背景
  - 可配置缩放倍率

## 核心代码修改

### 1. 相机模块更新 (`scene/cameras.py`)
- 新增偏心投影参数：`fovx_r`, `fovx_l`, `fovy_t`, `fovy_b`
- 新增 `ocmodel` 参数用于控制是否启用偏心投影模式
- 集成新的投影矩阵计算函数 `getProjectionMatrix_p`
- 支持混合模式：自动检测偏心参数，部分相机使用偏心投影，部分使用标准投影

### 2. 数据集读取器更新 (`scene/dataset_readers.py`)
- 支持从 `sparse/0/fov_results.txt` 读取FOV参数文件
- 新增FOV参数到 `CameraInfo` 结构
- 修改相机读取逻辑以支持偏心投影
- 支持图片名称的精确匹配和basename匹配
- 自动统计和报告偏心相机与标准相机的数量

### 3. 图形工具更新 (`utils/graphics_utils.py`)
- 新增 `getProjectionMatrix_p` 函数
- 实现偏心投影矩阵计算

### 4. 参数系统更新 (`arguments/__init__.py`)
- 新增 `--ocmodel` 命令行参数（默认：False）
- 用于启用/禁用偏心投影模式

### 5. 相机工具更新 (`utils/camera_utils.py`)
- 支持将 `ocmodel` 参数传递到相机对象
- 自动统计实际使用偏心投影的相机数量
- 输出详细的投影模式统计信息

### 6. 训练脚本更新 (`train.py`)
- 修复模型加载参数
- 优化透明度重置阈值(0.01—>0.005)

## 数据格式

### FOV结果文件格式 (`fov_results.txt`)

支持两种格式（索引字段是可选的，匹配仅通过图片名称进行）：

**格式1（带索引，8列）**：
```
索引,图片名,fovx_r,fovx_l,fovy_t,fovy_b,宽度,高度
```
示例：
```
0,40.png,0.275951,-0.236702,0.309598,-0.538812,400.00,700.00
```

**格式2（无索引，7列）**：
```
图片名,fovx_r,fovx_l,fovy_t,fovy_b,宽度,高度
```
示例：
```
40.png,0.275951,-0.236702,0.309598,-0.538812,400.00,700.00
```

**注意**：
- 索引字段（第一列）是可选的，不会被使用
- 匹配通过图片名称进行，图片名称必须与COLMAP中的名称一致
- 图片名称支持完整路径或仅文件名
- 文件位置：必须放在数据集的 `sparse/0/fov_results.txt` 路径下

## 使用方法

### 1. 计算FOV参数
```bash
cd ocScripts/calcFov
python calc_fov_oc.py
```
注意：使用前需要修改脚本中的配置参数（路径、缩放比例等）

### 2. 标注图像
```bash
cd ocScripts/offCenterAnno
python image_annotator.py
```

### 3. 处理标注数据
```bash
python crop_and_resize copy.py
```

### 4. 准备FOV结果文件
将生成的 `fov_results.txt` 文件放置在数据集的以下位置：
```
数据集根目录/
  └── sparse/
      └── 0/
          └── fov_results.txt
```

### 5. 训练模型（启用偏心投影）
```bash
python train.py -s 数据集路径 --ocmodel
```

### 6. 渲染/测试（启用偏心投影）
```bash
python render.py -m 模型路径 --ocmodel
```

**重要说明**：
- 使用 `--ocmodel` 参数启用偏心投影模式
- 如果不使用该参数，即使有 `fov_results.txt` 文件，系统也会使用标准投影
- 系统支持混合模式：当 `--ocmodel` 启用时，只有具有有效偏心参数的相机会使用偏心投影，其他相机仍使用标准投影

## 技术细节

### 偏心投影矩阵计算
新的投影矩阵考虑了非对称的视场角，通过以下参数定义：
- `fovx_r`: 右边界视场角
- `fovx_l`: 左边界视场角  
- `fovy_t`: 上边界视场角
- `fovy_b`: 下边界视场角

### 投影模式选择逻辑
系统会根据以下规则自动选择投影模式：
1. 如果 `--ocmodel` 参数未启用，所有相机使用标准投影
2. 如果 `--ocmodel` 参数已启用：
   - 对于有有效偏心参数的相机（`fovx_r ≠ fovx_l` 或 `fovy_t ≠ fovy_b`），使用偏心投影
   - 对于没有有效偏心参数的相机，使用标准投影
   - 系统会自动统计并报告两种投影模式的使用情况

### 渲染优化
- 透明度重置阈值从0.01调整为0.005
- 支持自定义投影矩阵的3D高斯渲染
- 优化了视锥体裁剪逻辑

## 文件结构
```
ocScripts/
├── calcFov/
│   ├── calc_fov_oc.py            # FOV计算主程序
│   └── data-qianqian-body/       # 示例数据
└── offCenterAnno/
    ├── image_annotator.py         # 图像标注工具
    ├── crop_and_resize copy.py    # 图像处理工具
    └── 这里是获取图片中需要偏心投影的范围.txt

数据集结构：
数据集根目录/
├── images/                        # 图像文件夹
└── sparse/
    └── 0/
        ├── cameras.txt           # COLMAP相机参数
        ├── images.txt            # COLMAP图像参数
        ├── points3D.ply          # 3D点云
        └── fov_results.txt       # 偏心投影FOV参数（需要手动放置）
```

## 注意事项
- 确保安装了PyQt5依赖（用于标注工具）
- FOV参数文件（`fov_results.txt`）必须放在数据集的 `sparse/0/` 目录下
- 图片名称匹配：确保 `fov_results.txt` 中的图片名称与 COLMAP `images.txt` 中的名称一致
- 使用偏心投影时，必须添加 `--ocmodel` 命令行参数
- 系统支持混合模式，可以同时处理有偏心参数和无偏心参数的相机
- 标注工具支持批量处理，建议先在小数据集上测试
- 如果所有相机都没有有效的偏心参数，即使启用 `--ocmodel`，系统也会使用标准投影并给出警告
