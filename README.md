# 偏心投影功能更新说明

## 概述
项目新增了偏心投影（Off-center Projection）功能，用于处理非中心投影的3D高斯渲染。

## 主要新增功能

### 1. 偏心投影计算模块
- **文件位置**: `ocScripts/calcFov/3dgs_calc_fov.py`
- **功能**: 计算偏心投影的FOV参数（r, l, b, t）
- **核心特性**:
  - 支持偏心投影矩阵计算
  - 实现自定义的渲染器类 `Rasterizer`
  - 支持FOV边界参数（fovx_r, fovx_l, fovy_t, fovy_b）

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
- 集成新的投影矩阵计算函数 `getProjectionMatrix_p`

### 2. 数据集读取器更新 (`scene/dataset_readers.py`)
- 支持读取 `fov_results.txt` 文件
- 新增FOV参数到 `CameraInfo` 结构
- 修改相机读取逻辑以支持偏心投影

### 3. 图形工具更新 (`utils/graphics_utils.py`)
- 新增 `getProjectionMatrix_p` 函数
- 实现偏心投影矩阵计算

### 4. 训练脚本更新 (`train.py`)
- 修复模型加载参数
- 优化透明度重置阈值(0.01—>0.005)

## 数据格式

### FOV结果文件格式 (`fov_results.txt`)
```
索引,图片名,fovx_r,fovx_l,fovy_t,fovy_b,宽度,高度
```
示例：
```
0,40.png,0.275951,-0.236702,0.309598,-0.538812,400.00,700.00
```

## 使用方法

### 1. 计算FOV参数
```bash
cd ocScripts/calcFov
python 3dgs_calc_fov.py
```

### 2. 标注图像
```bash
cd ocScripts/offCenterAnno
python image_annotator.py
```

### 3. 处理标注数据
```bash
python crop_and_resize copy.py
```

## 技术细节

### 偏心投影矩阵计算
新的投影矩阵考虑了非对称的视场角，通过以下参数定义：
- `fovx_r`: 右边界视场角
- `fovx_l`: 左边界视场角  
- `fovy_t`: 上边界视场角
- `fovy_b`: 下边界视场角

### 渲染优化
- 透明度重置阈值从0.01调整为0.005
- 支持自定义投影矩阵的3D高斯渲染
- 优化了视锥体裁剪逻辑

## 文件结构
```
ocScripts/
├── calcFov/
│   ├── 3dgs_calc_fov.py          # FOV计算主程序
│   └── data-qianqian-body/       # 示例数据
└── offCenterAnno/
    ├── image_annotator.py         # 图像标注工具
    ├── crop_and_resize copy.py    # 图像处理工具
    └── 这里是获取图片中需要偏心投影的范围.txt
```

## 注意事项
- 确保安装了PyQt5依赖（用于标注工具）
- FOV参数文件需要放在正确的路径下
- 标注工具支持批量处理，建议先在小数据集上测试