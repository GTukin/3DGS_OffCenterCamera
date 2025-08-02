import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox,
                           QSpinBox, QMenu)
from PyQt5.QtCore import Qt, QRect, QPoint, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QCursor

class ImageAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image = None
        self.drawing = False
        self.moving = False
        self.resizing = False
        self.start_point = None
        self.end_point = None
        self.rectangles = []  # 存储所有矩形框 [(rect, class_name), ...]
        self.current_rect = None  # 当前正在移动的矩形框
        self.selected_rect_index = -1  # 当前选中的矩形框索引
        self.current_class = "default"
        self.box_width = 512  # 默认选框宽度
        self.box_height = 512  # 默认选框高度
        self.resize_handle_size = 10  # 调整大小的手柄尺寸
        self.scale_factor = 1.0  # 图片缩放比例
        self.image_offset = QPoint(0, 0)  # 图片在label中的偏移量
        
        # 图片文件夹相关
        self.image_folder = None
        self.save_folder = None
        self.current_image_index = -1
        self.image_files = []
        
        # 初始化UI
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('图片标注工具')
        self.setGeometry(100, 100, 2400, 1400)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()
        
        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        
        # 添加文件夹选择
        folder_layout = QHBoxLayout()
        self.image_folder_btn = QPushButton('选择图片文件夹')
        self.image_folder_btn.clicked.connect(self.select_image_folder)
        self.save_folder_btn = QPushButton('选择保存文件夹')
        self.save_folder_btn.clicked.connect(self.select_save_folder)
        folder_layout.addWidget(self.image_folder_btn)
        folder_layout.addWidget(self.save_folder_btn)
        control_layout.addLayout(folder_layout)
        
        # 添加图片导航
        nav_layout = QHBoxLayout()
        prev_btn = QPushButton('上一张 (A)')
        next_btn = QPushButton('下一张 (D)')
        prev_btn.clicked.connect(self.load_previous_image)
        next_btn.clicked.connect(self.load_next_image)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        control_layout.addLayout(nav_layout)
        
        # 添加当前图片信息
        self.image_info_label = QLabel('未加载图片')
        control_layout.addWidget(self.image_info_label)
        
        # 添加类别输入框
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText('输入类别名称')
        control_layout.addWidget(QLabel('类别名称:'))
        control_layout.addWidget(self.class_input)
        
        # 添加选框大小设置 - 分别设置宽度和高度
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel('选框宽度:'))
        self.width_input = QSpinBox()
        self.width_input.setRange(50, 2048)
        self.width_input.setValue(512)
        self.width_input.valueChanged.connect(self.update_box_width)
        size_layout.addWidget(self.width_input)
        control_layout.addLayout(size_layout)
        
        size_layout2 = QHBoxLayout()
        size_layout2.addWidget(QLabel('选框高度:'))
        self.height_input = QSpinBox()
        self.height_input.setRange(50, 2048)
        self.height_input.setValue(512)
        self.height_input.valueChanged.connect(self.update_box_height)
        size_layout2.addWidget(self.height_input)
        control_layout.addLayout(size_layout2)
        
        # 添加按钮
        create_box_btn = QPushButton('创建选框')
        create_box_btn.clicked.connect(self.create_box)
        control_layout.addWidget(create_box_btn)
        
        # 添加保存按钮
        save_btn = QPushButton('保存标注')
        save_btn.clicked.connect(self.save_current_annotations)
        control_layout.addWidget(save_btn)
        
        # 添加选框选项控件
        self.box_options_widget = QWidget()
        self.box_options_layout = QVBoxLayout()
        self.box_options_widget.setLayout(self.box_options_layout)
        self.box_options_widget.hide()  # 初始隐藏
        
        # 添加选框选项
        self.box_options_layout.addWidget(QLabel('选框选项:'))
        
        # 删除选框按钮
        delete_box_btn = QPushButton('删除选框')
        delete_box_btn.clicked.connect(self.delete_selected_box)
        self.box_options_layout.addWidget(delete_box_btn)
        
        # 调整大小按钮
        resize_box_btn = QPushButton('调整大小')
        resize_box_btn.clicked.connect(self.start_resize_box)
        self.box_options_layout.addWidget(resize_box_btn)
        
        # 添加类别选择
        self.box_options_layout.addWidget(QLabel('选择类别:'))
        self.box_class_input = QLineEdit()
        self.box_class_input.textChanged.connect(self.update_box_class)
        self.box_options_layout.addWidget(self.box_class_input)
        
        # 添加选框位置调整
        self.box_options_layout.addWidget(QLabel('选框位置:'))
        pos_layout = QHBoxLayout()
        self.pos_x_input = QSpinBox()
        self.pos_y_input = QSpinBox()
        self.pos_x_input.setRange(0, 10000)
        self.pos_y_input.setRange(0, 10000)
        self.pos_x_input.valueChanged.connect(self.update_box_position)
        self.pos_y_input.valueChanged.connect(self.update_box_position)
        pos_layout.addWidget(QLabel('X:'))
        pos_layout.addWidget(self.pos_x_input)
        pos_layout.addWidget(QLabel('Y:'))
        pos_layout.addWidget(self.pos_y_input)
        self.box_options_layout.addLayout(pos_layout)
        
        # 添加选框大小调整
        self.box_options_layout.addWidget(QLabel('选框尺寸:'))
        size_layout = QHBoxLayout()
        self.box_width_input = QSpinBox()
        self.box_height_input = QSpinBox()
        self.box_width_input.setRange(50, 1000)
        self.box_height_input.setRange(50, 1000)
        self.box_width_input.valueChanged.connect(self.update_box_dimensions)
        self.box_height_input.valueChanged.connect(self.update_box_dimensions)
        size_layout.addWidget(QLabel('宽:'))
        size_layout.addWidget(self.box_width_input)
        size_layout.addWidget(QLabel('高:'))
        size_layout.addWidget(self.box_height_input)
        self.box_options_layout.addLayout(size_layout)
        
        self.box_options_layout.addStretch()
        control_layout.addWidget(self.box_options_widget)
        
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # 右侧图片显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.contextMenuEvent)
        
        layout.addWidget(control_panel)
        layout.addWidget(self.image_label)
        
        main_widget.setLayout(layout)
        
        # 设置快捷键
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        """设置快捷键"""
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        
        # 上一张图片快捷键 (A)
        prev_shortcut = QShortcut(QKeySequence('A'), self)
        prev_shortcut.activated.connect(self.load_previous_image)
        
        # 下一张图片快捷键 (D)
        next_shortcut = QShortcut(QKeySequence('D'), self)
        next_shortcut.activated.connect(self.load_next_image)
        
        # 创建选框快捷键 (W)
        create_box_shortcut = QShortcut(QKeySequence('W'), self)
        create_box_shortcut.activated.connect(self.create_box)
        
    def select_image_folder(self):
        """选择图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder:
            self.image_folder = folder
            self.image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_files.sort()
            self.current_image_index = -1
            if self.image_files:
                self.load_next_image()
                
    def select_save_folder(self):
        """选择保存文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if folder:
            self.save_folder = folder
            
    def load_previous_image(self):
        """加载上一张图片"""
        if not self.image_files:
            return
            
        # 保存当前图片的标注
        self.save_current_annotations()
        
        # 加载上一张图片
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            
    def load_next_image(self):
        """加载下一张图片"""
        if not self.image_files:
            return
            
        # 保存当前图片的标注
        self.save_current_annotations()
        
        # 加载下一张图片
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_current_image()
            
    def load_current_image(self):
        """加载当前索引的图片"""
        if 0 <= self.current_image_index < len(self.image_files):
            file_name = os.path.join(self.image_folder, self.image_files[self.current_image_index])
            try:
                self.image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                if self.image is None:
                    raise Exception("无法读取图片")
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.rectangles = []  # 清空之前的标注
                self.load_annotations()  # 加载已有的标注
                self.display_image()
                self.update_image_info()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片失败：{str(e)}")
                
    def update_image_info(self):
        """更新图片信息显示"""
        if self.current_image_index >= 0:
            self.image_info_label.setText(f'当前图片: {self.current_image_index + 1}/{len(self.image_files)}')
            
    def save_current_annotations(self):
        """保存当前图片的标注"""
        if not self.rectangles or not self.save_folder or self.current_image_index < 0:
            return
            
        # 获取当前图片文件名（不含扩展名）
        current_image_name = os.path.splitext(self.image_files[self.current_image_index])[0]
        annotation_file = os.path.join(self.save_folder, f"{current_image_name}.txt")
        
        # 保存标注信息
        with open(annotation_file, 'w') as f:
            for rect, class_name in self.rectangles:
                x1, y1 = rect.left(), rect.top()
                # 使用left + width 和 top + height 来确保正确的宽度和高度
                x2 = x1 + rect.width()
                y2 = y1 + rect.height()
                f.write(f"{class_name},{x1},{y1},{x2},{y2}\n")
                
    def load_annotations(self):
        """加载已有的标注信息"""
        if not self.save_folder or self.current_image_index < 0:
            return
            
        # 获取当前图片文件名（不含扩展名）
        current_image_name = os.path.splitext(self.image_files[self.current_image_index])[0]
        annotation_file = os.path.join(self.save_folder, f"{current_image_name}.txt")
        
        # 如果存在标注文件，则加载
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) == 5:
                            class_name, x1, y1, x2, y2 = parts
                            rect = QRect(int(x1), int(y1), int(x2)-int(x1), int(y2)-int(y1))
                            self.rectangles.append((rect, class_name))
            except Exception as e:
                QMessageBox.warning(self, "警告", f"加载标注文件失败：{str(e)}")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            try:
                # 使用numpy读取图片，避免中文路径问题
                self.image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), cv2.IMREAD_COLOR)
                if self.image is None:
                    raise Exception("无法读取图片")
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                self.display_image()
                self.rectangles = []  # 清空之前的标注
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片失败：{str(e)}")
            
    def display_image(self):
        if self.image is not None:
            height, width = self.image.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # 计算缩放比例和偏移量
            label_size = self.image_label.size()
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio)
            self.scale_factor = scaled_pixmap.width() / pixmap.width()
            
            # 计算图片在label中的偏移量
            self.image_offset = QPoint(
                (label_size.width() - scaled_pixmap.width()) // 2,
                (label_size.height() - scaled_pixmap.height()) // 2
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            
            # 更新图片信息显示
            self.image_info_label.setText(f'图片尺寸: {width}x{height}, 选框大小: {self.box_width}x{self.box_height}')
            
    def update_box_width(self):
        self.box_width = self.width_input.value()
        
    def update_box_height(self):
        self.box_height = self.height_input.value()
        
    def create_box(self):
        if self.image is None:
            QMessageBox.warning(self, "警告", "请先加载图片")
            return
            
        # 获取图片的实际尺寸
        height, width = self.image.shape[:2]
        
        # 计算图片在label中的实际显示区域
        label_size = self.image_label.size()
        scaled_pixmap = self.image_label.pixmap()
        if scaled_pixmap is None:
            return
            
        # 计算图片在label中的偏移量
        x_offset = (label_size.width() - scaled_pixmap.width()) // 2
        y_offset = (label_size.height() - scaled_pixmap.height()) // 2
        
        # 计算缩放比例
        scale_factor = scaled_pixmap.width() / width
        
        # 在图片中心创建新选框
        center_x = (width // 2)
        center_y = (height // 2)
        half_width = self.box_width // 2  # 使用设置的box_width
        half_height = self.box_height // 2  # 使用设置的box_height
        
        # 创建选框，使用设置的宽度和高度
        rect = QRect(center_x - half_width, center_y - half_height, 
                    self.box_width, self.box_height)
        
        # 确保选框不会超出图片边界
        rect = rect.intersected(QRect(0, 0, width, height))
        
        self.current_rect = rect
        self.update_image()
        
    def contextMenuEvent(self, event):
        if self.image is None:
            return
            
        # 检查是否点击在某个选框内
        clicked_pos = self.map_to_image_coordinates(event)
        for i, (rect, _) in enumerate(self.rectangles):
            if rect.contains(clicked_pos):
                self.selected_rect_index = i
                self.current_rect = rect
                self.update_box_options()
                self.box_options_widget.show()
                self.update_image()
                return

    def update_box_options(self):
        """更新选框选项控件的值"""
        if self.selected_rect_index >= 0:
            rect, class_name = self.rectangles[self.selected_rect_index]
            self.box_class_input.setText(class_name)
            self.pos_x_input.setValue(rect.x())
            self.pos_y_input.setValue(rect.y())
            self.box_width_input.setValue(rect.width())
            self.box_height_input.setValue(rect.height())

    def delete_selected_box(self):
        """删除选中的选框"""
        if self.selected_rect_index >= 0:
            self.rectangles.pop(self.selected_rect_index)
            self.selected_rect_index = -1
            self.current_rect = None
            self.box_options_widget.hide()
            self.update_image()

    def start_resize_box(self):
        """开始调整选框大小"""
        if self.selected_rect_index >= 0:
            self.resizing = True
            self.current_rect = self.rectangles[self.selected_rect_index][0]

    def update_box_class(self):
        """更新选框类别"""
        if self.selected_rect_index >= 0:
            class_name = self.box_class_input.text()
            rect, _ = self.rectangles[self.selected_rect_index]
            self.rectangles[self.selected_rect_index] = (rect, class_name)
            self.update_image()

    def update_box_position(self):
        """更新选框位置"""
        if self.selected_rect_index >= 0:
            x = self.pos_x_input.value()
            y = self.pos_y_input.value()
            rect, class_name = self.rectangles[self.selected_rect_index]
            rect.moveTo(x, y)
            self.rectangles[self.selected_rect_index] = (rect, class_name)
            self.update_image()

    def update_box_dimensions(self):
        """更新选框尺寸"""
        if self.selected_rect_index >= 0:
            # 使用设置的box_width和box_height
            width = self.box_width_input.value()
            height = self.box_height_input.value()
            rect, class_name = self.rectangles[self.selected_rect_index]
            
            # 保持选框中心点不变
            center = rect.center()
            rect.setSize(QSize(width, height))
            rect.moveCenter(center)
            
            self.rectangles[self.selected_rect_index] = (rect, class_name)
            self.update_image()

    def map_to_image_coordinates(self, pos):
        """将鼠标位置映射到图片坐标系"""
        if self.image is None:
            return pos
        
        # 减去图片偏移量
        pos = pos - self.image_offset
        # 应用缩放比例
        return QPoint(int(pos.x() / self.scale_factor), int(pos.y() / self.scale_factor))

    def map_from_image_coordinates(self, pos):
        """将图片坐标系映射到显示坐标系"""
        if self.image is None:
            return pos
        
        # 应用缩放比例
        pos = QPoint(int(pos.x() * self.scale_factor), int(pos.y() * self.scale_factor))
        # 加上图片偏移量
        return pos + self.image_offset

    def mouse_press_event(self, event):
        if self.image is None:
            return
            
        clicked_pos = self.map_to_image_coordinates(event.pos())
        
        # 如果正在调整大小
        if self.resizing and self.current_rect is not None:
            self.start_point = clicked_pos
            self.original_rect = self.current_rect.copy()  # 保存原始矩形
            return
            
        # 如果正在移动选框
        if self.current_rect is not None:
            if self.current_rect.contains(clicked_pos):
                self.moving = True
                self.offset = clicked_pos - self.current_rect.topLeft()
            return
            
        # 检查是否点击在已保存的选框内
        for i, (rect, _) in enumerate(self.rectangles):
            if rect.contains(clicked_pos):
                self.selected_rect_index = i
                self.current_rect = rect
                self.moving = True
                self.offset = clicked_pos - rect.topLeft()
                self.update_image()
                return
                
        # 如果没有点击在任何选框内，取消选中状态
        self.selected_rect_index = -1
        self.update_image()
            
    def mouse_move_event(self, event):
        if self.image is None:
            return
            
        current_pos = self.map_to_image_coordinates(event.pos())
        
        # 调整大小
        if self.resizing and self.current_rect is not None and self.start_point is not None:
            # 使用设置的box_width和box_height
            self.box_width = self.box_width_input.value()
            self.box_height = self.box_height_input.value()
            self.width_input.setValue(self.box_width)
            self.height_input.setValue(self.box_height)
            
            # 保持选框中心不变
            center = self.original_rect.center()
            self.current_rect.setSize(QSize(self.box_width, self.box_height))
            self.current_rect.moveCenter(center)
            
            self.update_image()
            return
            
        # 移动选框
        if self.moving and self.current_rect is not None:
            new_pos = current_pos - self.offset
            # 确保选框不会超出图片边界
            height, width = self.image.shape[:2]
            new_pos.setX(max(0, min(new_pos.x(), width - self.current_rect.width())))
            new_pos.setY(max(0, min(new_pos.y(), height - self.current_rect.height())))
            
            self.current_rect.moveTopLeft(new_pos)
            self.update_image()
            
    def mouse_release_event(self, event):
        if self.moving:
            self.moving = False
            if self.selected_rect_index >= 0 and self.selected_rect_index < len(self.rectangles):
                # 更新已保存的选框
                self.rectangles[self.selected_rect_index] = (self.current_rect, 
                    self.rectangles[self.selected_rect_index][1])
            else:
                # 添加新的选框
                class_name = self.class_input.text() if self.class_input.text() else "default"
                self.rectangles.append((self.current_rect, class_name))
            self.current_rect = None
            self.update_image()
        elif self.resizing:
            self.resizing = False
            if self.selected_rect_index >= 0 and self.selected_rect_index < len(self.rectangles):
                # 更新已保存的选框
                self.rectangles[self.selected_rect_index] = (self.current_rect, 
                    self.rectangles[self.selected_rect_index][1])
            self.current_rect = None
            self.update_image()
            
    def update_image(self):
        if self.image is not None:
            # 创建图片副本用于绘制
            display_image = self.image.copy()
            
            # 计算线条粗细（根据图片尺寸自适应）
            height, width = display_image.shape[:2]
            line_thickness = max(2, int(min(width, height) / 500))
            
            # 绘制所有已保存的矩形框
            for i, (rect, class_name) in enumerate(self.rectangles):
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()
                color = (0, 0, 255) if i == self.selected_rect_index else (0, 255, 0)
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, line_thickness)
                font_scale = max(0.5, min(width, height) / 1000)
                cv2.putText(display_image, class_name, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, line_thickness)
                
                if i == self.selected_rect_index:
                    handle_size = max(10, int(min(width, height) / 100))
                    cv2.rectangle(display_image, 
                                (x2-handle_size, y2-handle_size), 
                                (x2, y2), 
                                (255, 0, 0), -1)
            
            # 绘制当前正在移动的矩形框
            if self.current_rect is not None:
                x1, y1 = self.current_rect.left(), self.current_rect.top()
                x2, y2 = self.current_rect.right(), self.current_rect.bottom()
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 0), line_thickness)
                
                if self.resizing:
                    handle_size = max(10, int(min(width, height) / 100))
                    cv2.rectangle(display_image, 
                                (x2-handle_size, y2-handle_size), 
                                (x2, y2), 
                                (255, 0, 0), -1)
            
            # 显示更新后的图片
            height, width = display_image.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(scaled_pixmap)
            
    def save_annotations(self):
        if not self.rectangles:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(self, "保存标注", "", "Text Files (*.txt)")
        if file_name:
            with open(file_name, 'w') as f:
                for rect, class_name in self.rectangles:
                    x1, y1 = rect.left(), rect.top()
                    x2, y2 = rect.right(), rect.bottom()
                    f.write(f"{class_name},{x1},{y1},{x2},{y2}\n")
                    
            # 保存带标注的图片
            image_name = file_name.rsplit('.', 1)[0] + '_annotated.jpg'
            display_image = self.image.copy()
            for rect, class_name in self.rectangles:
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_image, class_name, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(image_name, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageAnnotator()
    ex.show()
    sys.exit(app.exec_()) 