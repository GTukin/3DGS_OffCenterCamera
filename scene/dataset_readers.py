#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fovx_r: float = 0.0
    fovx_l: float = 0.0
    fovy_t: float = 0.0
    fovy_b: float = 0.0

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, fov_results_dict):
    cam_infos = []
    offcenter_count = 0  # 统计有偏心参数的相机数量
    standard_count = 0   # 统计无偏心参数的相机数量
    matched_count = 0    # 统计成功匹配fov_results的相机数量
    unmatched_names = [] # 记录未匹配的图片名称（最多记录前5个）
    
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        # 尝试精确匹配和去除路径的匹配
        img_name_from_colmap = extr.name
        fov_data = fov_results_dict.get(img_name_from_colmap)
        
        # 如果精确匹配失败，尝试只匹配文件名（去除路径）
        if fov_data is None:
            img_name_basename = os.path.basename(img_name_from_colmap)
            fov_data = fov_results_dict.get(img_name_basename)
            if fov_data is not None:
                # 更新字典，使用basename作为key，方便后续匹配
                fov_results_dict[img_name_from_colmap] = fov_data
        
        if fov_data:
            matched_count += 1
        elif len(unmatched_names) < 5:
            unmatched_names.append(img_name_from_colmap)
        intr = cam_intrinsics[extr.camera_id]
        fovx_r = fov_data['fovx_r'] if fov_data else 0.0
        fovx_l = fov_data['fovx_l'] if fov_data else 0.0
        fovy_t = fov_data['fovy_t'] if fov_data else 0.0
        fovy_b = fov_data['fovy_b'] if fov_data else 0.0
        width_p = fov_data['width_p'] if fov_data and 'width_p' in fov_data else 0.0
        height_p = fov_data['height_p'] if fov_data and 'height_p' in fov_data else 0.0
        
        # 检查是否有有效的偏心参数
        has_valid_offcenter = (abs(fovx_r - fovx_l) > 1e-6) or (abs(fovy_t - fovy_b) > 1e-6)
        if has_valid_offcenter:
            offcenter_count += 1
        else:
            standard_count += 1
        
        # 如果有偏心参数，使用fov_results.txt中的宽高；否则使用COLMAP的原始宽高
        if fov_data and width_p > 0 and height_p > 0:
            height = height_p
            width = width_p
        else:
            # 没有偏心参数时，使用COLMAP的原始宽高
            height = intr.height
            width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              fovx_r=fovx_r, fovx_l=fovx_l, fovy_t=fovy_t, fovy_b=fovy_b)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    
    # 输出核心统计信息
    total_cameras = len(cam_infos)
    
    if matched_count == 0 and len(fov_results_dict) > 0:
        print(f"[参数检查] ⚠ 警告: 没有匹配到任何偏心参数！请检查fov_results.txt中的图片名称是否与COLMAP images.txt中的名称一致")
    
    if offcenter_count > 0 and standard_count > 0:
        print(f"[混合模式] ✓ 检测到混合模式：{offcenter_count}个相机有偏心参数，{standard_count}个相机无偏心参数")
    elif offcenter_count > 0:
        print(f"[投影模式] 所有{offcenter_count}个相机都有偏心参数，将使用偏心投影")
    elif len(fov_results_dict) > 0:
        print(f"[投影模式] 所有{total_cameras}个相机都没有偏心参数，将使用标准投影")
    
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    # 读取fov_results.txt
    fov_results_path = os.path.join(path, "sparse/0", "fov_results.txt")
    fov_results_dict = {}
    fov_results_count = 0
    fov_results_skipped = 0
    fov_image_names = set()  # 收集fov_results.txt中的所有图片名称
    
    if os.path.exists(fov_results_path):
        with open(fov_results_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # 跳过空行和注释行
                    continue
                parts = line.split(',')
                # 支持两种格式：
                # 格式1（8列，带索引）: 索引,图片名,fovx_r,fovx_l,fovy_t,fovy_b,宽度,高度
                # 格式2（7列，无索引）: 图片名,fovx_r,fovx_l,fovy_t,fovy_b,宽度,高度
                if len(parts) == 8:
                    try:
                        idx, img_name, fovx_r, fovx_l, fovy_t, fovy_b, width_p, height_p = parts
                        # 索引字段被读取但不使用，匹配仅通过图片名称进行
                    except (ValueError, IndexError) as e:
                        fov_results_skipped += 1
                        continue
                elif len(parts) == 7:
                    try:
                        # 无索引格式：直接跳过第一列（索引）
                        img_name, fovx_r, fovx_l, fovy_t, fovy_b, width_p, height_p = parts
                    except (ValueError, IndexError) as e:
                        fov_results_skipped += 1
                        continue
                else:
                    fov_results_skipped += 1
                    continue
                
                # 处理数据（两种格式统一处理）
                try:
                    # 去除图片名称中的空格
                    img_name = img_name.strip()
                    fov_image_names.add(img_name)  # 添加到集合中
                    fov_results_dict[img_name] = {
                        'fovx_r': float(fovx_r),
                        'fovx_l': float(fovx_l),
                        'fovy_t': float(fovy_t),
                        'fovy_b': float(fovy_b),
                        'width_p': float(width_p),
                        'height_p': float(height_p)
                    }
                    fov_results_count += 1
                except (ValueError, IndexError) as e:
                    fov_results_skipped += 1
        if fov_results_count > 0:
            print(f"[读取fov_results.txt] 成功读取: {fov_results_count}条记录" + (f", 跳过: {fov_results_skipped}条" if fov_results_skipped > 0 else ""))
    else:
        print(f"[读取fov_results.txt] 文件不存在: {fov_results_path}")
    
    # 收集COLMAP images中的所有图片名称（用于匹配检查）
    colmap_image_names_full = {}  # 完整路径 -> 原始名称
    colmap_image_names_basename = {}  # basename -> 原始名称列表
    if cam_extrinsics:
        for extr in cam_extrinsics.values():
            full_name = extr.name
            basename = os.path.basename(full_name)
            colmap_image_names_full[full_name] = full_name
            # 收集所有basename对应的原始名称
            if basename not in colmap_image_names_basename:
                colmap_image_names_basename[basename] = []
            colmap_image_names_basename[basename].append(full_name)
    
    # 匹配检查：检测fov_results.txt中的图片名称是否能在COLMAP images中找到
    if fov_image_names and cam_extrinsics:
        matched_count_check = 0
        for fov_name in fov_image_names:
            # 方法1: 精确匹配（完整路径）
            if fov_name in colmap_image_names_full:
                matched_count_check += 1
            else:
                # 方法2: basename匹配
                fov_basename = os.path.basename(fov_name)
                if fov_basename in colmap_image_names_basename:
                    matched_count_check += 1
        
        if matched_count_check < len(fov_image_names):
            print(f"[匹配检查] ⚠ 警告: fov_results.txt中有 {len(fov_image_names) - matched_count_check} 个图片名称无法在COLMAP images.txt中找到匹配")
        # 匹配成功时不输出，避免冗余信息
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir),fov_results_dict=fov_results_dict)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
