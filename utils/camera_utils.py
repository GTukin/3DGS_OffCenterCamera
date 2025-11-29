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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from utils.general_utils import PILtoTorch
from PIL import Image
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  fovx_r=cam_info.fovx_r, fovx_l=cam_info.fovx_l, fovy_t=cam_info.fovy_t, fovy_b=cam_info.fovy_b,
                  ocmodel=args.ocmodel)


def cameraList_from_camInfos(cam_infos, resolution_scale, args, camera_set_name=""):
    camera_list = []
    offcenter_proj_count = 0  # 统计实际使用偏心投影的相机数量
    standard_proj_count = 0    # 统计使用标准投影的相机数量
    
    for id, c in enumerate(cam_infos):
        camera = loadCam(args, id, c, resolution_scale)
        camera_list.append(camera)
        
        # 统计实际使用的投影类型
        if camera.using_offcenter_projection:
            offcenter_proj_count += 1
        else:
            standard_proj_count += 1

    # 输出最终统计信息
    total_cameras = len(camera_list)
    
    # 构建相机集标识前缀
    set_prefix = f"[{camera_set_name}] " if camera_set_name else ""
    
    if args.ocmodel:
        if total_cameras == 0:
            return camera_list  # 测试集为空时直接返回，不显示统计信息
        
        if offcenter_proj_count > 0 and standard_proj_count > 0:
            print(f"[混合模式] {set_prefix}ocmodel已启用，{offcenter_proj_count}个相机使用偏心投影，{standard_proj_count}个相机使用标准投影")
        elif offcenter_proj_count > 0:
            print(f"[投影模式] {set_prefix}ocmodel已启用，所有{offcenter_proj_count}个相机都使用偏心投影")
        else:
            print(f"[投影模式] {set_prefix}ocmodel已启用，{total_cameras}个相机使用标准投影（无有效偏心参数）")
    else:
        if offcenter_proj_count == 0:
            print(f"[投影模式] {set_prefix}ocmodel未启用，所有{standard_proj_count}个相机都使用标准投影")
        else:
            print(f"[异常] {set_prefix}ocmodel未启用，但检测到{offcenter_proj_count}个相机使用偏心投影（这不应该发生）")

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
