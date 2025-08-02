#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@ Description:
@ Date     : 2024/05/20 17:20:00
@ Author   : sunyifan
@ Version  : 1.0
"""

import math
import numpy as np
from tqdm import tqdm
from loguru import logger
from math import sqrt, ceil
from scipy.spatial.transform import Rotation as R

from render_python import computeColorFromSH
from render_python import computeCov2D, computeCov3D, computeCov2D_fov
from render_python import transformPoint4x4, in_frustum
from render_python import getWorld2View2, getProjectionMatrix, ndc2Pix, in_frustum, get_proj_matrix_p,get_proj_matrix_p_t


class Rasterizer:
    def __init__(self) -> None:
        pass

    def forward(
        self,
        P,  # int, num of guassians
        D,  # int, degree of spherical harmonics
        M,  # int, num of sh base function
        background,  # color of background, default black
        width,  # int, width of output image
        height,  # int, height of output image
        means3D,  # ()center position of 3d gaussian
        shs,  # spherical harmonics coefficient
        colors_precomp,
        opacities,  # opacities
        scales,  # scale of 3d gaussians
        scale_modifier,  # default 1
        rotations,  # rotation of 3d gaussians
        cov3d_precomp,
        viewmatrix,  # matrix for view transformation
        projmatrix,  # *(4, 4), matrix for transformation, aka mvp
        cam_pos,  # position of camera
        tan_fovx,  # float, tan value of fovx
        tan_fovy,  # float, tan value of fovy
        prefiltered,
        fovx_r=None,  # float, right fov boundary
        fovx_l=None,  # float, left fov boundary
        fovy_t=None,  # float, top fov boundary
        fovy_b=None,  # float, bottom fov boundary
    ) -> None:

        focal_y = height / (2 * tan_fovy)  # focal of y axis
        focal_x = width / (2 * tan_fovx)

        # run preprocessing per-Gaussians
        # transformation, bounding, conversion of SHs to RGB
        logger.info("Starting preprocess per 3d gaussian...")
        preprocessed = self.preprocess(
            P,
            D,
            M,
            means3D,
            scales,
            scale_modifier,
            rotations,
            opacities,
            shs,
            viewmatrix,
            projmatrix,
            cam_pos,
            width,
            height,
            focal_x,
            focal_y,
            tan_fovx,
            tan_fovy,
            fovx_r,
            fovx_l,
            fovy_t,
            fovy_b,
        )

        # produce [depth] key and corresponding guassian indices
        # sort indices by depth
        depths = preprocessed["depths"]
        point_list = np.argsort(depths)

        # render
        logger.info("Starting render...")
        out_color = self.render(
            point_list,
            width,
            height,
            preprocessed["points_xy_image"],
            preprocessed["rgbs"],
            preprocessed["conic_opacity"],
            background,
        )
        return out_color

    def preprocess(
        self,
        P,
        D,
        M,
        orig_points,
        scales,
        scale_modifier,
        rotations,
        opacities,
        shs,
        viewmatrix,
        projmatrix,
        cam_pos,
        W,
        H,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        fovx_r=None,
        fovx_l=None,
        fovy_t=None,
        fovy_b=None,
    ):

        rgbs = []  # rgb colors of gaussians
        cov3Ds = []  # covariance of 3d gaussians
        depths = []  # depth of 3d gaussians after view&proj transformation
        radii = []  # radius of 2d gaussians
        conic_opacity = []  # covariance inverse of 2d gaussian and opacity
        points_xy_image = []  # mean of 2d guassians
        for idx in range(P):
            # make sure point in frustum
            p_orig = orig_points[idx]
            p_view = in_frustum(p_orig, viewmatrix)
            if p_view is None:
                continue
            depths.append(p_view[2])

            # transform point, from world to ndc
            # Notice, projmatrix already processed as mvp matrix
            p_hom = transformPoint4x4(p_orig, projmatrix)  # 转换到ndc平面
            p_w = 1 / (p_hom[3] + 0.0000001)
            p_proj = [p_hom[0] * p_w, p_hom[1] * p_w, p_hom[2] * p_w]  

            # compute 3d covarance by scaling and rotation parameters
            scale = scales[idx]
            rotation = rotations[idx]
            cov3D = computeCov3D(scale, scale_modifier, rotation)
            cov3Ds.append(cov3D)

            # compute 2D screen-space covariance matrix
            # based on splatting, -> JW Sigma W^T J^T
            if fovx_r is not None and fovx_l is not None and fovy_t is not None and fovy_b is not None:
                cov = computeCov2D_fov(
                    p_orig, focal_x, focal_y, W, H, fovx_r, fovx_l, fovy_t, fovy_b, cov3D, viewmatrix
                )
            else:
                cov = computeCov2D(
                    p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix
                )

            # invert covarance(EWA splatting)
            det = cov[0] * cov[2] - cov[1] * cov[1]
            if det == 0:
                depths.pop()
                cov3Ds.pop()
                continue
            det_inv = 1 / det
            conic = [cov[2] * det_inv, -cov[1] * det_inv, cov[0] * det_inv]
            conic_opacity.append([conic[0], conic[1], conic[2], opacities[idx]])

            # compute radius, by finding eigenvalues of 2d covariance
            # transfrom point from NDC to Pixel
            mid = 0.5 * (cov[0] + cov[2])
            lambda1 = mid + sqrt(max(0.1, mid * mid - det))
            lambda2 = mid - sqrt(max(0.1, mid * mid - det))
            my_radius = ceil(3 * sqrt(max(lambda1, lambda2)))
            point_image = [ndc2Pix(p_proj[0], W), ndc2Pix(p_proj[1], H)]

            radii.append(my_radius)
            points_xy_image.append(point_image)

            # convert spherical harmonics coefficients to RGB color
            sh = shs[idx]
            result = computeColorFromSH(D, p_orig, cam_pos, sh)
            rgbs.append(result)

        return dict(
            rgbs=rgbs,
            cov3Ds=cov3Ds,
            depths=depths,
            radii=radii,
            conic_opacity=conic_opacity,
            points_xy_image=points_xy_image,
        )

    def render(
        self, point_list, W, H, points_xy_image, features, conic_opacity, bg_color
    ):

        out_color = np.zeros((H, W, 3))
        pbar = tqdm(range(H * W))

        # loop pixel
        for i in range(H):
            for j in range(W):
                pbar.update(1)
                pixf = [i, j]
                C = [0, 0, 0]

                # loop gaussian
                for idx in point_list:

                    # init helper variables, transmirrance
                    T = 1

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    xy = points_xy_image[idx]  # center of 2d gaussian
                    d = [
                        xy[0] - pixf[0],
                        xy[1] - pixf[1],
                    ]  # distance from center of pixel
                    con_o = conic_opacity[idx]
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )
                    if power > 0:
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    alpha = min(0.99, con_o[3] * np.exp(power))
                    if alpha < 1 / 255:
                        continue
                    test_T = T * (1 - alpha)
                    if test_T < 0.0001:
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    color = features[idx]
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * T

                    T = test_T

                # get final color
                for ch in range(3):
                    out_color[i,j, ch] = C[ch] + T * bg_color[ch]

        return out_color


if __name__ == "__main__":
    # 定义“笑脸”高斯点
    pts = np.array([
        [0.5, 1.0, -2],   # 右眼
        [-0.5, 1.0, -2],  # 左眼
        [0.0, 0.3, -2],   # 鼻子
        [0.5, -0.5, -2],  # 嘴右
        [0.25, -0.7, -2], # 嘴右下
        [0.0, -0.8, -2],  # 嘴中
        [-0.25, -0.7, -2],# 嘴左下
        [-0.5, -0.5, -2], # 嘴左
    ])
    n = len(pts)
    shs = np.zeros((n, 16, 3))

    # 不同颜色和方向性
    shs[0, 0, 2] = 1.0      # 右眼：蓝色
    shs[0, 1, 2] = 0.7      # 右眼：蓝色方向性

    shs[1, 0, 1] = 1.0      # 左眼：绿色
    shs[1, 2, 1] = 0.7      # 左眼：绿色方向性

    shs[2, 0, 0] = 1.0      # 鼻子：黄色
    shs[2, 0, 1] = 1.0
    shs[2, 1, 0] = 0.2

    shs[3, 0, 0] = 1.0      # 嘴右：红色
    shs[3, 1, 0] = 1.0
    shs[3, 2, 0] = 1.0

    shs[4, 0, 1] = 1.0      # 嘴右下：青色
    shs[4, 0, 2] = 1.0
    shs[4, 1, 2] = 0.5

    shs[5, 0, 0] = 1.0      # 嘴中：紫色
    shs[5, 0, 2] = 1.0
    shs[5, 2, 0] = 0.5

    shs[6, 0, 0] = 1.0      # 嘴左下：橙色
    shs[6, 0, 1] = 0.5
    shs[6, 1, 1] = 0.5

    shs[7, 0, 0] = 1.0      # 嘴左：粉色
    shs[7, 0, 1] = 0.5
    shs[7, 0, 2] = 0.5
    shs[7, 2, 2] = 0.5

    # 设置不同朝向
    rotations = []
    angles = [0, 45, 90, 135, 180, 225, 270, 315]  # 每个高斯点不同旋转角度
    for angle in angles:
        rot = R.from_euler('z', angle, degrees=True).as_matrix()
        rotations.append(rot)
    rotations = np.array(rotations)

    scales = np.ones((n, 3)) * 2  # 高斯点更大

    opacities = np.ones((n, 1))

    # set camera
    cam_pos = np.array([0, 0, 5])
    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    viewmatrix = getWorld2View2(R=R, t=cam_pos)

    width=700
    height=700
    x_img=150
    y_img=150
    #Xo = -(x_img - width / 2)
    #Yo = -(y_img - height / 2)
    #W=700
    #H=700
    fx=700
    fy=700
    near=0.01
    far=100
    #proj_param = {"Xo":Xo, "Yo": Yo, "W": 700, "H": 700,"fx":fx,"fy":fy,"near":0.01,"far":100} #Xo, Yo, W, H, fx, fy, near, far
    #projmatrix = get_proj_matrix_p(**proj_param) # 偏心矩阵
     # 读取fov_results-test.txt，获取序号0的fovx_r, fovx_l, fovy_t, fovy_b
    fov_file = 'fov_results-test.txt'
    fovx_r = fovx_l = fovy_t = fovy_b = None
    with open(fov_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 6 and parts[0] == '0':
                # parts: 0, 文件名, fovx_r, fovx_l, fovy_t, fovy_b
                fovx_r = float(parts[2])
                fovx_l = float(parts[3])
                fovy_t = float(parts[4])
                fovy_b = float(parts[5])
                break 
    proj_param = {"fovx_r":fovx_r, "fovx_l": fovx_l, "fovy_t": fovy_t, "fovy_b": fovy_b,"near":0.01,"far":100}
    projmatrix = get_proj_matrix_p_t(**proj_param) 

    fovx_rad = 2 * np.arctan(700 / (2 * fx)) #原视场角 用于计算焦距
    fovy_rad = 2 * np.arctan(700 / (2 * fy)) #原视场角 用于计算焦距
    #fovx_deg = np.degrees(fovx_rad)
    #fovy_deg = np.degrees(fovy_rad)
    #proj_param = {"znear": 0.01, "zfar": 100, "fovX": fovx_rad, "fovY": fovy_rad}
    #projmatrix = getProjectionMatrix(**proj_param)

    #import pdb;pdb.set_trace()

    projmatrix = np.dot(projmatrix, viewmatrix)


    fovx_rad = 2 * np.arctan(700 / (2 * fx)) #原视场角 用于计算焦距
    fovy_rad = 2 * np.arctan(700 / (2 * fy)) #原视场角 用于计算焦距
    fovx_deg = np.degrees(fovx_rad)
    fovy_deg = np.degrees(fovy_rad)

    tanfovx = math.tan(fovx_deg * 0.5) 
    tanfovy = math.tan(fovy_deg * 0.5)



    # 用读取到的fovx_r, fovx_l, fovy_t, fovy_b

    #import pdb;pdb.set_trace()
    # render
    rasterizer = Rasterizer()
    out_color = rasterizer.forward(
        P=len(pts),
        D=3,
        M=16,
        background=np.array([0, 0, 0]),
        width=400,
        height=300,
        means3D=pts,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        scale_modifier=1,
        rotations=rotations,
        cov3d_precomp=None,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        cam_pos=cam_pos,
        tan_fovx=tanfovx,
        tan_fovy=tanfovy,
        prefiltered=None,
        fovx_r=fovx_r,
        fovx_l=fovx_l,
        fovy_t=fovy_t,
        fovy_b=fovy_b,
    )

    import matplotlib.pyplot as plt

    plt.imshow(out_color)
    plt.show()
