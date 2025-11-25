import numpy as np
import cv2
from pathlib import Path
from scipy import optimize  # 实际没直接用到，留着也没事
from visualize import visualize_matches, visualize_epipolar_lines
from load_data import load_dataset
from utils import *

# ===================== 主流程 =====================

def main():
    # ===== 路径设置 =====
    DATA_DIR = "./PRCV/data"  # 请确保该文件夹下有 00000022/23/24.jpg 和 *_cam.txt
    OUTPUT_ROOT = Path("./PRCV/sfm_output")
    MATCH_DIR = OUTPUT_ROOT / "特征匹配结果"
    RECONS_DIR = OUTPUT_ROOT / "重建结果"

    OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)
    MATCH_DIR.mkdir(exist_ok=True, parents=True)
    RECONS_DIR.mkdir(exist_ok=True, parents=True)

    # ===== 1. 读取数据 =====
    images, cams_ext_gt, K = load_dataset(DATA_DIR)
    print("Loaded images and camera parameters.")
    print("Intrinsic K:\n", K)

    # GT 外参：假设文件里的 extrinsic 是 world->camera [R|t]
    R_gt = [ext[:3, :3] for ext in cams_ext_gt]
    t_gt = [ext[:3, 3] for ext in cams_ext_gt]

    # ===== 2. 特征提取 + 描述子（Harris + patch） =====
    keypoints = []
    descriptors = []

    for idx, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, resp = harris_corners(
            gray,
            window_size=5,
            k=0.04,
            thresh_rel=0.01,
            nms_radius=4,
            max_points=2000,
        )
        print(f"Image {idx}: detected {len(kps)} Harris corners.")
        kps, desc = build_patch_descriptors(gray, kps, patch_size=11)
        print(f"Image {idx}: valid keypoints with descriptors = {len(kps)}")
        keypoints.append(kps)
        descriptors.append(desc)

    # ===== 3. 特征匹配（brute force + Lowe ratio） =====
    pairs = [(0, 1), (0, 2), (1, 2)]
    matches_dict = {}
    for i, j in pairs:
        m = match_descriptors(descriptors[i], descriptors[j], ratio=0.75)
        print(f"Raw matches between {i}-{j}: {len(m)}")
        matches_dict[(i, j)] = m

    # ===== 4. 几何验证：对每一对图像做 RANSAC-F，删除外点 =====
    F_dict = {}
    inliers_dict = {}
    matches_inliers_dict = {}

    for (i, j) in pairs:
        matches_ij = matches_dict[(i, j)]
        if len(matches_ij) < 8:
            print(f"Not enough matches for RANSAC between {i}-{j}.")
            continue

        pts_i = np.array([keypoints[i][m[0]] for m in matches_ij], dtype=np.float64)
        pts_j = np.array([keypoints[j][m[1]] for m in matches_ij], dtype=np.float64)

        F_ij, inliers_ij = ransac_fundamental(pts_i, pts_j, num_iters=2000, thresh=1e-3)
        F_dict[(i, j)] = F_ij
        inliers_dict[(i, j)] = inliers_ij

        matches_inliers = [matches_ij[k] for k, ok in enumerate(inliers_ij) if ok]
        matches_inliers_dict[(i, j)] = matches_inliers

        print(
            f"RANSAC for pair {i}-{j}: inliers {inliers_ij.sum()} / {len(inliers_ij)}"
        )
        print(f"Estimated F{i}{j}:\n", F_ij)

    # 必须要有 0-1 这一对作为初始化
    if (0, 1) not in F_dict:
        raise RuntimeError("Pair (0,1) does not have a valid fundamental matrix.")

    # ===== 5. 可视化每一对的匹配 & 极线（作业图2 / 图3） =====
    for (i, j) in pairs:
        img_i, img_j = images[i], images[j]
        kps_i, kps_j = keypoints[i], keypoints[j]

        # 若有 RANSAC 内点，则用内点匹配；否则用全部匹配
        matches_to_draw = matches_inliers_dict.get((i, j), matches_dict[(i, j)])

        # 5.1 匹配可视化：0_1.jpg, 0_2.jpg, 1_2.jpg
        match_path = MATCH_DIR / f"{i}_{j}.jpg"
        visualize_matches(
            img_i,
            img_j,
            kps_i,
            kps_j,
            matches_to_draw,
            max_draw=200,
            save_path=match_path,
        )

        # 5.2 极线可视化：Epi0_1.jpg, Epi0_2.jpg, Epi1_2.jpg
        F_ij = F_dict.get((i, j), None)
        if F_ij is not None and len(matches_to_draw) > 0:
            pts_i = np.array(
                [kps_i[m[0]] for m in matches_to_draw], dtype=np.float64
            )
            pts_j = np.array(
                [kps_j[m[1]] for m in matches_to_draw], dtype=np.float64
            )
            epi_path = MATCH_DIR / f"Epi{i}_{j}.jpg"
            visualize_epipolar_lines(
                img_i,
                img_j,
                pts_i,
                pts_j,
                F_ij,
                num_lines=20,
                save_path=epi_path,
            )

    # 下面的 SfM 流程以 0-1 这一对做初始化
    matches_01_all = matches_dict[(0, 1)]
    inliers01 = inliers_dict[(0, 1)]
    F01 = F_dict[(0, 1)]

    if inliers01.sum() < 8:
        raise RuntimeError("Not enough inliers between image 0 and 1.")

    pts0_all = np.array(
        [keypoints[0][m[0]] for m in matches_01_all], dtype=np.float64
    )
    pts1_all = np.array(
        [keypoints[1][m[1]] for m in matches_01_all], dtype=np.float64
    )

    pts0_in = pts0_all[inliers01]
    pts1_in = pts1_all[inliers01]

    print("Using pair 0-1 for initialization.")
    print("Estimated F01:\n", F01)
    print(f"RANSAC inliers between 0-1: {inliers01.sum()} / {len(inliers01)}")

    # ===== 6. 由 F 得到 E，再分解得到相机 1 的位姿（相对相机 0） =====
    E01 = K.T @ F01 @ K
    Ue, Se, Vte = np.linalg.svd(E01)
    Se[2] = 0.0
    E01 = Ue @ np.diag(Se) @ Vte
    print("Estimated E01:\n", E01)

    R1, t1 = decompose_essential(E01, K, pts0_in, pts1_in)
    print("Initial pose for camera 1 (relative to camera 0):")
    print("R1 =\n", R1)
    print("t1 =\n", t1)

    # 相机 0 作为世界坐标: [I | 0]
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.zeros(3, dtype=np.float64)

    # ===== 7. 三角化 0-1 内点，得到初始 3D 点云 =====
    P0 = np.hstack([R0, t0.reshape(3, 1)])
    P1 = np.hstack([R1, t1.reshape(3, 1)])

    pts0_n = normalize_pixel_points(pts0_in, K)
    pts1_n = normalize_pixel_points(pts1_in, K)

    points_3d = []
    colors_3d = []
    obs_cam_idx = []
    obs_pt_idx = []
    obs_uv = []

    map0 = {}  # key: kp index in img0, val: 3D point index
    map1 = {}

    img0 = images[0]
    inlier_indices = np.where(inliers01)[0]

    for idx_pt, (i_match, x0n, x1n) in enumerate(
        zip(inlier_indices, pts0_n, pts1_n)
    ):
        # 三角化
        X = triangulate_point(P0, P1, x0n, x1n)
        points_3d.append(X)

        # 颜色：从 image 0 取
        u0, v0 = pts0_in[idx_pt]
        u0i = int(round(u0))
        v0i = int(round(v0))
        h0, w0, _ = img0.shape
        if 0 <= v0i < h0 and 0 <= u0i < w0:
            bgr = img0[v0i, u0i].astype(np.int32)
            rgb = bgr[::-1]
        else:
            rgb = np.array([255, 255, 255], dtype=np.int32)
        colors_3d.append(rgb)

        pt_idx = len(points_3d) - 1

        # 观测：相机 0
        obs_cam_idx.append(0)
        obs_pt_idx.append(pt_idx)
        obs_uv.append(pts0_in[idx_pt])

        # 观测：相机 1
        obs_cam_idx.append(1)
        obs_pt_idx.append(pt_idx)
        obs_uv.append(pts1_in[idx_pt])

        # 建立 kp index -> point index 映射
        kp0_id = matches_01_all[i_match][0]
        kp1_id = matches_01_all[i_match][1]
        map0[kp0_id] = pt_idx
        map1[kp1_id] = pt_idx

    points_3d = np.array(points_3d, dtype=np.float64)
    colors_3d = np.array(colors_3d, dtype=np.int32)
    obs_cam_idx = np.array(obs_cam_idx, dtype=np.int32)
    obs_pt_idx = np.array(obs_pt_idx, dtype=np.int32)
    obs_uv = np.array(obs_uv, dtype=np.float64)

    print(f"Initial 3D points from 0-1: {len(points_3d)}")

    # ===== 8. 利用 0-2 / 1-2 匹配 + 已有 3D 点，PnP 估计相机 2 位姿 =====
    matches_02_in = matches_inliers_dict.get((0, 2), matches_dict.get((0, 2), []))
    matches_12_in = matches_inliers_dict.get((1, 2), matches_dict.get((1, 2), []))

    pts3d_pnp = []
    pts2_pnp = []

    # 用 0-2 匹配：已有 map0 的 2D 点可以提供 3D-2D 对
    for (i0, i2, _) in matches_02_in:
        if i0 in map0:
            pt_idx = map0[i0]
            pts3d_pnp.append(points_3d[pt_idx])
            pts2_pnp.append(keypoints[2][i2])

    # 用 1-2 匹配
    for (i1, i2, _) in matches_12_in:
        if i1 in map1:
            pt_idx = map1[i1]
            pts3d_pnp.append(points_3d[pt_idx])
            pts2_pnp.append(keypoints[2][i2])

    pts3d_pnp = np.array(pts3d_pnp, dtype=np.float64)
    pts2_pnp = np.array(pts2_pnp, dtype=np.float64)

    n_pnp = len(pts3d_pnp)
    print(f"PnP correspondences for camera 2: {n_pnp}")

    if n_pnp >= 4:
        # 正常路径：用非线性 PnP 拟合相机 2 位姿
        r2, t2 = pnp_nonlinear(pts3d_pnp, pts2_pnp, K)
        R2 = rodrigues(r2)
        print("Initial pose for camera 2 (relative to camera 0) from PnP:")
        print("R2 =\n", R2)
        print("t2 =\n", t2)
    else:
        # 兜底方案：3D-2D 对太少，不做 PnP，
        # 改用 (0,2) 这对图像的两视图几何来估计相机 2 位姿
        print("WARNING: very few 3D-2D correspondences, "
              "fall back to two-view pose estimation using pair (0,2).")

        matches_02 = matches_dict[(0, 2)]
        if len(matches_02) < 8:
            raise RuntimeError(
                "Not enough 3D-2D correspondences for PnP, "
                "and also not enough matches between image 0 and 2 "
                "to estimate F02."
            )

        # 用全部 0-2 匹配做 RANSAC 估计 F02
        pts0_02 = np.array([keypoints[0][m[0]] for m in matches_02],
                           dtype=np.float64)
        pts2_02 = np.array([keypoints[2][m[1]] for m in matches_02],
                           dtype=np.float64)

        F02, inliers02 = ransac_fundamental(
            pts0_02, pts2_02, num_iters=2000, thresh=1e-3
        )
        print("Fallback F02 (0-2):\n", F02)
        pts0_02_in = pts0_02[inliers02]
        pts2_02_in = pts2_02[inliers02]

        # 由 F02 -> E02 -> 分解得到 (R2, t2)
        E02 = K.T @ F02 @ K
        Ue, Se, Vte = np.linalg.svd(E02)
        Se[2] = 0.0
        E02 = Ue @ np.diag(Se) @ Vte

        R2, t2 = decompose_essential(E02, K, pts0_02_in, pts2_02_in)
        print("Pose for camera 2 from essential matrix (0-2):")
        print("R2 =\n", R2)
        print("t2 =\n", t2)


    # ===== 9. 再次三角化：利用第三张图像引入新的 3D 点 =====
    #   - 对 0-2 / 1-2 匹配进行遍历：
    #       若某个特征点已有对应 3D 点，则只增加新的观测；
    #       若没有，则在相应两台相机之间三角化生成新点。
    P0 = np.hstack([R0, t0.reshape(3, 1)])
    P1 = np.hstack([R1, t1.reshape(3, 1)])
    P2 = np.hstack([R2, t2.reshape(3, 1)])

    map2 = {}  # 第三张图的特征点 -> 3D 点

    # 9.1 利用 0-2 匹配
    for (i0, i2, _) in matches_02_in:
        kp0 = keypoints[0][i0]
        kp2 = keypoints[2][i2]

        if i0 in map0:
            # 该点已有 3D，增加相机 2 的观测
            pt_idx = map0[i0]
            obs_cam_idx = np.append(obs_cam_idx, 2)
            obs_pt_idx = np.append(obs_pt_idx, pt_idx)
            obs_uv = np.vstack([obs_uv, kp2])
            map2[i2] = pt_idx
        else:
            # 新 3D 点：在相机 0 & 2 之间三角化
            x0n = normalize_pixel_points(kp0[None, :], K)[0]
            x2n = normalize_pixel_points(kp2[None, :], K)[0]
            X = triangulate_point(P0, P2, x0n, x2n)
            points_3d = np.vstack([points_3d, X[None, :]])

            # 颜色从第 0 张图取
            u0, v0 = kp0
            u0i = int(round(u0))
            v0i = int(round(v0))
            h0, w0, _ = img0.shape
            if 0 <= v0i < h0 and 0 <= u0i < w0:
                bgr = img0[v0i, u0i].astype(np.int32)
                rgb = bgr[::-1]
            else:
                rgb = np.array([255, 255, 255], dtype=np.int32)
            colors_3d = np.vstack([colors_3d, rgb[None, :]])

            pt_idx = len(points_3d) - 1

            # 观测 0 和 2
            obs_cam_idx = np.append(obs_cam_idx, [0, 2])
            obs_pt_idx = np.append(obs_pt_idx, [pt_idx, pt_idx])
            obs_uv = np.vstack([obs_uv, kp0, kp2])

            map0[i0] = pt_idx
            map2[i2] = pt_idx

    # 9.2 利用 1-2 匹配
    for (i1, i2, _) in matches_12_in:
        kp1 = keypoints[1][i1]
        kp2 = keypoints[2][i2]

        if (i1 in map1) and (i2 in map2):
            # 该匹配两端都已有 3D，对应的点应该是同一个；给第 1 张图增加观测
            pt_idx = map1[i1]
            # 简单检查一致性
            if map2[i2] != pt_idx:
                # 若不一致，保守起见跳过
                continue
            obs_cam_idx = np.append(obs_cam_idx, 1)
            obs_pt_idx = np.append(obs_pt_idx, pt_idx)
            obs_uv = np.vstack([obs_uv, kp1])
        elif (i1 in map1) and (i2 not in map2):
            # 1 端已有 3D，2 端新观测
            pt_idx = map1[i1]
            obs_cam_idx = np.append(obs_cam_idx, 2)
            obs_pt_idx = np.append(obs_pt_idx, pt_idx)
            obs_uv = np.vstack([obs_uv, kp2])
            map2[i2] = pt_idx
        elif (i1 not in map1) and (i2 in map2):
            # 2 端已有 3D，1 端新观测
            pt_idx = map2[i2]
            obs_cam_idx = np.append(obs_cam_idx, 1)
            obs_pt_idx = np.append(obs_pt_idx, pt_idx)
            obs_uv = np.vstack([obs_uv, kp1])
            map1[i1] = pt_idx
        else:
            # 两端都没有 3D 点：在 1 & 2 之间三角化
            x1n = normalize_pixel_points(kp1[None, :], K)[0]
            x2n = normalize_pixel_points(kp2[None, :], K)[0]
            X = triangulate_point(P1, P2, x1n, x2n)
            points_3d = np.vstack([points_3d, X[None, :]])

            # 颜色从第 1 张图取
            img1 = images[1]
            u1, v1 = kp1
            u1i = int(round(u1))
            v1i = int(round(v1))
            h1, w1, _ = img1.shape
            if 0 <= v1i < h1 and 0 <= u1i < w1:
                bgr = img1[v1i, u1i].astype(np.int32)
                rgb = bgr[::-1]
            else:
                rgb = np.array([255, 255, 255], dtype=np.int32)
            colors_3d = np.vstack([colors_3d, rgb[None, :]])

            pt_idx = len(points_3d) - 1
            obs_cam_idx = np.append(obs_cam_idx, [1, 2])
            obs_pt_idx = np.append(obs_pt_idx, [pt_idx, pt_idx])
            obs_uv = np.vstack([obs_uv, kp1, kp2])

            map1[i1] = pt_idx
            map2[i2] = pt_idx

    print(f"Total 3D points (after using image 3): {len(points_3d)}")
    print(f"Total observations for BA: {len(obs_cam_idx)}")

    # ===== 10. Bundle Adjustment：联合优化三张图位姿和所有 3D 点 =====
    R_list_init = [R0, R1, R2]
    t_list_init = [t0, t1, t2]

    points_3d_opt, R_list_opt, t_list_opt = bundle_adjustment(
        points_3d,
        obs_cam_idx,
        obs_pt_idx,
        obs_uv,
        K,
        R_list_init,
        t_list_init,
    )

    R0_opt, R1_opt, R2_opt = R_list_opt
    t0_opt, t1_opt, t2_opt = t_list_opt

    # ===== 11. 根据重投影误差删除错误 3D 点 =====
    def compute_pointwise_reprojection_errors(points_3d, R_list, t_list, K,
                                              obs_cam_idx, obs_pt_idx, obs_uv):
        """按 3D 点统计平均重投影误差，返回 (point_errs, track_len)。"""
        points_3d = np.asarray(points_3d, dtype=np.float64)
        obs_cam_idx = np.asarray(obs_cam_idx, dtype=np.int32)
        obs_pt_idx = np.asarray(obs_pt_idx, dtype=np.int32)
        obs_uv = np.asarray(obs_uv, dtype=np.float64)

        n_pts = points_3d.shape[0]
        err_sum = np.zeros(n_pts, dtype=np.float64)
        cnt = np.zeros(n_pts, dtype=np.int32)

        for cam_i, pt_i, uv in zip(obs_cam_idx, obs_pt_idx, obs_uv):
            R = R_list[cam_i]
            t = t_list[cam_i]
            X = points_3d[pt_i]
            Xc = R @ X + t
            u_pred = K[0, 0] * Xc[0] / (Xc[2] + 1e-12) + K[0, 2]
            v_pred = K[1, 1] * Xc[1] / (Xc[2] + 1e-12) + K[1, 2]
            err = np.sqrt((u_pred - uv[0]) ** 2 + (v_pred - uv[1]) ** 2)
            err_sum[pt_i] += err
            cnt[pt_i] += 1

        mean_err = err_sum / (cnt + 1e-12)
        return mean_err, cnt

    point_errs, track_len = compute_pointwise_reprojection_errors(
        points_3d_opt,
        R_list_opt,
        t_list_opt,
        K,
        obs_cam_idx,
        obs_pt_idx,
        obs_uv,
    )

    # 简单阈值：重投影误差 > 5 像素视为异常点
    thresh_bad = 5.0
    keep_mask = point_errs <= thresh_bad
    n_before = len(points_3d_opt)
    n_after = int(keep_mask.sum())
    print(
        f"Filter 3D points by reprojection error: keep {n_after}/{n_before} "
        f"(threshold = {thresh_bad:.2f} px)"
    )

    # 过滤 3D 点和颜色
    points_3d_final = points_3d_opt[keep_mask]
    colors_3d_final = colors_3d[keep_mask]

    # 过滤观测并重建新的 pt_idx
    old_to_new = -np.ones(len(points_3d_opt), dtype=np.int32)
    keep_indices = np.where(keep_mask)[0]
    for new_idx, old_idx in enumerate(keep_indices):
        old_to_new[old_idx] = new_idx

    mask_obs_keep = old_to_new[obs_pt_idx] >= 0
    obs_cam_idx_final = obs_cam_idx[mask_obs_keep]
    obs_pt_idx_final = old_to_new[obs_pt_idx[mask_obs_keep]]
    obs_uv_final = obs_uv[mask_obs_keep]

    # 用过滤后的点重新计算每个相机的重投影误差
    reproj_errors = compute_reprojection_errors(
        points_3d_final,
        R_list_opt,
        t_list_opt,
        K,
        obs_cam_idx_final,
        obs_pt_idx_final,
        obs_uv_final,
    )
    print("Reprojection errors (pixels) for cameras 0,1,2:", reproj_errors)

    # 也给出所有观测的整体平均重投影误差
    all_err_sum = 0.0
    all_cnt = 0
    for cam_i, pt_i, uv in zip(
        obs_cam_idx_final, obs_pt_idx_final, obs_uv_final
    ):
        R = R_list_opt[cam_i]
        t = t_list_opt[cam_i]
        X = points_3d_final[pt_i]
        Xc = R @ X + t
        u_pred = K[0, 0] * Xc[0] / (Xc[2] + 1e-12) + K[0, 2]
        v_pred = K[1, 1] * Xc[1] / (Xc[2] + 1e-12) + K[1, 2]
        err = np.sqrt((u_pred - uv[0]) ** 2 + (v_pred - uv[1]) ** 2)
        all_err_sum += err
        all_cnt += 1
    global_reproj = all_err_sum / max(all_cnt, 1)
    print(f"Global mean reprojection error = {global_reproj:.3f} px")

    # ===== 12. 计算位姿误差（相对 GT） =====
    pose_err_values = []  # 保存 max(rot,trans)，用于写 txt
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        Ri_gt, ti_gt = R_gt[i], t_gt[i]
        Rj_gt, tj_gt = R_gt[j], t_gt[j]

        Ri_est, ti_est = R_list_opt[i], t_list_opt[i]
        Rj_est, tj_est = R_list_opt[j], t_list_opt[j]

        rot_err, trans_err = pose_error_pair(
            Ri_gt,
            ti_gt,
            Rj_gt,
            tj_gt,
            Ri_est,
            ti_est,
            Rj_est,
            tj_est,
        )
        pose_err = max(rot_err, trans_err)
        pose_err_values.append(((i, j), pose_err))

        print(
            f"Pose error between camera {i}-{j}: "
            f"rot = {rot_err:.3f} deg, trans = {trans_err:.3f} deg, "
            f"max = {pose_err:.3f} deg",
        )

    # ===== 13. 输出重建结果到 txt（格式尽量与“结果示例”一致） =====
    # 13.1 点云：每行 [id x y z r g b track_len]
    pc_path = RECONS_DIR / "点云.txt"
    with open(pc_path, "w", encoding="utf-8") as f:
        for idx, (X, rgb, trk) in enumerate(
            zip(points_3d_final, colors_3d_final, track_len[keep_mask])
        ):
            r, g, b = rgb
            # 示例里多写了一个“track length”列，我们也仿照写上
            f.write(
                f"{idx} {X[0]} {X[1]} {X[2]} "
                f"{int(r)} {int(g)} {int(b)} {int(trk)}\n"
            )
    print(f"Saved point cloud to: {pc_path}")

    # 13.2 相机位姿（4x4 外参），每个相机一个 4x4 矩阵
    cam_path = RECONS_DIR / "图像位姿.txt"
    with open(cam_path, "w", encoding="utf-8") as f:
        for cam_id, (R, t) in enumerate(zip(R_list_opt, t_list_opt)):
            # 示例里图像名是 1.jpg/2.jpg/3.jpg，我们就用 cam_id+1
            img_name = f"{cam_id+1}.jpg"
            f.write(f"{img_name}:\n")
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            for r in range(4):
                row = " ".join(str(x) for x in T[r, :])
                f.write(row + "\n")
            f.write("\n")
    print(f"Saved camera extrinsics to: {cam_path}")

    # 13.3 误差统计：位姿误差 + 重投影误差 + 点云数量
    err_path = RECONS_DIR / "误差.txt"
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("位姿误差：\n")
        for (i, j), e in pose_err_values:
            f.write(f"{i}-{j}:\n")
            f.write(f"{e:.3f}\n")
        f.write("\n重投影误差：\n")
        f.write(f"{global_reproj:.3f}\n\n")
        f.write("点云数量：\n")
        f.write(str(len(points_3d_final)) + "\n")
    print(f"Saved error report to: {err_path}")


if __name__ == "__main__":
    main()
