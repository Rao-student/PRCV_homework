import numpy as np
import cv2
from pathlib import Path
from scipy import optimize
# ===================== 特征：Harris + patch 描述子 =====================
def harris_corners(gray,
                   window_size=5,
                   k=0.04,
                   thresh_rel=0.01,
                   nms_radius=4,
                   max_points=2000):
    """
    自己实现 Harris 角点检测：
    gray: (H,W) float32 / uint8
    返回：keypoints (N,2) (x,y)，responses (N,)
    """
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    gray_norm = gray / 255.0

    # 梯度
    Ix = cv2.Sobel(gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_norm, cv2.CV_32F, 0, 1, ksize=3)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    Sxx = cv2.GaussianBlur(Ixx, (window_size, window_size), 1)
    Syy = cv2.GaussianBlur(Iyy, (window_size, window_size), 1)
    Sxy = cv2.GaussianBlur(Ixy, (window_size, window_size), 1)

    detM = Sxx * Syy - Sxy * Sxy
    traceM = Sxx + Syy
    R = detM - k * (traceM ** 2)

    R_max = float(R.max())
    if R_max <= 0:
        return np.zeros((0, 2)), np.zeros((0,))

    H, W = R.shape
    ys, xs = np.where(R > thresh_rel * R_max)

    keypoints = []
    responses = []

    for y, x in zip(ys, xs):
        if (x < nms_radius or x >= W - nms_radius or
                y < nms_radius or y >= H - nms_radius):
            continue
        patch = R[y - nms_radius:y + nms_radius + 1,
                  x - nms_radius:x + nms_radius + 1]
        if R[y, x] >= patch.max():
            keypoints.append((x, y))
            responses.append(R[y, x])

    keypoints = np.array(keypoints, dtype=np.float32)
    responses = np.array(responses, dtype=np.float32)

    if len(responses) > max_points:
        idx = np.argsort(-responses)[:max_points]
        keypoints = keypoints[idx]
        responses = responses[idx]

    return keypoints, responses


def build_patch_descriptors(gray, keypoints, patch_size=11):
    """
    为每个角点提取灰度 patch 描述子。
    返回：valid_keypoints (M,2), descriptors (M,D)
    """
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    gray_norm = gray / 255.0
    H, W = gray_norm.shape
    r = patch_size // 2

    valid_kps = []
    descs = []

    for (x, y) in keypoints:
        x = int(round(x))
        y = int(round(y))
        if x < r or x >= W - r or y < r or y >= H - r:
            continue
        patch = gray_norm[y - r:y + r + 1, x - r:x + r + 1]
        vec = patch.reshape(-1).astype(np.float32)
        vec -= vec.mean()
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec /= norm
        valid_kps.append((x, y))
        descs.append(vec)

    if len(descs) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, patch_size * patch_size), dtype=np.float32)

    return np.array(valid_kps, dtype=np.float32), np.array(descs, dtype=np.float32)


def match_descriptors(desc1, desc2, ratio=0.75):
    """
    暴力匹配 + Lowe 比率测试。
    desc1: (N1,D), desc2: (N2,D)
    返回：matches = [(i1, i2, dist), ...]
    """
    matches = []
    if len(desc1) == 0 or len(desc2) == 0:
        return matches

    for i, d in enumerate(desc1):
        diff = desc2 - d[None, :]
        dists = np.sum(diff * diff, axis=1)
        if len(dists) < 2:
            continue
        idx = np.argsort(dists)
        best, second = float(dists[idx[0]]), float(dists[idx[1]])
        if second <= 1e-12:
            continue
        if best / second < ratio:
            matches.append((i, int(idx[0]), best))

    matches.sort(key=lambda x: x[2])
    return matches


# ===================== 几何：F, E, 分解，三角化 =====================

def _hartley_normalization(pts):
    """Hartley 归一化：让点的均值在原点，平均距离为 sqrt(2)。"""
    pts = np.asarray(pts, dtype=np.float64)
    mean = pts.mean(axis=0)
    pts_centered = pts - mean
    dist = np.sqrt(np.sum(pts_centered ** 2, axis=1))
    scale = np.sqrt(2) / (dist.mean() + 1e-12)

    T = np.array([
        [scale, 0, -scale * mean[0]],
        [0, scale, -scale * mean[1]],
        [0, 0, 1],
    ], dtype=np.float64)

    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))]).T
    pts_norm_h = T @ pts_h
    pts_norm = (pts_norm_h[:2, :] / pts_norm_h[2, :]).T
    return pts_norm, T


def eight_point_fundamental_normalized(pts1, pts2):
    """
    采用 Hartley 归一化的 8 点法估计 F，提升数值稳定性。
    返回：去归一化后的 F (3x3)，满足 x2.T F x1 ≈ 0
    """
    N = pts1.shape[0]
    if N < 8:
        raise ValueError("Need at least 8 points for 8-point algorithm")

    pts1_norm, T1 = _hartley_normalization(pts1)
    pts2_norm, T2 = _hartley_normalization(pts2)

    x1 = pts1_norm[:, 0]
    y1 = pts1_norm[:, 1]
    x2 = pts2_norm[:, 0]
    y2 = pts2_norm[:, 1]

    A = np.zeros((N, 9), dtype=np.float64)
    A[:, 0] = x2 * x1
    A[:, 1] = x2 * y1
    A[:, 2] = x2
    A[:, 3] = y2 * x1
    A[:, 4] = y2 * y1
    A[:, 5] = y2
    A[:, 6] = x1
    A[:, 7] = y1
    A[:, 8] = 1.0

    # SVD(A) -> 最小奇异值对应的右奇异向量
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    # 施加 rank-2 约束
    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0.0
    F_norm = U @ np.diag(S) @ Vt
    # 去归一化：F = T2^T * F_norm * T1
    F = T2.T @ F_norm @ T1
    return F


def sampson_error(F, pts1, pts2):
    """
    计算 Sampson distance，用于 RANSAC 内点判定
    pts1, pts2: (N,2)
    """
    N = pts1.shape[0]
    pts1_h = np.hstack([pts1, np.ones((N, 1))])
    pts2_h = np.hstack([pts2, np.ones((N, 1))])

    Fx1 = F @ pts1_h.T            # (3,N)
    Ftx2 = F.T @ pts2_h.T         # (3,N)

    x2tFx1 = np.sum(pts2_h * (F @ pts1_h.T).T, axis=1)  # (N,)

    num = x2tFx1 ** 2
    denom = Fx1[0, :] ** 2 + Fx1[1, :] ** 2 + Ftx2[0, :] ** 2 + Ftx2[1, :] ** 2 + 1e-12
    return num / denom


def ransac_fundamental(pts1, pts2, num_iters=4000, thresh=0.5):
    """
    RANSAC 估计 F，内部使用 Hartley 归一化的 8 点法并在像素坐标中计算
    Sampson 误差。默认阈值 0.5 像素，能更好地兼顾召回与精度。
    返回：F_best, inlier_mask (N,)
    """
    N = pts1.shape[0]
    if N < 8:
        raise ValueError("Need at least 8 correspondences for RANSAC F")

    best_F = None
    best_inliers = None
    best_count = -1

    rng = np.random.default_rng(42)

    for _ in range(num_iters):
        sample_ids = rng.choice(N, size=8, replace=False)
        F_candidate = eight_point_fundamental_normalized(
            pts1[sample_ids], pts2[sample_ids]
        )

        errs = sampson_error(F_candidate, pts1, pts2)
        inlier_mask = errs < thresh
        count = int(inlier_mask.sum())

        if count > best_count:
            best_count = count
            best_F = F_candidate
            best_inliers = inlier_mask

    # 用内点重新估计一次 F
    F_refined = eight_point_fundamental_normalized(pts1[best_inliers], pts2[best_inliers])
    return F_refined, best_inliers


def normalize_pixel_points(pts, K):
    """
    像素坐标 -> 相机归一化平面坐标（x/z, y/z）
    pts: (N,2)
    """
    N = pts.shape[0]
    pts_h = np.hstack([pts, np.ones((N, 1))]).T  # (3,N)
    K_inv = np.linalg.inv(K)
    norm = K_inv @ pts_h
    norm /= norm[2, :]
    return norm[:2, :].T  # (N,2)


def triangulate_point(P0, P1, x0, x1):
    """
    线性三角化单个点。
    P0, P1: (3,4) 投影矩阵，对应归一化坐标系或像素坐标系
    x0, x1: (2,) 对应的点坐标（u,v 或 x_n,y_n）
    返回：X (3,) 世界坐标（齐次坐标归一化后的前三维）
    """
    u0, v0 = x0
    u1, v1 = x1

    A = np.zeros((4, 4), dtype=np.float64)
    A[0, :] = u0 * P0[2, :] - P0[0, :]
    A[1, :] = v0 * P0[2, :] - P0[1, :]
    A[2, :] = u1 * P1[2, :] - P1[0, :]
    A[3, :] = v1 * P1[2, :] - P1[1, :]

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    X_h /= X_h[3] + 1e-12
    return X_h[:3]


def decompose_essential(E, K, pts1_pix, pts2_pix):
    """
    从本质矩阵 E 分解出 (R,t)，并用三角化 + 正深度判定选择正确的一组。
    pts1_pix, pts2_pix 只用于做少量三角化检查。
    """
    U, S, Vt = np.linalg.svd(E)
    # 确保是正确的旋转矩阵
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[2, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]], dtype=np.float64)

    R_candidates = [U @ W @ Vt,
                    U @ W.T @ Vt]
    t_candidates = [U[:, 2], -U[:, 2]]

    # 归一化坐标
    pts1_n = normalize_pixel_points(pts1_pix, K)
    pts2_n = normalize_pixel_points(pts2_pix, K)

    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])

    best_count = -1
    best_R, best_t = None, None

    # 用少量点就够了
    n_check = min(50, pts1_n.shape[0])

    for R in R_candidates:
        if np.linalg.det(R) < 0:
            R = -R
        for t in t_candidates:
            P1 = np.hstack([R, t.reshape(3, 1)])
            front_count = 0
            for i in range(n_check):
                X = triangulate_point(P0, P1, pts1_n[i], pts2_n[i])
                # 在相机 0 中的深度
                z0 = X[2]
                X1 = R @ X + t
                z1 = X1[2]
                if z0 > 0 and z1 > 0:
                    front_count += 1
            if front_count > best_count:
                best_count = front_count
                best_R, best_t = R, t

    return best_R, best_t


# ===================== PnP & BA：旋转向量工具 =====================

def rodrigues(rvec):
    """axis-angle -> 旋转矩阵"""
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rvec / theta
    kx, ky, kz = k
    K = np.array([[0, -kz, ky],
                  [kz, 0, -kx],
                  [-ky, kx, 0]], dtype=np.float64)
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rotmat_to_rodrigues(R):
    """旋转矩阵 -> axis-angle 向量"""
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=np.float64) / (2 * np.sin(theta))
    return axis * theta


def project_points(X, rvec, tvec, K):
    """
    将 3D 点投影到像素平面。
    X: (N,3) 世界坐标
    rvec: (3,), tvec: (3,)
    K: (3,3)
    返回：uv_pred (N,2)
    """
    R = rodrigues(rvec)
    Xc = (R @ X.T) + tvec.reshape(3, 1)  # (3,N)
    x = Xc[0, :] / (Xc[2, :] + 1e-12)
    y = Xc[1, :] / (Xc[2, :] + 1e-12)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)


def pnp_nonlinear(X_world, x_pix, K):
    """
    简单的非线性 PnP：优化 rvec, tvec，使得重投影误差最小。
    X_world: (N,3), x_pix: (N,2)
    """
    X_world = np.asarray(X_world, dtype=np.float64)
    x_pix = np.asarray(x_pix, dtype=np.float64)
    assert len(X_world) == len(x_pix)

    # 初始猜测：优先使用 EPnP 结果，再做非线性细化
    x0 = np.zeros(6, dtype=np.float64)
    success, rvec_pnp, tvec_pnp = cv2.solvePnP(
        X_world,
        x_pix,
        K,
        None,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if success:
        x0[:3] = rvec_pnp.reshape(-1)
        x0[3:] = tvec_pnp.reshape(-1)
    else:
        x0[5] = 1.0

    def residuals(params):
        rvec = params[:3]
        tvec = params[3:]
        uv_pred = project_points(X_world, rvec, tvec, K)
        res = (uv_pred - x_pix).reshape(-1)
        return res

    res = optimize.least_squares(residuals, x0, method="lm", max_nfev=200)
    r_opt = res.x[:3]
    t_opt = res.x[3:]
    return r_opt, t_opt


def bundle_adjustment(points_3d,
                      obs_cam_idx,
                      obs_pt_idx,
                      obs_uv,
                      K,
                      R_list,
                      t_list):
    """
    简单 BA：优化相机 1、2 的 rvec,tvec，以及所有 3D 点。
    相机 0 作为参考系固定为 [I|0]。
    points_3d: (N,3)
    obs_cam_idx: (M,) 观测对应的相机索引 0/1/2
    obs_pt_idx:  (M,) 对应的点索引
    obs_uv:      (M,2) 像素坐标
    R_list, t_list: 初始的三个相机外参（R0,R1,R2），(t0,t1,t2)
    """

    points_3d = np.asarray(points_3d, dtype=np.float64)
    obs_cam_idx = np.asarray(obs_cam_idx, dtype=np.int32)
    obs_pt_idx = np.asarray(obs_pt_idx, dtype=np.int32)
    obs_uv = np.asarray(obs_uv, dtype=np.float64)

    n_points = points_3d.shape[0]

    # 相机 0 固定，只有相机 1、2 的 rvec,tvec 作为变量
    r0 = rotmat_to_rodrigues(R_list[0])
    t0 = t_list[0]
    r1 = rotmat_to_rodrigues(R_list[1])
    t1 = t_list[1]
    r2 = rotmat_to_rodrigues(R_list[2])
    t2 = t_list[2]

    # 参数向量：[r1(3), t1(3), r2(3), t2(3), X0(3), X1(3), ...]
    x0 = np.concatenate([
        r1, t1,
        r2, t2,
        points_3d.reshape(-1)
    ])

    def ba_residuals(params):
        r1 = params[0:3]
        t1 = params[3:6]
        r2 = params[6:9]
        t2 = params[9:12]
        pts = params[12:].reshape((n_points, 3))

        residuals = []

        for cam_i, pt_i, uv in zip(obs_cam_idx, obs_pt_idx, obs_uv):
            if cam_i == 0:
                R = rodrigues(r0)
                t = t0
            elif cam_i == 1:
                R = rodrigues(r1)
                t = t1
            else:
                R = rodrigues(r2)
                t = t2

            X = pts[pt_i]
            Xc = R @ X + t
            u_pred = K[0, 0] * Xc[0] / (Xc[2] + 1e-12) + K[0, 2]
            v_pred = K[1, 1] * Xc[1] / (Xc[2] + 1e-12) + K[1, 2]
            residuals.append(u_pred - uv[0])
            residuals.append(v_pred - uv[1])

        return np.array(residuals, dtype=np.float64)

    print("Running bundle adjustment ...")
    res = optimize.least_squares(
        ba_residuals, x0, method="lm",
        max_nfev=800, verbose=2
    )

    params_opt = res.x
    r1_opt = params_opt[0:3]
    t1_opt = params_opt[3:6]
    r2_opt = params_opt[6:9]
    t2_opt = params_opt[9:12]
    pts_opt = params_opt[12:].reshape((n_points, 3))

    R0_opt, t0_opt = R_list[0], t_list[0]
    R1_opt, t1_opt = rodrigues(r1_opt), t1_opt
    R2_opt, t2_opt = rodrigues(r2_opt), t2_opt

    R_list_opt = [R0_opt, R1_opt, R2_opt]
    t_list_opt = [t0_opt, t1_opt, t2_opt]

    return pts_opt, R_list_opt, t_list_opt


# ===================== 误差计算 =====================

def relative_pose(Ri, ti, Rj, tj):
    """
    世界->相机外参为 [R|t] 时，相机 i -> 相机 j 的相对位姿：
    x_j = R_ij x_i + t_ij
    """
    R_ij = Rj @ Ri.T
    t_ij = tj - R_ij @ ti
    return R_ij, t_ij


def rotation_angle(R):
    """从旋转矩阵计算旋转角（弧度）"""
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return theta


def pose_error_pair(Ri_gt, ti_gt, Rj_gt, tj_gt,
                    Ri_est, ti_est, Rj_est, tj_est):
    """
    计算一对相机 (i,j) 的位姿误差：
    - 旋转误差：相对旋转矩阵的夹角
    - 平移误差：平移方向向量之间的夹角
    返回 (rot_err_deg, trans_err_deg)
    """
    R_ij_gt, t_ij_gt = relative_pose(Ri_gt, ti_gt, Rj_gt, tj_gt)
    R_ij_est, t_ij_est = relative_pose(Ri_est, ti_est, Rj_est, tj_est)

    dR = R_ij_gt.T @ R_ij_est
    rot_err = rotation_angle(dR) * 180.0 / np.pi

    # 平移用方向
    tg = t_ij_gt / (np.linalg.norm(t_ij_gt) + 1e-12)
    te = t_ij_est / (np.linalg.norm(t_ij_est) + 1e-12)
    cos_t = np.clip(np.dot(tg, te), -1.0, 1.0)
    trans_err = np.arccos(cos_t) * 180.0 / np.pi

    return rot_err, trans_err


def compute_reprojection_errors(points_3d,
                                R_list,
                                t_list,
                                K,
                                obs_cam_idx,
                                obs_pt_idx,
                                obs_uv):
    """
    按相机统计平均重投影误差。
    返回：errors (3,) 三个相机的平均误差（像素）
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    obs_cam_idx = np.asarray(obs_cam_idx, dtype=np.int32)
    obs_pt_idx = np.asarray(obs_pt_idx, dtype=np.int32)
    obs_uv = np.asarray(obs_uv, dtype=np.float64)

    n_cam = len(R_list)
    sum_err = np.zeros(n_cam, dtype=np.float64)
    cnt = np.zeros(n_cam, dtype=np.int32)

    for cam_i, pt_i, uv in zip(obs_cam_idx, obs_pt_idx, obs_uv):
        R = R_list[cam_i]
        t = t_list[cam_i]
        X = points_3d[pt_i]

        Xc = R @ X + t
        u_pred = K[0, 0] * Xc[0] / (Xc[2] + 1e-12) + K[0, 2]
        v_pred = K[1, 1] * Xc[1] / (Xc[2] + 1e-12) + K[1, 2]
        err = np.sqrt((u_pred - uv[0]) ** 2 + (v_pred - uv[1]) ** 2)

        sum_err[cam_i] += err
        cnt[cam_i] += 1

    errors = sum_err / (cnt + 1e-12)
    return errors
