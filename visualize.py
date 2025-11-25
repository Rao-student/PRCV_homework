import numpy as np
import cv2
from pathlib import Path
from scipy import optimize

def visualize_matches(img1, img2,
                      kps1, kps2,
                      matches,
                      max_draw=200,
                      save_path=None):
    """
    可视化两幅图像之间的特征匹配。
    img1, img2: BGR 图像
    kps1, kps2: (N,2) 像素坐标 (x,y)
    matches: [(idx1, idx2, dist), ...]
    max_draw: 最多画多少条匹配线
    save_path: 保存路径（None 则只返回图像）
    """
    if len(matches) == 0:
        print("No matches to visualize.")
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    W = w1 + w2

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    # 取前 max_draw 条（matches 已经按距离排过序）
    draw_matches = matches[:min(max_draw, len(matches))]

    rng = np.random.default_rng(0)

    for (i1, i2, _) in draw_matches:
        x1, y1 = kps1[i1]
        x2, y2 = kps2[i2]

        color = rng.integers(0, 255, size=3).tolist()

        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)) + w1, int(round(y2)))  # 第二张图要加上偏移 w1

        cv2.circle(canvas, pt1, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt2, 3, color, -1, lineType=cv2.LINE_AA)
        cv2.line(canvas, pt1, pt2, color, 1, lineType=cv2.LINE_AA)

    if save_path is not None:
        cv2.imwrite(str(save_path), canvas)
        print(f"Saved match visualization to: {save_path}")

    return canvas

def _line_segment_in_image(a, b, c, w, h):
    """
    给定直线 ax+by+c=0 和图像尺寸 w,h，求在图像内的线段端点（最多返回两个点）。
    若不足两个有效点，返回 None。
    """
    pts = []

    # 与 x = 0, x = w-1 相交
    xs = [0, w - 1]
    for x in xs:
        if abs(b) > 1e-6:
            y = -(a * x + c) / b
            if 0 <= y < h:
                pts.append((int(round(x)), int(round(y))))

    # 与 y = 0, y = h-1 相交
    ys = [0, h - 1]
    for y in ys:
        if abs(a) > 1e-6:
            x = -(b * y + c) / a
            if 0 <= x < w:
                pts.append((int(round(x)), int(round(y))))

    # 去重并取前两个
    if len(pts) < 2:
        return None
    p1 = pts[0]
    p2 = None
    for p in pts[1:]:
        if p != p1:
            p2 = p
            break
    if p2 is None:
        return None
    return p1, p2


def visualize_epipolar_lines(img1, img2,
                             pts1, pts2,
                             F,
                             num_lines=20,
                             save_path=None):
    """
    可视化极线：
    - 左边图：显示 pts1 中的一些点和它们在左图中的极线（由 right->left 的 F^T 得到）
    - 右边图：显示 pts2 中对应点和右图中的极线（由 left->right 的 F 得到）
    pts1, pts2: (N,2)，对应 F 的 x2^T F x1 = 0
    F: 3x3 基础矩阵
    """
    N = pts1.shape[0]
    if N == 0:
        print("No points to draw epipolar lines.")
        return None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    H = max(h1, h2)
    W = w1 + w2

    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    img1_vis = img1.copy()
    img2_vis = img2.copy()
    canvas[:h1, :w1] = img1_vis
    canvas[:h2, w1:w1 + w2] = img2_vis

    # 选一些点画线（均匀 subsample）
    step = max(1, N // num_lines)
    idxs = list(range(0, N, step))[:num_lines]

    rng = np.random.default_rng(1)

    for i in idxs:
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        color = rng.integers(0, 255, size=3).tolist()

        # -------- 右图中的极线（由左图点 x1 通过 F）--------
        x1_h = np.array([x1, y1, 1.0], dtype=np.float64)
        l2 = F @ x1_h  # (a,b,c)
        a2, b2, c2 = l2
        seg2 = _line_segment_in_image(a2, b2, c2, w2, h2)
        if seg2 is not None:
            (x2a, y2a), (x2b, y2b) = seg2
            pt2a = (x2a + w1, y2a)
            pt2b = (x2b + w1, y2b)
            cv2.line(canvas, pt2a, pt2b, color, 1, lineType=cv2.LINE_AA)

        # 在右图画对应匹配点
        pt2 = (int(round(x2)) + w1, int(round(y2)))
        cv2.circle(canvas, pt2, 4, color, -1, lineType=cv2.LINE_AA)

        # -------- 左图中的极线（由右图点 x2 通过 F^T）--------
        x2_h = np.array([x2, y2, 1.0], dtype=np.float64)
        l1 = F.T @ x2_h
        a1, b1, c1 = l1
        seg1 = _line_segment_in_image(a1, b1, c1, w1, h1)
        if seg1 is not None:
            (x1a, y1a), (x1b, y1b) = seg1
            pt1a = (x1a, y1a)
            pt1b = (x1b, y1b)
            cv2.line(canvas, pt1a, pt1b, color, 1, lineType=cv2.LINE_AA)

        # 在左图画原始点
        pt1 = (int(round(x1)), int(round(y1)))
        cv2.circle(canvas, pt1, 4, color, -1, lineType=cv2.LINE_AA)

    if save_path is not None:
        cv2.imwrite(str(save_path), canvas)
        print(f"Saved epipolar visualization to: {save_path}")

    return canvas
