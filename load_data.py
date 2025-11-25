import numpy as np
import cv2
from pathlib import Path

# ===================== 工具函数：相机参数读写 =====================

def load_camera_file(cam_path):
    """读取 *_cam.txt 文件，返回 (4x4 extrinsic, 3x3 intrinsic)"""
    with open(cam_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # 简单解析：遇到 'extrinsic' 后读 4 行，遇到 'intrinsic' 后读 3 行
    ext = None
    K = None
    i = 0
    while i < len(lines):
        if lines[i].lower().startswith("extrinsic"):
            mat = []
            for j in range(1, 5):
                mat.append([float(x) for x in lines[i + j].split()])
            ext = np.array(mat, dtype=np.float64)
            i += 5
        elif lines[i].lower().startswith("intrinsic"):
            mat = []
            for j in range(1, 4):
                mat.append([float(x) for x in lines[i + j].split()])
            K = np.array(mat, dtype=np.float64)
            i += 4
        else:
            i += 1

    if ext is None or K is None:
        raise ValueError(f"camera file parse error: {cam_path}")
    return ext, K


def load_dataset(data_dir):
    """
    读 3 张图片和对应相机文件。
    默认名字为 00000022/23/24.jpg 和 *_cam.txt。
    """
    img_ids = ["00000022", "00000023", "00000024"]
    images = []
    cams_ext = []
    Ks = []

    for img_id in img_ids:
        img_path = Path(data_dir) / f"{img_id}.jpg"
        cam_path = Path(data_dir) / f"{img_id}_cam.txt"

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        images.append(img)

        ext, K = load_camera_file(cam_path)
        cams_ext.append(ext)
        Ks.append(K)

    # 三张图的内参几乎相同，这里就取第一张作为 K
    K = Ks[0]
    return images, cams_ext, K