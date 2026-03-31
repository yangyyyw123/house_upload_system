import cv2
import numpy as np
from skimage.morphology import skeletonize
from skan import csr
from sklearn.decomposition import PCA

# === 基础函数 ===
def compute_skeleton(mask):
    return skeletonize(mask.astype(bool), method='lee').astype(np.uint8)

def get_pca_normal_directions(coords, window_size=7):
    normal_map = {}
    half = window_size // 2

    for i in range(len(coords)):
        window_coords = coords[max(0, i - half):min(len(coords), i + half + 1)]
        if len(window_coords) < 2:
            continue
        pca = PCA(n_components=2)
        pca.fit(window_coords[:, [1, 0]])
        direction = pca.components_[0]
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal)
        normal_map[tuple(coords[i])] = normal
    return normal_map

def get_crack_angle(coords):
    # coords 是形如 [(y1, x1), (y2, x2), ...]
    coords = np.array(coords)
    if coords.shape[0] < 2:
        return None
    pca = PCA(n_components=2)
    pca.fit(coords[:, ::-1])  # 注意 x 在前 (x, y)
    direction = pca.components_[0]  # 主方向向量
    angle_rad = np.arctan2(direction[1], direction[0])
    angle_deg = np.rad2deg(angle_rad)
    return angle_deg    #角度单位为 度（°），相对于水平方向，范围为 [-180°, 180°]


def measure_width(coord, normal, mask, max_dist=20):
    h, w = mask.shape
    dist = 0
    for sign in [-1, 1]:
        for i in range(1, max_dist):
            px = int(round(coord[1] + sign * normal[0] * i))
            py = int(round(coord[0] + sign * normal[1] * i))
            if 0 <= px < w and 0 <= py < h:
                if mask[py, px] == 0:
                    dist += i
                    break
            else:
                break
    return dist

def compute_length(coords):
    if len(coords) < 2:
        return 0.0

    length = 0.0
    for i in range(1, len(coords)):
        p1 = coords[i-1]
        p2 = coords[i]
        length += np.linalg.norm(p2 - p1)

    return length

def draw_segment_overlay(image, coords, normal_map, mask, max_point=None, max_width=None, angle=None):
    overlay = image.copy()
    for i, coord in enumerate(coords):
        y, x = coord
        if (y, x) not in normal_map:
            continue
        dx, dy = normal_map[(y, x)]
        endpoints = []
        for sign in [-1, 1]:
            for j in range(1, 20):
                px = int(round(x + sign * dx * j))
                py = int(round(y + sign * dy * j))
                if 0 <= px < mask.shape[1] and 0 <= py < mask.shape[0]:
                    if mask[py, px] == 0:
                        endpoints.append((px, py))
                        break
                else:
                    break
        if len(endpoints) == 2:
            cv2.line(overlay, endpoints[0], endpoints[1], (255, 255, 0), 1)
        cv2.circle(overlay, (x, y), 1, (255, 0, 0), -1)

    if max_point and angle is not None:
        cv2.circle(overlay, (max_point[1], max_point[0]), 5, (0, 0, 255), -1)
        text = f"Max_width:{max_width}, angle:{angle:.1f}"
        cv2.putText(overlay, text, (max_point[1]+5, max_point[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return overlay