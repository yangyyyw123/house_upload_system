import numpy as np

def stitch_patches(patch_list, coords):
    """
    将多个 patch 和其位置坐标拼接为整张图。
    patch_list: list of [H, W] uint8 masks (0 or 1)
    coords: list of (row_idx, col_idx), 从 (1, 1) 开始编号
    """
    if not patch_list:
        return None

    h, w = patch_list[0].shape

    # 适配从 (1,1) 开始编号
    max_row = max(r for r, _ in coords)
    max_col = max(c for _, c in coords)

    stitched = np.zeros((max_row * h, max_col * w), dtype=np.uint8)

    for patch, (r, c) in zip(patch_list, coords):
        stitched[(r - 1) * h : r * h, (c - 1) * w : c * w] = patch

    return stitched