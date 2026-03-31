import cv2
import numpy as np
import os

def find_target(image_path, factor, aspect_ratio_tolerance, area_talerance):
    image = cv2.imread(image_path)
    
    # 灰度 & 二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    """第一步：筛选所有符合矩形条件的候选项"""
    candidate_rects = []
    rect_info_list = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_talerance:
            continue

        # 多边形拟合
        epsilon = factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 筛选矩形：4个顶点 + 几乎是凸的 + 面积够大
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 进一步检查角度接近 90 度
            angles = []
            for i in range(4):
                pt1 = approx[i][0]
                pt2 = approx[(i + 1) % 4][0]
                pt3 = approx[(i + 2) % 4][0]
                vec1 = pt1 - pt2
                vec2 = pt3 - pt2
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle_degree = np.degrees(np.arccos(cos_angle))
                angles.append(angle_degree)

            # 检查角度是否接近 90°
            angle_deviation = max(abs(angle_degree - 90) for angle_degree in angles)
            if angle_deviation < 10:
                rect = cv2.minAreaRect(approx)
                cx, cy = rect[0]
                w, h = rect[1]
                angle = rect[2]
                width, height = (w, h) if w >= h else (h, w)
                aspect_ratio = width / height

                size = None  # 或 0
                if 80 <= angle <= 90 or 0 <= angle <= 5:   # 水平角度，物理尺寸有5，2，1
                    if abs(aspect_ratio - 1.3334) < aspect_ratio_tolerance:
                        size = 5
                    elif abs(aspect_ratio - 6.23505) < aspect_ratio_tolerance:
                        size = 2
                    elif abs(aspect_ratio - 21.7559) < aspect_ratio_tolerance:
                        size = 1
                else:   # 倾斜角度，物理尺寸有6,8
                    if abs(aspect_ratio - 1.875) < aspect_ratio_tolerance:
                        size = 8   
                    elif abs(aspect_ratio - 5.833) < aspect_ratio_tolerance:
                        size = 6
                                    
                candidate_rects.append({
                    'contour': approx,
                    'center': (cx, cy),
                    'width': width,
                    'height': height,
                    'angle': angle,
                    'size': size
                })

    """第二步：检测相近的中心坐标对（相近的两个都丢弃）"""
    distance_thresh = 10
    to_remove = set()

    for i in range(len(candidate_rects)):
        for j in range(i + 1, len(candidate_rects)):
            cx1, cy1 = candidate_rects[i]['center']
            cx2, cy2 = candidate_rects[j]['center']
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            if dist < distance_thresh:
                to_remove.add(i)
                to_remove.add(j)

    """第三步：只保留未被标记删除的矩形，进行绘制和编号"""
    idx = 1
    for i, rect_info in enumerate(candidate_rects):
        if i in to_remove:
            continue

        box = rect_info['contour'].reshape(4, 2)
        cx, cy = rect_info['center']
        width = rect_info['width']
        height = rect_info['height']
        angle = rect_info['angle']
        size = rect_info['size']

        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        text_pos = (int(cx - 10), int(cy + 10))
        cv2.putText(image, str(idx), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        rect_info_list.append({
            'id': idx,
            'center': (cx, cy),
            'width': width,
            'height': height,
            'angle': angle,
            'size': size
                })

        idx += 1

    return image, rect_info_list


