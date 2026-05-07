def calculate_physical_ratio(rect_info_list):
    # 支持的标定尺寸
    valid_sizes = [1, 2, 5, 6, 8]

    # 初始化计数器和总比例字典
    count_dict = {size: 0 for size in valid_sizes}
    ratio_sum_dict = {size: 0.0 for size in valid_sizes}

    for info in rect_info_list:
        size = info['size']
        h = info['height']
        if size in valid_sizes and h > 0:
            count_dict[size] += 1
            ratio_sum_dict[size] += size / h

    # 计算每类的平均比例
    ratios = [
        ratio_sum_dict[size] / count_dict[size]
        for size in valid_sizes
        if count_dict[size] > 0
    ]

    # 返回平均值
    if ratios:
        return sum(ratios) / len(ratios)
    else:
        return None  # 或 return 0
