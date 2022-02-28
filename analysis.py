from numpy import (uint8 as np_uint8, ones as np_ones, array as np_array, zeros as np_zeros, uint16 as np_uint16,
                   sum as np_sum, mean as np_mean, round as np_round, where as np_where, sqrt as np_sqrt,
                   dot as np_dot, histogram as np_histogram, linalg as np_linalg, cross as np_cross)
from pandas import DataFrame as pd_DataFrame
from queue import Queue as queue_Queue
from cv2 import (cvtColor as cv2_cvtColor, COLOR_GRAY2RGB as cv2_COLOR_GRAY2RGB, threshold as cv2_threshold,
                 THRESH_BINARY as cv2_THRESH_BINARY, THRESH_OTSU as cv2_THRESH_OTSU, morphologyEx as cv2_morphologyEx,
                 DIST_L2 as cv2_DIST_L2, connectedComponents as cv2_connectedComponents, dilate as cv2_dilate,
                 watershed as cv2_watershed, distanceTransform as cv2_distanceTransform, MORPH_OPEN as cv2_MORPH_OPEN,
                 subtract as cv2_subtract, RETR_TREE as cv2_RETR_TREE, contourArea as cv2_contourArea,
                 CHAIN_APPROX_NONE as cv2_CHAIN_APPROX_NONE, findContours as cv2_findContours,
                 boundingRect as cv2_boundingRect, split as cv2_split, RETR_EXTERNAL as cv2_RETR_EXTERNAL
                 )
from skimage.feature import peak_local_max as feature_peak_local_max
from skimage.measure import regionprops_table as measure_regionprops_table

from scipy.ndimage.measurements import center_of_mass as ndi_center_of_mass

shift_list = [[-1, 0], [0, 1], [1, 0], [0, -1], [-1, -1], [-1, 1], [1, 1], [1, -1]]
check_list = [[-1, 0], [0, 1], [1, 0], [0, -1]]


def pred_2_contours(golgi, pred, pred_threshold, selected_threshold):
    mask_useful = cv2_cvtColor((pred > pred_threshold).astype(np_uint8), cv2_COLOR_GRAY2RGB)
    mask_grey = mask_useful[:, :, 0]
    ret1, thresh = cv2_threshold(mask_grey, 0, 255, cv2_THRESH_BINARY + cv2_THRESH_OTSU)

    kernel = np_ones((3, 3), np_uint8)
    opening = cv2_morphologyEx(thresh, cv2_MORPH_OPEN, kernel, iterations=2)

    dist_transform = cv2_distanceTransform(opening, cv2_DIST_L2, 5)

    ret2, sure_fg = cv2_threshold(dist_transform, 0.2 * dist_transform.max(), 255, 0)
    sure_fg = np_uint8(sure_fg)

    ret3, markers = cv2_connectedComponents(sure_fg)
    sure_bg = cv2_dilate(opening, kernel, iterations=3)
    unknown = cv2_subtract(sure_bg, sure_fg)

    markers = markers + 10
    markers[unknown == 255] = 0
    markers = cv2_watershed(mask_useful, markers)

    props = measure_regionprops_table(markers, intensity_image=pred[:, :, 0],
                                      properties=['label', 'coords', 'centroid',
                                                  'area', 'mean_intensity'])

    df_props = pd_DataFrame(props)
    df_props = df_props[df_props.mean_intensity > selected_threshold]

    df_props.sort_values(by='mean_intensity', ascending=False, inplace=True)
    df_props.reset_index(inplace=True)

    golgi_coords = np_array(df_props['coords'])
    golgi_contours = find_golgi_contours(golgi, golgi_coords)
    return golgi_contours


def find_golgi_contours(composited_img, coord_array):
    """
    Find each predicted golgi contours
    :param composited_img:
    :param coord_array:
    :return:
    """
    x_max, y_max, _ = composited_img.shape
    x_min = 0
    y_min = 0
    composted_labeled = np_zeros((x_max, y_max, 4), np_uint16)
    composted_labeled[:, :, :3] = composited_img
    counter_lists = []
    for i, coords in enumerate(coord_array):
        contours_queue = queue_Queue()
        counter_list = []
        x, y = (0, 0)
        for _, coord in enumerate(coords):
            x, y = coord
            if np_sum(composited_img[x][y]) == 0:
                continue
            else:
                break
        # step1: find most top contour point
        while True:
            up = x - 1
            if up > x_min:
                if np_sum(composited_img[up][y]) > 0:
                    x = up
                else:
                    break
            else:
                break
        start = [x, y]
        contours_queue.put(start)
        # step 3: find contour
        while True:
            x, y = find_contours(composted_labeled, contours_queue)
            if x == -1:
                break
            elif x == -2:
                counter_list = None
                break
            elif x == 0:
                continue
            else:
                counter_list.append([x, y])
                composted_labeled[x, y, 3] = 1
        # step 4: finish one golgi
        if counter_list is not None:
            counter_lists.append(counter_list)
    return counter_lists


def filter_golgi(composited, golgiContours):
    validGolgi = []
    validGolgi_rect_coord = []
    validGolgi_centroid = []
    invalidGolgi = []

    oriGolgi = composited.copy()

    # Find predicted golgi's centroids
    for i in range(len(golgiContours)):
        golgi_contours_np = np_array(golgiContours[i])
        if len(golgi_contours_np) == 0:
            continue
        x, y, w, h = cv2_boundingRect(golgi_contours_np)
        selected_golgi = oriGolgi[x - 1:x + w + 1, y - 1:y + h + 1]
        new_contours = golgi_contours_np - [x - 1, y - 1]
        # clearn other singal in the rectangle
        selected_golgi, valid = clean_contours(selected_golgi, new_contours)
        if not valid:
            invalidGolgi.append(selected_golgi)
            continue
        r, g, b = cv2_split(selected_golgi)
        r_peak, r_msg = filter_by_peak(r)
        g_peak, g_msg = filter_by_peak(g)
        b_peak, b_msg = filter_by_peak(b)
        if r_peak & g_peak & b_peak:
            r_centroid = np_round(ndi_center_of_mass(r), 5) + [x - 1, y - 1] + [0.5, 0.5]
            g_centroid = np_round(ndi_center_of_mass(g), 5) + [x - 1, y - 1] + [0.5, 0.5]
            b_centroid = np_round(ndi_center_of_mass(b), 5) + [x - 1, y - 1] + [0.5, 0.5]
            validGolgi_centroid.append(np_array([r_centroid, g_centroid, b_centroid]))
            validGolgi_rect_coord.append((x, y, w, h))
            validGolgi.append(selected_golgi)
        else:
            invalidGolgi.append(selected_golgi)
    validGolgi_centroid = np_array(validGolgi_centroid)
    validGolgi_rect_coord = np_array(validGolgi_rect_coord)
    validGolgi = np_array(validGolgi, dtype=object)
    return validGolgi, validGolgi_rect_coord, validGolgi_centroid, invalidGolgi


def find_contours(img_labeled, con_queue):
    # finish
    if con_queue.empty():
        return [-1, -1]
    point = con_queue.get()
    x_max, y_max, _ = img_labeled.shape
    x_min, y_min = (0, 0)

    x, y = point
    # check if contours is close to the edge of image
    if x == x_min or y == y_min or x == x_max or y == y_max:
        return [-2, -2]

    # check if point has been regarded as a contour
    if img_labeled[x, y, 3] == 1:
        return [0, 0]
    for shift in shift_list:
        x, y = [a + b for a, b in zip(shift, point)]
        if x < 0 or y < 0 or x >= x_max or y >= y_max:
            continue
        # check if point is in the img
        if np_sum(img_labeled[x, y, :3]) == 0:
            continue
        else:
            if check_contours(img_labeled, [x, y]):
                con_queue.put([x, y])
    return point


def check_contours(img_labeled, point):
    """
    :param img_labeled:
    :param point: [x,y] list
    :return: if point is a edge
    """
    x_max, y_max, _ = img_labeled.shape
    x, y = point
    if img_labeled[x, y, 3] != 0:
        return False
    for shift in check_list:
        x, y = [a + b for a, b in zip(shift, point)]
        if x < 0 or y < 0 or x >= x_max or y >= y_max:
            return False
        if np_sum(img_labeled[x, y, :3]) == 0:
            return True
    return False


def clean_contours(img, contours):
    h, w, _ = img.shape
    for x in range(h):
        y_array = np_where(contours[:, 0] == x)[0]
        y_min = w
        y_max = 0
        if len(y_array) > 0:
            y_min = contours[y_array][:, 1].min()
            y_max = contours[y_array][:, 1].max()
        for y in range(0, y_min):
            img[x, y] = [0, 0, 0]
        for y in range(y_max + 1, w):
            img[x, y] = [0, 0, 0]
    return img, True


def filter_by_peak(channel, min_distance=1, kernel_size=0, threshold_rel=0.9):
    if channel.sum() == 0:
        return False, "No signal in this channel."
    _, binary = cv2_threshold(channel, 0, 255, cv2_THRESH_BINARY)
    contours, _ = cv2_findContours(binary.astype(np_uint8), cv2_RETR_TREE, cv2_CHAIN_APPROX_NONE)
    if len(contours) != 1:
        return False, "More than 1 contour in this channel."
    if kernel_size == 0:
        footprint = None
    else:
        footprint = np_ones((kernel_size, kernel_size))
    local_maxi_coord = feature_peak_local_max(channel, min_distance=min_distance,
                                              threshold_rel=threshold_rel,
                                              footprint=footprint)  # find peak
    num_of_peak = len(np_where(local_maxi_coord is True))
    if num_of_peak == 1:
        return True, "Num of peak is 1."
    elif num_of_peak > 2:
        return False, "Num of peaks is {}.".format(len(local_maxi_coord))
    elif num_of_peak == 2:
        dist = np_sqrt((local_maxi_coord[0][0] - local_maxi_coord[1][0]) ** 2 + (
                local_maxi_coord[0][1] - local_maxi_coord[1][1]) ** 2)
        if dist <= 10:
            return True, "Num of peaks is 2, and distance between peaks is {}.".format(dist)
        else:
            return False, "Num of peaks is 2, and distance between peaks is {}.".format(dist)
    else:
        return False, ""


def chromatic_shift(lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, r_centroid, g_centroid, b_centroid):
    """

    :param lr_x_blue: LinearRegression Model
    :param lr_y_blue: LinearRegression Model
    :param lr_x_green: LinearRegression Model
    :param lr_y_green: LinearRegression Model
    :param r_centroid: one center of mass coordinate
    :param g_centroid: one center of mass coordinate
    :param b_centroid: one center of mass coordinate
    :return:
    """
    pred_x_blue = lr_x_blue.predict([b_centroid])
    pred_y_blue = lr_y_blue.predict([b_centroid])

    pred_x_green = lr_x_green.predict([g_centroid])
    pred_y_green = lr_y_green.predict([g_centroid])

    r_np = np_array(r_centroid).reshape(2)
    g_new = np_round((g_centroid[0] + pred_x_green, g_centroid[1] + pred_y_green), 5).reshape(2)
    b_new = np_round((b_centroid[0] + pred_x_blue, b_centroid[1] + pred_y_blue), 5).reshape(2)
    return r_np, g_new, b_new


def LQ(r, g, b):
    """
    Calculate LQ. 蓝绿向量在蓝红向量上的投影/蓝红向量
    :param r: r channel's centroid
    :param g: g channel's centroid
    :param b: b channel's centroid
    :return: LQ
    """
    x = g - b
    y = r - b
    # x_proj = np_dot(x, y) / np_sqrt(y.dot(y))
    # lq = x_proj / np_sqrt(y.dot(y))
    lq = np_dot(x, y) / np_dot(y, y)
    return lq


def cal_total_intensity(golgi):
    """
    Calucuate golgi each channel's total intensity
    :param golgi: selected golgi
    :return: np.array([r_intensity, g_intensity, b_intensity])
    """
    r = golgi[:, :, 0]
    g = golgi[:, :, 1]
    b = golgi[:, :, 2]
    r_intensity = r.sum()
    g_intensity = g.sum()
    b_intensity = b.sum()
    return np_array([r_intensity, g_intensity, b_intensity])


# sum of pixel in each channel larger than 30 times of background SD
def criteria_1(golgi, SD_30):
    r_sum = golgi[:, :, 0].sum()
    g_sum = golgi[:, :, 1].sum()
    b_sum = golgi[:, :, 2].sum()
    return r_sum >= SD_30[0] and g_sum >= SD_30[1] and b_sum >= SD_30[2]


# d1 larger than 70nm. 1pixel = 64nm. d1 = distance between red and blue centroid X(X in ImageJ = Y in opencv)
def criteria_2(r, g, b):
    y = r - b
    d1 = np_sqrt(y.dot(y))
    return 70 / 64 <= d1 <= 10


# tan(a) or tan(b) must be <= 0.3.

def cal_tan(n1, n2):
    cos_ = np_dot(n1, n2) / (np_linalg.norm(n1) * np_linalg.norm(n2))
    sin_ = np_cross(n1, n2) / (np_linalg.norm(n1) * np_linalg.norm(n2))
    return sin_, cos_


def criteria_3(r, g, b):
    n1 = b - g
    n2 = b - r
    t1_sin, t1_cos = cal_tan(n1, n2)
    t1_tan = abs(t1_sin / t1_cos)

    n3 = r - g
    n4 = r - b
    t2_sin, t2_cos = cal_tan(n3, n4)
    t2_tan = abs(t2_sin / t2_cos)

    return t1_tan <= 0.3 or t2_tan <= 0.3


def shift_and_criteria(golgi, golgiCentroid, lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, _30SD):
    shiftedCentroid = []
    validShiftedCentroid = []

    intensityList = []
    validIntensity = []

    shiftedLq = []
    validShiftedLq = []

    validGolgiIndex = []
    for i in range(len(golgi)):
        centroid_r, centroid_g, centroid_b = golgiCentroid[i]
        r_c, g_c, b_c = chromatic_shift(lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, centroid_r, centroid_g,
                                        centroid_b)
        lq = LQ(r_c, g_c, b_c)
        shiftedLq.append(lq)
        shiftedCentroid.append(np_array([r_c, g_c, b_c]))

        intensity = cal_total_intensity(golgi[i])
        intensityList.append(intensity)

        if criteria_1(golgi[i], _30SD) & criteria_2(r_c, g_c, b_c) & criteria_3(r_c, g_c, b_c):
            validShiftedLq.append(lq)
            validShiftedCentroid.append(np_array([r_c, g_c, b_c]))
            validGolgiIndex.append(i)
            validIntensity.append(intensity)
    shiftedCentroid = np_array(shiftedCentroid)
    validShiftedCentroid = np_array(validShiftedCentroid)
    shiftedLq = np_array(shiftedLq)
    validShiftedLq = np_array(validShiftedLq)
    intensityList = np_array(intensityList)
    validIntensity = np_array(validIntensity)

    return validGolgiIndex, [validShiftedCentroid, validShiftedLq, validIntensity], [shiftedCentroid, shiftedLq,
                                                                                     intensityList]


def get_img_hist(img, bins=5):
    df = pd_DataFrame(img, columns=['intensity'])
    df_count = df.value_counts().rename_axis('intensity').reset_index(name='counts').sort_values(
        by='intensity').reset_index(drop=True).reset_index()
    df_count['group'] = (df_count['index'] / bins).astype('int')
    df_count_group = df_count.groupby('group').sum().drop('index', axis=1)
    df_count_group['members'] = df_count.groupby('group').nunique('index')['index']
    df_count_group['mean_intensity'] = (df_count_group['intensity'] / df_count_group['members']).astype('int')
    df_count_group = df_count_group.sort_values(by='counts', ascending=False).reset_index()
    r1 = df_count_group.loc[0]['mean_intensity']
    return df_count_group, r1


def sub_mean_r1r2(img, r1, r2, a=1, b=1, plus=0):
    bg = img[img >= r1]
    bg = bg[bg <= r2]
    mean, std = bg.mean(), bg.std()
    img_substract = int(mean * a + std * b + plus)
    maxx = img.max()
    img_bgst = img - img_substract
    img_bgst[img_bgst > maxx] = 0
    img_bgst[img_bgst < 0] = 0
    img_bgst = img_bgst.astype(np_uint16)
    return img_bgst, mean, std


def area_contours(img_1c, max_area=-1):
    _, binary = cv2_threshold(img_1c, 0, 255, cv2_THRESH_BINARY)
    contours, hierarchy = cv2_findContours(binary.astype(np_uint8), cv2_RETR_EXTERNAL, cv2_CHAIN_APPROX_NONE)
    area_list = []
    for contour in contours:
        area = cv2_contourArea(contour)
        if max_area > 0:
            if area < max_area:
                area_list.append(area)
        else:
            area_list.append(area)
    return area_list


def find_bg_mean(img_1c, MAX_CONTOURS_AREA, R2_R1_DIFF):
    hist_ori, r1 = get_img_hist(img_1c.flatten())
    r2 = r1
    bgst_, mean, std = sub_mean_r1r2(img_1c, r1, r2, b=5)
    area_list = area_contours(bgst_)
    max_contour_area = max(area_list)
    flag = False
    bins_1 = []
    max_contours_area = []
    # bins_2_5 = []
    # bins_6_9 = []
    bins_ratio = []
    while True:
        max_contours_area.append(max_contour_area)
        if max_contour_area < MAX_CONTOURS_AREA:
            if r2 - r1 == R2_R1_DIFF:
                break
        bgst_, mean, std = sub_mean_r1r2(img_1c, r1, r2, b=5)
        r2 = r2 + 10
        area_list = area_contours(bgst_)
        max_contour_area = max(area_list)
        counts, bins = np_histogram(area_list, bins=range(0, 1000, 20))
        bins_1.append(counts[0])
        # bins_2_5.append(np.mean(counts[1:5]))
        # bins_6_9.append(np.mean(counts[5:9]))
        ratio = np_mean(counts[5:9]) / np_mean(counts[1:5])
        bins_ratio.append(ratio)
        if max_contour_area < 10000 and counts[0] < 150 and np_mean(counts[1:5]) >= 20 and ratio < 0.31:
            flag = True
            break
    if not flag:
        for i, ratio in enumerate(bins_ratio):
            if ratio < 0.26 and bins_1[i] < 135 and max_contours_area[i] < MAX_CONTOURS_AREA:
                r2 = r1 + 10 * i
                flag = True
                break
    if not flag:
        raise Exception("Please find BG manually.")
    else:
        bgst_, mean, std = sub_mean_r1r2(img_1c, r1, r2 - 10, b=5)
    return bgst_, mean, std
