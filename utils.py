from os.path import (join as os_path_join, exists as os_path_exists)
from os import (listdir as os_listdir, makedirs as os_makedirs, mkdir as os_mkdir)
from shutil import rmtree as shutil_rmtree

from numpy import (int32 as np_int32, dstack as np_dstack, array as np_array, zeros as np_zeros, uint16 as np_uint16,
                   arange as np_arange, concatenate as np_concatenate, round as np_round, where as np_where,
                   float as np_float, amax as np_amax)
from pandas import (DataFrame as pd_DataFrame, ExcelWriter as pd_ExcelWriter, read_csv as pd_read_csv,
                    read_excel as pd_read_excel)

from cv2 import (imread as cv2_imread, fillPoly as cv2_fillPoly)
from roifile import ImagejRoi as roifile_ImagejRoi
from read_roi import read_roi_zip
from skimage.draw import ellipse
from time import sleep as time_sleep
from zipfile import (ZipFile as zipfile_ZipFile, ZIP_DEFLATED as zipfile_ZIP_DEFLATED)
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

from patchify import patchify, unpatchify
from sklearn.linear_model import LinearRegression
from tifffile import imwrite as tifwrite


def save_tif(filedir, filename, img):
    if filedir != "":
        if not os_path_exists(filedir):
            os_mkdir(filedir)
        tifwrite(filedir + "/" + filename, img)
    else:
        tifwrite(filename, img)


def read_tif(path, IMG_HEIGHT, IMG_WIDTH):
    img = cv2_imread(path, -1)
    h, w = img.shape
    h_expand = IMG_HEIGHT - h
    w_expand = IMG_WIDTH - w
    if h_expand + w_expand > 0:
        new_img = np_zeros((IMG_HEIGHT, IMG_WIDTH))
        new_img[0:h, 0:w] = img
        new_img[h + h_expand:, :w] = img[h - h_expand:h, :][::-1, :]
        new_img[:, -w_expand:] = new_img[:, -w_expand * 2:-w_expand][:, ::-1]
        img = new_img
    return img


def roi_to_bginfo(img, path, IMG_HEIGHT, IMG_WIDTH):
    mean_list = np_array([0, 0, 0])
    std_list = np_array([0, 0, 0])
    roi = read_roi_zip(path)
    mask_list = []
    for v in roi.values():
        mask_ = np_zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np_int32)
        if v['type'] == 'oval':
            l = v['left']
            t = v['top']
            h = v["height"] / 2
            w = v["width"] / 2
            rr, cc = ellipse(l + w, t + h, w, h)
            mask_[cc, rr] = [1]
        elif v['type'] == 'polygon':
            mask_x = v["x"]
            mask_y = v["y"]
            mask_cor = np_dstack((mask_x, mask_y))
            mask_cor = mask_cor.reshape((-1, 1, 2))
            cv2_fillPoly(mask_, np_int32([mask_cor]), color=1)
        mask_list.append(mask_)
    # plt.imshow(np_squeeze(mask_list[0]))
    # plt.show()
    r_list, g_list, b_list = [], [], []
    channel_list = np_zeros((3, len(mask_list), 2))
    for i, mask in enumerate(mask_list):
        selected_bg = img[np_where(mask > 0)[:2]]
        for j in range(3):
            c = selected_bg[:, j]
            mean, std = c.mean(), c.std()
            channel_list[j][i] = [mean, std]
    for i in range(3):
        channel = channel_list[i]
        mean_list[i], std_list[i] = channel.mean(axis=0)
    return mean_list, std_list


def train_beads(beads_path):
    """
    Read beads information from csv file
    :param beads_path:
    :return:
    """
    try:
        df_beads = pd_read_csv(beads_path, names=['red_y', 'red_x', 'green_y', 'green_x', 'blue_y', 'blue_x'])
    except Exception as e:
        raise e
    # green channel
    X_green = np_array(df_beads.loc[:, ['green_x', 'green_y']])
    Y_x_green = np_array(df_beads['red_x'] - df_beads['green_x'])
    Y_y_green = np_array(df_beads['red_y'] - df_beads['green_y'])

    lr_x_green = LinearRegression()
    lr_x_green.fit(X_green, Y_x_green)

    lr_y_green = LinearRegression()
    lr_y_green.fit(X_green, Y_y_green)

    pred_x_green = lr_x_green.predict(X_green)
    pred_y_green = lr_y_green.predict(X_green)

    # blue channel
    X_blue = np_array(df_beads.loc[:, ['blue_x', 'blue_y']])
    Y_x_blue = np_array(df_beads['red_x'] - df_beads['blue_x'])
    Y_y_blue = np_array(df_beads['red_y'] - df_beads['blue_y'])

    lr_x_blue = LinearRegression()
    lr_x_blue.fit(X_blue, Y_x_blue)

    lr_y_blue = LinearRegression()
    lr_y_blue.fit(X_blue, Y_y_blue)

    pred_x_blue = lr_x_blue.predict(X_blue)
    pred_y_blue = lr_y_blue.predict(X_blue)

    pred_beads = pd_DataFrame({'red_y': df_beads['red_y'], 'red_x': df_beads['red_x'],
                               'green_y': df_beads['green_y'] + pred_y_green,
                               'green_x': df_beads['green_x'] + pred_x_green,
                               'blue_y': df_beads['blue_y'] + pred_y_blue,
                               'blue_x': df_beads['blue_x'] + pred_x_blue})
    return lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, df_beads, pred_beads


def read_bg_info(excel_path, mead5sd_cell, _30sd_cell, sheet_name=1):
    # print(excel_path)
    try:
        df = pd_read_excel(excel_path, sheet_name=sheet_name)
    except Exception as e:
        raise e
    # mean5sd_coords = [cell_to_coords(x) for x in mead5sd_cell]
    _30std_coords = [cell_to_coords(x) for x in _30sd_cell]
    # mean_5_std = np_array([df.iloc[coord[0]][coord[1]] for coord in mean5sd_coords]).astype(np_float)
    _30std = np_array([df.iloc[coord[0]][coord[1]] for coord in _30std_coords]).astype(np_float)
    # mean_5_std = np_array(df.iloc[3][[3, 8, 13]]).astype(np_float)
    # _30std = np_array(df.iloc[4][[2, 7, 12]]).astype(np_float)
    # return mean_5_std, _30std
    return _30std


def cell_to_coords(cell_ref):
    row = int(cell_ref[1]) - 2
    col = ord(cell_ref[0]) - ord("A")
    return [row, col]


def img_substract(img, substraction):
    maxx = img.max()
    img_bgst = img - substraction
    img_bgst[img_bgst > maxx] = 0
    img_bgst[img_bgst < 0] = 0
    img_bgst = img_bgst.astype(np_uint16)
    return img_bgst


def patch_image(images_input, patched_height, patched_weight, patched_channel, step):
    input_shape = images_input.shape
    if len(input_shape) != 4:
        raise Exception("The dims of input images are not 4.")
    images_patched_list = []
    batch = images_input.shape[0]
    for i in range(batch):
        image = images_input[i]
        image_patched = patchify(image, (patched_height, patched_weight, patched_channel), step=step)
        images_patched_list.append(image_patched.reshape(-1, patched_height, patched_weight, patched_channel))
    images_patched = np_array(images_patched_list).reshape(-1, 256, 256, 3)
    return images_patched, np_array(images_patched_list).shape[:2], image_patched.shape[:2]


def unpatch_images(images_input, all_patched_shape, single_patched_shape, pathched_height, pathched_weight, ori_height,
                   ori_weight):
    """

    :param images_input: patched images
    :param all_patched_shape:  After appending all reshaped patched images
    :param single_patched_shape: a patched image's shape
    :param pathched_height: patchify target height
    :param pathched_weight: patchify target weight
    :param ori_height: original height
    :param ori_weight: original weight
    :return:
    """
    unpatched_images_input = images_input.reshape(all_patched_shape[0], all_patched_shape[1], pathched_height,
                                                  pathched_weight)
    unpatch_images_list = []
    for i in range(all_patched_shape[0]):
        image = unpatched_images_input[i].reshape(single_patched_shape[0], single_patched_shape[1], pathched_height,
                                                  pathched_weight)
        unpatch_image = unpatchify(image, (ori_height, ori_weight))
        unpatch_images_list.append(unpatch_image)
    return np_array(unpatch_images_list).reshape(-1, ori_height, ori_weight, 1)


def coord2list(coord):
    x, y, w, h = coord
    x = x - 1
    y = y - 1
    w = w + 2
    h = h + 2
    coord_list = [[y, x], [y + h, x], [y + h, x + w], [y, x + w]]
    return coord_list


def coord2roi(coords, output_folder, zip_name):
    temp_path = os_path_join(output_folder, "roi_temp")
    if not os_path_exists(temp_path):
        os_makedirs(temp_path)
    for i, coord in enumerate(coords):
        coord_list = coord2list(coord)
        roi = roifile_ImagejRoi.frompoints(coord_list)
        roi.tofile(temp_path + '/' + str(i) + '.roi')
    files_name = os_listdir(temp_path)
    files = [os_path_join(temp_path, i) for i in files_name]
    zip_file = os_path_join(output_folder, zip_name)
    get_zip(files, zip_file)
    shutil_rmtree(temp_path, ignore_errors=True)
    print("Create " + zip_file)


def get_zip(files, zip_name):
    zp = zipfile_ZipFile(zip_name, 'w', zipfile_ZIP_DEFLATED)
    for file in files:
        zp.write(file)
    zp.close()
    time_sleep(1)


# For qt
def golgi_lq_2pdf(images, lq_list, output_folder, pdf_name, columns=6):
    # for images
    num_images = len(images)
    rows = int(num_images / 4 + 1)

    # for histogram
    lq = np_array(lq_list)
    if len(lq_list) == 0:
        print("No valid golgi.")
        return
    mean, std = np_round(lq.mean(), 2), np_round(lq.std(), 2)
    max_, min_ = lq.max(), lq.min()
    x_tick_max = 0.5 * (int(max_ / 0.5) + 1)
    x_tick_min = 0.5 * (int(min_ / 0.5) - 1)

    x_ticks = np_arange(x_tick_min, x_tick_max + 0.25, 0.25)
    x_ticks_label = np_arange(x_tick_min, x_tick_max + 0.25, 0.5)

    output_path = os_path_join(output_folder, pdf_name)

    with PdfPages(output_path) as pdf:
        plt.figure(figsize=(3 * columns, 3 * rows + 10))
        gs = GridSpec(rows + 3, 6)
        for i in range(num_images):
            ax = plt.subplot(gs[int(i / columns), int(i % columns)])
            img = images[i] / np_amax(images[i], axis=(0, 1))
            ax.imshow(img)
            ax.set_title("lq: %.4f" % lq[i])
        ax = plt.subplot(gs[-3:-1, 1:columns - 1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlim(x_tick_min, x_tick_max)
        ax.set_xticks(x_ticks_label)

        count_arr, data, _ = ax.hist(lq, bins=len(x_ticks), color='gray', rwidth=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        y = count_arr.max()
        ax.set_ylim(0, y)
        ax.set_title("{}±{}(n={})".format(mean, std, len(lq)), verticalalignment='bottom', horizontalalignment='center')

        pdf.savefig()
        plt.close()
    print("Create " + output_path)


def golgi_plt2pdf(images, output_folder, pdf_name, columns=6):
    num_images = len(images)
    rows = int(num_images / 4 + 1)
    output_path = os_path_join(output_folder, pdf_name)
    with PdfPages(output_path) as pdf:
        plt.figure(figsize=(3 * columns, 3 * rows))
        for i in range(num_images):
            plt.subplot(rows, columns, 1 + i)
            img = images[i] / np_amax(images[i], axis=(0, 1))
            plt.imshow(img)
            # plt.imshow(images[i] / (images[i] + 0.1))
        pdf.savefig()
        plt.close()
    print("Create " + output_path)


def lq_hist2pdf(lq_list, output_folder, pdf_name):
    output_path = os_path_join(output_folder, pdf_name)
    lq = np_array(lq_list)
    if len(lq_list) == 0:
        print("No valid golgi.")
        return
    mean, std = np_round(lq.mean(), 2), np_round(lq.std(), 2)
    max_, min_ = lq.max(), lq.min()
    x_tick_max = 0.5 * (int(max_ / 0.5) + 1)
    x_tick_min = 0.5 * (int(min_ / 0.5) - 1)

    x_ticks = np_arange(x_tick_min, x_tick_max + 0.25, 0.25)
    x_ticks_label = np_arange(x_tick_min, x_tick_max + 0.25, 0.5)
    with PdfPages(output_path) as pdf:
        ax = plt.axes()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlim(x_tick_min, x_tick_max)
        plt.xticks(x_ticks_label)

        count_arr, data, _ = plt.hist(lq, bins=len(x_ticks), color='gray', rwidth=0.8)

        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        y = count_arr.max()
        plt.ylim(0, y)

        plt.title("{}±{}(n={})".format(mean, std, len(lq)), verticalalignment='bottom', horizontalalignment='center')
        pdf.savefig()
        plt.close()
    print("Create " + output_path)


def get_excel_writer(folder_path, filename):
    if not os_path_exists(folder_path):
        os_makedirs(folder_path)
    output_path = os_path_join(folder_path, filename)
    writer = pd_ExcelWriter(output_path)
    return writer, output_path


def style_apply(series, colors, back_ground=''):
    """
    :param series: one column of DataFrame, pd_Series
    :param colors: dict {column keywords:color}
    :param back_ground:
    :return:
    """
    ret = []
    flag = False
    for col in series:
        for keyword in colors:
            if series.name.endswith(keyword):
                flag = True
                ret.append('background-color: ' + colors[keyword])
        if not flag:
            ret.append("")
    return ret


def generate_excel_sheet(writer, sheet_name, header, data):
    if len(header) != data.shape[1]:
        raise Exception("Header length incompatible with data length.")
    df = pd_DataFrame(columns=header, data=data)
    style_df = df.style.apply(style_apply, colors={"_R": '#FF3300', "_G": '#33CC33', "_B": '#00CCFF'})
    style_df.to_excel(writer, sheet_name=sheet_name)


def generate_output_excel(original_intensity, shifted_intensity, original_centroid, shifted_centroid, lq, folder_path,
                          filename, sheet_name1="original data", sheet_name2="shifted data"):
    if not os_path_exists(folder_path):
        os_makedirs(folder_path)
    output_path = os_path_join(folder_path, filename)
    writer = pd_ExcelWriter(output_path)
    df1 = pd_DataFrame(
        {"intensity_R": original_intensity[:, 0], "Xm_R": original_centroid[:, :, 1][:, 0],
         "Ym_R": original_centroid[:, :, 0][:, 0],
         "intensity_G": original_intensity[:, 1], "Xm_G": original_centroid[:, :, 1][:, 1],
         "Ym_G": original_centroid[:, :, 0][:, 1],
         "intensity_B": original_intensity[:, 2], "Xm_B": original_centroid[:, :, 1][:, 2],
         "Ym_B": original_centroid[:, :, 0][:, 2]})

    df2 = pd_DataFrame(
        {"intensity_R": shifted_intensity[:, 0], "Xm_R": shifted_centroid[:, :, 1][:, 0],
         "Ym_R": shifted_centroid[:, :, 0][:, 0],
         "intensity_G": shifted_intensity[:, 1], "Xm_G": shifted_centroid[:, :, 1][:, 1],
         "Ym_G": shifted_centroid[:, :, 0][:, 1],
         "intensity_B": shifted_intensity[:, 2], "Xm_B": shifted_centroid[:, :, 1][:, 2],
         "Ym_B": shifted_centroid[:, :, 0][:, 2],
         "LQ": lq})
    df1.to_excel(writer, sheet_name=sheet_name1)
    df2.to_excel(writer, sheet_name=sheet_name2)
    writer.save()
    return "Create " + output_path


def write_data_excel(writer, original_centroid, valid_data_list, shifted_data_list):
    validShiftedCentroid, validShiftedLq, validIntensity = valid_data_list
    shiftedCentroid, shiftedLq, intensity = shifted_data_list
    if len(intensity) > 0:
        sheet1_name = "original data"
        sheet1_header = ["intensity_R", "Xm_R", "Ym_R", "intensity_G", "Xm_G", "Ym_G", "intensity_B", "Xm_B",
                         "Ym_B"]
        original_centroid = original_centroid.reshape(-1, 6)
        sheet1_data = np_concatenate([intensity, original_centroid], axis=1)[:,
                      [0, 4, 3, 1, 6, 5, 2, 8, 7]]
        generate_excel_sheet(writer, sheet_name=sheet1_name, header=sheet1_header,
                             data=sheet1_data)

        sheet2_name = "shifted data"
        sheet2_header = ["intensity_R", "Xm_R", "Ym_R", "intensity_G", "Xm_G", "Ym_G", "intensity_B", "Xm_B",
                         "Ym_B", "LQ"]
        shifted_centroid = shiftedCentroid.reshape(-1, 6)
        shifted_lq = shiftedLq.reshape(-1, 1)
        sheet2_data = np_concatenate([intensity, shifted_centroid, shifted_lq], axis=1)[:,
                      [0, 4, 3, 1, 6, 5, 2, 8, 7, 9]]
        generate_excel_sheet(writer, sheet_name=sheet2_name, header=sheet2_header,
                             data=sheet2_data)
    if len(validIntensity) > 0:
        sheet3_name = "criteria shifted data"
        valid_shifted_centroid = validShiftedCentroid.reshape(-1, 6)
        valid_shifted_lq = validShiftedLq.reshape(-1, 1)
        sheet3_data = np_concatenate([validIntensity, valid_shifted_centroid, valid_shifted_lq], axis=1)[:,
                      [0, 4, 3, 1, 6, 5, 2, 8, 7, 9]]
        generate_excel_sheet(writer, sheet_name=sheet3_name, header=sheet2_header,
                             data=sheet3_data)
    else:
        print("There is no background information(30*SD), so no criteria was used.")
