import logging
from configparser import ConfigParser as configparser_ConfigParser
from os.path import (split as os_path_split, join as os_path_join)
from os import listdir as os_listdir
from numpy import (zeros as np_zeros, uint16 as np_uint16,
                   dstack as np_dstack, array as np_array,
                   pad as np_pad, append as np_append)

from PyQt5.QtCore import (pyqtSignal as Signal, QObject, QThread)

from analysis import find_bg_mean, pred_2_contours, shift_and_criteria, filter_golgi
from model import load_model_func
from utils import train_beads, read_tif, read_bg_info, img_substract, roi_to_bginfo, save_tif, patch_image, \
    unpatch_images, get_excel_writer, write_data_excel, coord2roi, golgi_plt2pdf, lq_hist2pdf
from beads_processing import process_bead

from tensorflow.keras.utils import normalize as keras_norm

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class QtFunctions(QObject):
    append_text = Signal(str)
    process_finished = Signal(int)
    beads_finished = Signal(int)

    config = configparser_ConfigParser()
    config_file = "config.ini"
    beads_files = []

    lr_model = []

    def __init__(self, lastModelPath, lastModel, logger: logging.Logger):
        super().__init__()
        self.logger = logger
        # logger.info("function initial")
        try:
            self.config.read(self.config_file, encoding="utf-8")

            self.folder_path = self.config.get("file_path", "folder_path").split(";")
            self.imagesId = [os_path_split(image_file)[-1] for image_file in self.folder_path]
            self.beads_mode = self.config.getint("file_path", "beads_mode")
            self.beads_path = self.config.get("file_path", "beads_path")
            self.model_path = self.config.get("file_path", "model_path")

            # 1,2,3
            self.bg_mode = self.config.getint("parameters", "bg_mode")
            if self.bg_mode == 3:
                self.R2_R1_DIFF = self.config.getfloat("parameters", "r2_r1_diff")
                self.MAX_CONTOURS_AREA = self.config.getint("parameters", "max_contours_area")
            self.PRED_THRESHOLD = self.config.getfloat("parameters", "pred_threshold")
            self.SELECTED_THRESHOLD = self.config.getfloat("parameters", "selected_threshold")
            if self.bg_mode == 1:
                self.red_bgst_identifier = self.config.get("image_information", "red_bgst_identifier")
                self.green_bgst_identifier = self.config.get("image_information", "green_bgst_identifier")
                self.blue_bgst_identifier = self.config.get("image_information", "blue_bgst_identifier")
            else:
                self.red_identifier = self.config.get("image_information", "red_identifier")
                self.green_identifier = self.config.get("image_information", "green_identifier")
                self.blue_identifier = self.config.get("image_information", "blue_identifier")

            self.img_height = self.config.getint("image_information", "image_height")
            self.img_width = self.config.getint("image_information", "image_width")

            # self.red_mean5sd = self.config.get("bg_information", "red_mean5sd")
            # self.green_mean5sd = self.config.get("bg_information", "green_mean5sd")
            # self.blue_mean5sd = self.config.get("bg_information", "blue_mean5sd")

            self.red_30sd = self.config.get("bg_information", "red_30sd")
            self.green_30sd = self.config.get("bg_information", "green_30sd")
            self.blue_30sd = self.config.get("bg_information", "blue_30sd")

            # 1 or 0 boolean
            self.excel_cell_ref = self.config.get("bg_information", "excel_cell_ref") == "1"
        except Exception as e:
            self.logger.error("Error when read config: {}".format(e), exc_info=True)
            raise Exception("Error when read config: {}".format(e))

        self.new_size = max(self.img_height, self.img_width)
        self.img_channels = 3
        self.bg_roi_name = "BG-RoiSet.zip"
        self.beads_vector = []
        self.valid_lq = None
        self.valid_golgi = None
        self.progress_browser = None
        if lastModelPath is not None and lastModelPath == self.model_path:
            self.model = lastModel
        else:
            self.model = None

    # @Slot()
    # def pipeline(self, progress_browser: QTextBrowser):
    def pipeline(self):
        self.logger.info('start...thread id: {}'.format(QThread.currentThread()))
        self.logger.info('pipeline doing stuff in: {}'.format(QThread.currentThread()))
        try:
            # 1. Read beads and Train beads chromatic shift LinearRegression Model
            try:
                self.load_beads()
            except Exception as e:
                self.logger.error("Error when load beads information: {}".format(e), exc_info=True)
                raise Exception("Error when load beads information: {}".format(e))
            else:
                self.logger.info("1. Reading beads and training chromatic shift model sucessfully.")
                self.append_text.emit("1. Reading beads and training chromatic shift model sucessfully.")
            # 2. Read images
            try:
                self.beads_finished.emit(1)
                golgi_images, bg_30SD = self.read_images()
            except Exception as e:
                self.logger.error("Error when load images: {}".format(e), exc_info=True)
                raise Exception("Error when load images: {}".format(e))
            else:
                self.append_text.emit("2. Reading images sucessfully.")
                self.logger.info("2. Reading images sucessfully.")
                ...
            # 3. Preprocessing images.
            try:
                h_padding = int((self.new_size - self.img_height) / 2)
                w_padding = int((self.new_size - self.img_width) / 2)
                # self.new_size = max(self.img_width,self.img_height)
                # padding to (self.new_size, self.new_size,3)
                golgi_images_pad = np_pad(golgi_images,
                                          ((0, 0), (h_padding, h_padding), (w_padding, w_padding), (0, 0)),
                                          mode="symmetric")
                # patchify to 256*256*3
                images, all_shape, single_shape = patch_image(golgi_images_pad, 256, 256, 3, 256)
                self.logger.info("Patched image shape: {}".format(images.shape))
                # normalization
                images_norm = keras_norm(images, axis=1)
            except Exception as e:
                self.logger.error("Error when Load model and prediction: {}".format(e), exc_info=True)
                raise Exception("Error when Load model and prediction: {}".format(e))
            else:
                self.append_text.emit("3. Preprocessing images sucessfully.")
                self.logger.info("3. Preprocessing images sucessfully.")

            # 4. Load model and prediction
            try:
                if self.model is None:
                    self.model = load_model_func(self.model_path)
                preds = self.model.predict(images_norm, verbose=1)

                pred_unpatched = unpatch_images(preds, all_shape, single_shape, 256, 256, self.img_height, self.img_width)
                pred_unpadding = pred_unpatched[:, h_padding:self.new_size - h_padding, w_padding:self.new_size - w_padding,
                                 :]
            except Exception as e:
                self.logger.error("Error when Load model and prediction: {}".format(e), exc_info=True)
                raise Exception("Error when Load model and prediction: {}".format(e))
            else:
                self.append_text.emit("4. Load model and prediction sucessfully.")
                self.logger.info("4. Load model and prediction sucessfully.")
                ...
            # 5.Analysis predicted result
            totalLQ = []
            for j, pred in enumerate(pred_unpadding):
                composited_golgi = golgi_images[j]
                # convert prediction to contours
                golgi_contours = pred_2_contours(composited_golgi, pred, self.PRED_THRESHOLD, self.SELECTED_THRESHOLD)
                # filtering golgi by peak check
                try:
                    golgi, golgi_rect_coord, golgi_centroid, invalid_golgi = filter_golgi(composited_golgi,
                                                                                          golgi_contours)
                except Exception as e:
                    err_str = "[filtering golgi by peak check] Error in {}, skip this folder. Error is {}".format(
                        self.folder_path[j], e)
                    self.logger.error(err_str, exc_info=True)
                    raise Exception(err_str)

                # Chromatic Shift and Check 3 criteria
                try:
                    validGolgiIndex, valid_data_list, shifted_data_list = shift_and_criteria(
                        golgi, golgi_centroid, self.lr_model[0], self.lr_model[1], self.lr_model[2], self.lr_model[3],
                        bg_30SD[j])
                except Exception as e:
                    self.logger.error(
                        "[Chromatic Shift and Check 3 criteria] Error in {}, skip this folder. Error is {}".format(
                            self.folder_path[j], e), exc_info=True)
                    raise Exception(
                        "[Chromatic Shift and Check 3 criteria] Error in {}, skip this folder. Error is {}".format(
                            self.folder_path[j], e))
                    continue
                # self.valid_data_list = valid_data_list
                # self.shifted_data_list = shifted_data_list

                validShiftedCentroid, validShiftedLq, validIntensity = valid_data_list
                shiftedCentroid, _, _ = shifted_data_list
                totalLQ.extend(validShiftedLq)

                valid_rect_coord = golgi_rect_coord[validGolgiIndex]
                valid_golgi = golgi[validGolgiIndex]
                if self.valid_lq is None:
                    self.valid_lq = validShiftedLq
                else:
                    self.valid_lq = np_append(self.valid_lq, validShiftedLq)

                if self.valid_golgi is None:
                    self.valid_golgi = valid_golgi
                else:
                    self.valid_golgi = np_append(self.valid_golgi, valid_golgi)

                # # Create excel writer
                # excel_writer, out_path = get_excel_writer(folder_path=self.folder_path[j][j] + "/result",
                #                                           filename=self.imagesId[j] + ".xlsx")
                #
                # try:
                #     # Write data into excel
                #     write_data_excel(excel_writer, golgi_centroid, valid_data_list, shifted_data_list)
                #     # excel_writer.save()
                #     print("Create {}".format(out_path))
                #     coord2roi(valid_rect_coord, self.folder_path[j] + "/result", "roi.zip")
                #     golgi_plt2pdf(valid_golgi, self.folder_path[j] + "/result", "golgi_valid.pdf")
                #     golgi_plt2pdf(golgi, self.folder_path[j] + "/result", "golgi_shifted.pdf")
                #     lq_hist2pdf(validShiftedLq, self.folder_path[j] + "/result", "golgi_lq_histogram.pdf")
                # except Exception as e:
                #     print(e)
                # finally:
                #     excel_writer.close()
                #     excel_writer.handles = None

            self.append_text.emit("5. Analysis predicted result sucessfully.")
            self.logger.info("5. Analysis predicted result sucessfully.")
            self.process_finished.emit(1)
        except Exception as e:
            self.append_text.emit("{}".format(e))

    def load_beads(self):
        self.logger.info("Beads path: {}".format(self.beads_path))
        if self.beads_mode == 1:
            # csv file
            lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, beads_df, pred_beads = train_beads(self.beads_path)
            self.lr_model = [lr_x_blue, lr_y_blue, lr_x_green, lr_y_green]
            self.beads_vector = [beads_df, pred_beads]
        elif self.beads_mode == 2:
            # beads image
            beads_r, beads_g, beads_b = None, None, None
            files = os_listdir(self.beads_path)
            if self.bg_mode == 1:
                red_identifier = self.red_bgst_identifier
                green_identifier = self.green_bgst_identifier
                blue_identifier = self.blue_bgst_identifier
            else:
                red_identifier = self.red_identifier
                green_identifier = self.green_identifier
                blue_identifier = self.blue_identifier
            for i, file_name in enumerate(files):
                file_path = os_path_join(self.beads_path, file_name)
                if red_identifier.upper() in file_name.upper():
                    beads_r = file_path
                    continue
                if green_identifier.upper() in file_name.upper():
                    beads_g = file_path
                    continue
                if blue_identifier.upper() in file_name.upper():
                    beads_b = file_path
                    continue
            beads_tif_path_list = [beads_r, beads_g, beads_b]
            if beads_r is None or beads_g is None or beads_b is None:
                raise Exception("Can not find enough beads tif files")
                # ". Found {} files, but requires 3 files.".format(len(beads_tif_path_list)))
            lr_x_blue, lr_y_blue, lr_x_green, lr_y_green, beads_df, pred_beads = process_bead(beads_tif_path_list,
                                                                                              False)
            self.lr_model = [lr_x_blue, lr_y_blue, lr_x_green, lr_y_green]
            self.beads_vector = [beads_df, pred_beads]
        else:
            raise Exception("Wrong beads mode in {}".format(self.config_file))

    def read_images(self):
        self.logger.info("Image folder path: {}".format(self.folder_path))
        golgiImages = np_zeros((len(self.folder_path), self.img_height, self.img_width, self.img_channels),
                               dtype=np_uint16)
        self.logger.info("Number of folders: {}".format(len(self.folder_path)))
        _30SD_list = np_zeros((len(self.folder_path), 3), dtype=float)
        flag = False
        for n, image_id in enumerate(self.folder_path):
            files = os_listdir(image_id)
            r_file_path = ""
            g_file_path = ""
            b_file_path = ""
            red_tif = None
            red_bgst_tif = None
            green_tif = None
            green_bgst_tif = None
            blue_tif = None
            blue_bgst_tif = None

            bg_roi_path = None
            bg_info_path = None
            for file_name in files:
                file_path = os_path_join(image_id, file_name)
                if self.bg_mode == 1:
                    # Read BGST tif files and bg excel file
                    if file_name.endswith("xlsx"):
                        bg_info_path = file_path
                    elif self.red_bgst_identifier.upper() in file_name.upper() \
                            and file_name.upper().endswith(".TIF"):
                        red_bgst_tif = read_tif(file_path, self.img_height, self.img_width)
                        r_file_path = file_path
                    elif self.green_bgst_identifier.upper() in file_name.upper() \
                            and file_name.upper().endswith(".TIF"):
                        green_bgst_tif = read_tif(file_path, self.img_height, self.img_width)
                        g_file_path = file_path
                    elif self.blue_bgst_identifier.upper() in file_name.upper() \
                            and file_name.upper().endswith(".TIF"):
                        blue_bgst_tif = read_tif(file_path, self.img_height, self.img_width)
                        b_file_path = file_path
                elif self.bg_mode == 2:
                    # Read original tif files
                    if self.red_identifier.upper() in file_name.upper() \
                            and file_name.upper().endswith(".TIF"):
                        red_tif = read_tif(file_path, self.img_height, self.img_width)
                        r_file_path = file_path
                        continue
                    if self.green_identifier.upper() in file_name.upper() \
                            and file_name.upper().endswith(".TIF"):
                        green_tif = read_tif(file_path, self.img_height, self.img_width)
                        g_file_path = file_path
                        continue
                    if self.blue_identifier.upper() in file_name.upper() \
                            and file_name.upper().endswith(".TIF"):
                        blue_tif = read_tif(file_path, self.img_height, self.img_width)
                        b_file_path = file_path
                        continue
                    # Read  BG-RoiSet.zip
                    if self.bg_roi_name.upper() in file_name.upper():
                        bg_roi_path = file_path
                elif self.bg_mode == 3:
                    # Using algorithm to find bg mean%std
                    # No other images need to read.
                    ...
            err_msg = ""
            if self.bg_mode == 1:
                if bg_info_path is None and self.excel_cell_ref:
                    err_msg += "Lack background information csv file. "
                if red_bgst_tif is None:
                    err_msg += "Lack red channel tif."
                if blue_bgst_tif is None:
                    err_msg += "Lack blue channel tif."
                if green_bgst_tif is None:
                    err_msg += "Lack green channel tif."
                else:
                    try:
                        # mean_5sd_coords = [self.red_mean5sd, self.green_mean5sd, self.blue_mean5sd]
                        _30sd_coords = [self.red_30sd, self.green_30sd, self.blue_30sd]
                        if self.excel_cell_ref:
                            # mean_5std, _30SD = read_bg_info(bg_info_path, mean_5sd_coords, _30sd_coords)
                            _30SD = read_bg_info(bg_info_path, None, _30sd_coords)
                        else:
                            # mean_5std = np_array(mean_5sd_coords)
                            _30SD = np_array(_30sd_coords)
                    except Exception as e:
                        raise e
                    if red_bgst_tif is not None and green_bgst_tif is not None and blue_bgst_tif is not None:
                        composite = np_dstack((red_bgst_tif, green_bgst_tif, blue_bgst_tif))
                    else:
                        raise Exception("No bgst tif files")
                        # cy3_bgst_tif = img_substract(red_tif, mean_5std[0])
                        # gfp_bgst_tif = img_substract(green_tif, mean_5std[1])
                        # cy5_bgst_tif = img_substract(blue_tif, mean_5std[2])
                        # composite = np_dstack((cy3_bgst_tif, gfp_bgst_tif, cy5_bgst_tif))
                    if len(composite) > 0:
                        golgiImages[n] = composite
                        _30SD_list[n] = _30SD
            elif self.bg_mode == 2:
                if bg_roi_path is None:
                    err_msg += "Lack BG-RoiSet.zip file. "
                    break
                if red_tif is not None and blue_tif is not None and green_tif is not None:
                    composite = np_dstack((red_tif, green_tif, blue_tif))
                else:
                    if red_tif is None:
                        err_msg += "Lack red channel tif."
                    if blue_tif is None:
                        err_msg += "Lack blue channel tif."
                    if green_tif is None:
                        err_msg += "Lack green channel tif."
                    break

                try:
                    mean_arr, std_arr = roi_to_bginfo(composite, bg_roi_path, self.img_height, self.img_width)
                except Exception as e:
                    raise e
                mean_5std = mean_arr + 5 * std_arr
                _30SD = 30 * std_arr
                red_bgst_tif = img_substract(red_tif, mean_5std[0])
                save_tif("", r_file_path.replace(".tif", "-BGST.tif"), red_bgst_tif)

                green_bgst_tif = img_substract(green_tif, mean_5std[1])
                save_tif("", g_file_path.replace(".tif", "-BGST.tif"), green_bgst_tif)

                blue_bgst_tif = img_substract(blue_tif, mean_5std[2])
                save_tif("", b_file_path.replace(".tif", "-BGST.tif"), blue_bgst_tif)

                composite_bgst = np_dstack((red_bgst_tif, green_bgst_tif, blue_bgst_tif))
                if len(composite_bgst) > 0:
                    golgiImages[n] = composite_bgst
                    _30SD_list[n] = _30SD
            elif self.bg_mode == 3:
                if red_tif is None:
                    err_msg += "Lack red channel tif."
                if blue_tif is None:
                    err_msg += "Lack blue channel tif."
                if green_tif is None:
                    err_msg += "Lack green channel tif."
                if len(err_msg) > 0:
                    break
                std_arr = []
                composited = []
                rgb_file_path = [r_file_path, g_file_path, b_file_path]
                for i, img_1c in enumerate([red_tif, green_tif, blue_tif]):
                    try:
                        bgst_, mean, std = find_bg_mean(img_1c, self.MAX_CONTOURS_AREA, self.R2_R1_DIFF)
                    except Exception as e:
                        raise e
                    std_arr.append(std)
                    composited.append(bgst_)
                    save_tif("", rgb_file_path[i].replace(".tif", "-BGST.tif"), bgst_)
                composite_bgst = np_dstack(composited)
                if len(composite_bgst) > 0:
                    golgiImages[n] = composite_bgst
                    _30SD_list[n] = 30 * np_array(std_arr)
            if len(err_msg) != 0:
                err_msg = " Path:[{}]: {}".format(image_id, err_msg)
                flag = True
        if flag:
            raise Exception(err_msg)
        return golgiImages, _30SD_list

    def get_model(self):
        return self.model_path, self.model

    def get_beads_vector(self):
        return self.beads_vector

    def get_golgi_lq(self):
        return self.valid_golgi, self.valid_lq

#
# func = QtFunctions(None, None)
# func.pipeline()
