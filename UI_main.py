import time
from sys import (exit as sys_exit, argv as sys_argv)
from configparser import ConfigParser as configparser_ConfigParser
from os.path import (exists as os_path_exists, split as os_path_split, isfile as os_path_isfile, isdir as os_path_isdir,
                     dirname as os_path_dirname)
from os import (listdir as os_listdir, getcwd as os_getcwd, mkdir as os_mkdir)
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRegularExpression, pyqtSignal as Signal, QThread
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLineEdit, QListView, QAbstractItemView, \
    QTreeView, QWidget, QVBoxLayout
from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from numpy import (delete as np_delete, amax as np_amax, arange as np_arange)

from lq_dialog import LqDialog
from utils import golgi_lq_2pdf
from ui.mainUI3 import Ui_MainWindow
from functions_for_qt import QtFunctions
from pandas import DataFrame as pd_DataFrame

import logging

GOLGI_DEFAULT = 0.4
PIXEL_DEFAULT = 0.05
MAX_CON_AREA_DEFAULT = 10000
R2_R1_DEFAULT = 5000

config_file = "config.ini"


def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    datetime = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = './Logs/'
    log_name = log_path + datetime + '.log'
    logfile = log_name
    if not os_path_exists(log_path):
        os_mkdir(log_path)

    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)

    console_logger = logging.StreamHandler()
    console_logger.setLevel(logging.INFO)
    console_logger.setFormatter(formatter)

    logger.addHandler(console_logger)
    logger.addHandler(handler)

    return logger


def open_file_dialog(lineEdit: QLineEdit, mode=1, filetype="", folder=""):
    fileDialog = QFileDialog()
    if len(folder) > 0:
        fileDialog.setDirectory(folder)
    path = ""
    if mode == 1:
        # multiple directories
        fileDialog.setFileMode(QFileDialog.Directory)
        # path = fileDialog.getExistingDirectory()
        fileDialog.setOption(QFileDialog.DontUseNativeDialog, True)
        fileDialog.setOption(QFileDialog.ShowDirsOnly, True)
        file_view = fileDialog.findChild(QListView)

        # to make it possible to select multiple directories:
        if file_view:
            file_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        f_tree_view = fileDialog.findChild(QTreeView)
        if f_tree_view:
            f_tree_view.setSelectionMode(QAbstractItemView.ExtendedSelection)

        if fileDialog.exec():
            path_list = fileDialog.selectedFiles()
            path = ';'.join(path_list)
    elif mode == 2:
        # single file
        fileDialog.setFileMode(QFileDialog.ExistingFile)
        if filetype != "":
            name_filter = "{} files (*.{} *.{})".format(filetype, filetype, filetype.upper())
            path = fileDialog.getOpenFileName(filter=name_filter)[0]
        else:
            path = fileDialog.getOpenFileName()[0]
    if len(path) > 0:
        lineEdit.setText(path)
    return path


class MainWindow(QMainWindow):
    # test_img = np.array([[[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
    #                      [[0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
    #                      [[0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 0, 0]],
    #                      [[0, 0, 1], [1.0, 1, 1], [1, 1, 1], [1, 1, 1], [0, 0, 0]],
    #                      [[0, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 0, 0]],
    #                      [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, 0]]])
    start_backgroung_work = Signal()

    def __init__(self):
        super().__init__()
        self.logger = get_logger()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.resize(1141, 870)

        self.cfg = configparser_ConfigParser()
        if os_path_exists(config_file):
            self.cfg.read(config_file)
            self.set_params_from_cfg()
        else:
            self.cfg['file_path'] = {}
            self.cfg['parameters'] = {}
            self.cfg['image_information'] = {}
            self.cfg['bg_information'] = {}

        # set tab 2 disable
        self.ui.tabWidget.setCurrentIndex(0)
        self.ui.tabWidget.setTabEnabled(1, False)
        # TAB 1
        # error_browser_height = 40
        # self.ui.error_browser.setFixedHeight(error_browser_height)
        # self.ui.err_frame.setFixedHeight(error_browser_height + 10)
        self.ui.line_1.setVisible(False)
        self.ui.param_bg_algorithmn.setVisible(False)

        # set validator on line edit
        self.ui.param_pixel_threshold.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+\.[0-9]+$")))
        self.ui.param_golgi_threshold.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+\.[0-9]+$")))
        self.ui.param_contours_area_max.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+$")))
        self.ui.param_r2_r1_diff.setValidator(QRegularExpressionValidator(QRegularExpression("[0-9]+\.[0-9]+$")))

        self.ui.Parameter_group.setMaximumHeight(200)
        self.ui.bg_mode_combobox.currentIndexChanged.connect(self.show_bg_mode3)

        self.ui.image_browse_btn.clicked.connect(self.img_btn_handler)
        self.ui.model_browse_btn.clicked.connect(self.model_btn_handler)
        self.ui.bead_browse_btn.clicked.connect(self.bead_btn_handler)
        self.ui.threshold_default_btn.clicked.connect(self.threshold_set_default)
        self.ui.bg_3_default_btn.clicked.connect(self.bg3_set_default)
        self.ui.model_path_default_btn.clicked.connect(self.model_path_set_default)

        self.ui.check_btn.clicked.connect(self.check_params)
        self.ui.next_btn.clicked.connect(self.next_btn_handler)

        # textChanged
        self.ui.folder_line_edit.textChanged.connect(self.handle_textchanged)
        self.ui.model_path_line_edit.textChanged.connect(self.handle_textchanged)
        self.ui.bead_line_edit.textChanged.connect(self.handle_textchanged)

        self.ui.red_identifier.textChanged.connect(self.handle_textchanged)
        self.ui.red_bgst_identifier.textChanged.connect(self.handle_textchanged)
        self.ui.red_30sd.textChanged.connect(self.handle_textchanged)
        # self.ui.red_mean5sd.textChanged.connect(self.handle_textchanged)

        self.ui.blue_identifier.textChanged.connect(self.handle_textchanged)
        self.ui.blue_bgst_identifier.textChanged.connect(self.handle_textchanged)
        # self.ui.blue_mean5sd.textChanged.connect(self.handle_textchanged)
        self.ui.blue_30sd.textChanged.connect(self.handle_textchanged)

        self.ui.green_identifier.textChanged.connect(self.handle_textchanged)
        self.ui.green_bgst_identifier.textChanged.connect(self.handle_textchanged)
        self.ui.green_30sd.textChanged.connect(self.handle_textchanged)
        # self.ui.green_mean5sd.textChanged.connect(self.handle_textchanged)

        self.ui.image_width.textChanged.connect(self.handle_textchanged)
        self.ui.image_height.textChanged.connect(self.handle_textchanged)

        self.ui.param_contours_area_max.textChanged.connect(self.handle_textchanged)
        self.ui.param_r2_r1_diff.textChanged.connect(self.handle_textchanged)
        self.ui.param_pixel_threshold.textChanged.connect(self.handle_textchanged)
        self.ui.param_golgi_threshold.textChanged.connect(self.handle_textchanged)

        self.ui.bg_mode_combobox.currentIndexChanged.connect(self.handle_textchanged)
        self.ui.bead_type_combobox.currentIndexChanged.connect(self.handle_textchanged)

        self.ui.excel_cell_ref_ratio.clicked.connect(self.handle_textchanged)
        self.ui.read_data_ratio.clicked.connect(self.handle_textchanged)

        # TAB 2
        self.func = None
        self.thread = QThread(self)
        self.axes_dict = {}
        self.ui.start_btn.clicked.connect(self.start_process)

        self.selected_list = []
        self.ui.show_hist_btn.clicked.connect(self.show_hist)
        self.ui.save_result_btn.clicked.connect(self.save_golgi_result)
        self.ui.save_beads_vector_map.setDisabled(True)
        self.ui.save_beads_vector_map.clicked.connect(self.save_beads_maps)

        self.lq_dialog = None

        self.saved_model = None
        self.saved_model_path = None

        self.beads_maps = None

    def handle_textchanged(self):
        self.ui.next_btn.setEnabled(False)
        self.ui.start_btn.setEnabled(False)
        self.ui.error_browser.clear()

    # @Slot(str)
    def update_message(self, text):
        self.ui.progress_text.append(text)

    def subplot_onclick(self, event):
        axes = event.inaxes
        if axes is None:
            return
        # axes_id = self.axes_dict[hash(id(axes))]
        if id(axes) not in self.axes_dict.keys():
            return
        axes_id = self.axes_dict[id(axes)]
        if axes_id in self.selected_list:
            self.selected_list.remove(axes_id)
        else:
            self.selected_list.append(axes_id)
        if len(axes.patches) > 0:
            axes.patches.pop()
            event.canvas.draw()
        else:
            ax_h, ax_w = axes.bbox.height, axes.bbox.width
            axes.add_patch(Rectangle((-0.5, -0.5), ax_h, ax_w, alpha=0.7))
            event.canvas.draw()

    def model_path_set_default(self):
        self.ui.model_path_line_edit.setText("./model/model.h5")

    def img_btn_handler(self):
        open_file_dialog(self.ui.folder_line_edit, mode=1)
        pass

    def model_btn_handler(self):
        open_file_dialog(self.ui.model_path_line_edit, filetype="h5", mode=2)
        pass

    def bead_btn_handler(self):
        if self.ui.bead_type_combobox.currentIndex() == 0:
            open_file_dialog(self.ui.bead_line_edit, filetype="csv", mode=2)
        else:
            open_file_dialog(self.ui.bead_line_edit, mode=1)
        pass

    def show_bg_mode3(self):
        if self.ui.bg_mode_combobox.currentIndex() == 2:
            self.ui.param_bg_algorithmn.setVisible(True)
            self.ui.line_1.setVisible(True)
            self.ui.Parameter_group.setMaximumHeight(277)
        else:
            self.ui.param_bg_algorithmn.setVisible(False)
            self.ui.line_1.setVisible(False)
            self.ui.Parameter_group.setMaximumHeight(200)

    def bg3_set_default(self):
        self.ui.param_r2_r1_diff.setText(str(R2_R1_DEFAULT))
        self.ui.param_contours_area_max.setText(str(MAX_CON_AREA_DEFAULT))

    def threshold_set_default(self):
        self.ui.param_pixel_threshold.setText(str(PIXEL_DEFAULT))
        self.ui.param_golgi_threshold.setText(str(GOLGI_DEFAULT))

    def set_params_from_cfg(self):
        self.ui.folder_line_edit.setText(self.cfg.get("file_path", "folder_path"))
        self.ui.bead_type_combobox.setCurrentIndex(self.cfg.getint("file_path", "beads_mode") - 1)
        self.ui.bead_line_edit.setText(self.cfg.get("file_path", "beads_path"))
        self.ui.model_path_line_edit.setText(self.cfg.get("file_path", "model_path"))

        self.ui.bg_mode_combobox.setCurrentIndex(self.cfg.getint("parameters", "bg_mode") - 1)
        self.ui.param_r2_r1_diff.setText(self.cfg.get("parameters", "r2_r1_diff"))
        self.ui.param_contours_area_max.setText(self.cfg.get("parameters", "max_contours_area"))
        self.ui.param_pixel_threshold.setText(self.cfg.get("parameters", "pred_threshold"))
        self.ui.param_golgi_threshold.setText(self.cfg.get("parameters", "selected_threshold"))

        self.ui.red_identifier.setText(self.cfg.get("image_information", "red_identifier"))
        self.ui.green_identifier.setText(self.cfg.get("image_information", "green_identifier"))
        self.ui.blue_identifier.setText(self.cfg.get("image_information", "blue_identifier"))

        self.ui.red_bgst_identifier.setText(self.cfg.get("image_information", "red_bgst_identifier"))
        self.ui.green_bgst_identifier.setText(self.cfg.get("image_information", "green_bgst_identifier"))
        self.ui.blue_bgst_identifier.setText(self.cfg.get("image_information", "blue_bgst_identifier"))

        self.ui.image_height.setText(self.cfg.get("image_information", "image_height"))
        self.ui.image_width.setText(self.cfg.get("image_information", "image_width"))

        # self.ui.red_mean5sd.setText(self.cfg.get("bg_information", "red_mean5sd"))
        # self.ui.green_mean5sd.setText(self.cfg.get("bg_information", "green_mean5sd"))
        # self.ui.blue_mean5sd.setText(self.cfg.get("bg_information", "blue_mean5sd"))

        self.ui.red_30sd.setText(self.cfg.get("bg_information", "red_30sd"))
        self.ui.green_30sd.setText(self.cfg.get("bg_information", "green_30sd"))
        self.ui.blue_30sd.setText(self.cfg.get("bg_information", "blue_30sd"))

        if self.cfg.get("bg_information", "excel_cell_ref") == "1":
            self.ui.excel_cell_ref_ratio.setChecked(True)
            self.ui.read_data_ratio.setChecked(False)
        else:
            self.ui.excel_cell_ref_ratio.setChecked(False)
            self.ui.read_data_ratio.setChecked(True)

    def check_params(self):
        folder_path = self.ui.folder_line_edit.text()
        model_path = self.ui.model_path_line_edit.text()
        beads_type = self.ui.bead_type_combobox.currentIndex()
        beads_path = self.ui.bead_line_edit.text()

        bg_mode = self.ui.bg_mode_combobox.currentText()
        r2_r1_diff = self.ui.param_r2_r1_diff.text()
        max_contours_area = self.ui.param_contours_area_max.text()
        pixel_threshold = self.ui.param_pixel_threshold.text()
        golgi_threshold = self.ui.param_golgi_threshold.text()

        err_msg = ""
        if len(folder_path) > 0:
            self.cfg['file_path']['folder_path'] = folder_path
        else:
            err_msg += "Image Folder is empty.\n"

        if len(model_path) > 0:
            if not model_path.endswith("h5"):
                err_msg += "Model File Path is wrong,should be h5 file.\n"
            elif not os_path_isfile(model_path):
                err_msg += "Model File is not exist. \n"
            else:
                self.cfg['file_path']['model_path'] = model_path
        else:
            err_msg += "Model File Path is empty.\n"

        if beads_type >= 0:
            self.cfg['file_path']['beads_mode'] = str(beads_type + 1)
        else:
            err_msg += "Beads Type is wrong.\n"

        if len(beads_path) > 0:
            if beads_type == 0:
                if not beads_path.endswith("csv"):
                    err_msg += "Beads Path is wrong, should be csv file.\n"
                elif not os_path_isfile(beads_path):
                    err_msg += "Beads csv file is not exist.\n"
                else:
                    self.cfg['file_path']['beads_path'] = beads_path
            elif beads_type == 1:
                if not os_path_isdir(beads_path):
                    err_msg += "Beads folder is not exist.\n"
                else:
                    self.cfg['file_path']['beads_path'] = beads_path
        else:
            err_msg += "Beads Path is empty.\n"

        if len(bg_mode) > 0:
            self.cfg['parameters']['bg_mode'] = bg_mode
        else:
            err_msg += "BG MODE is wrong.\n"

        if len(self.ui.red_identifier.text()) > 0:
            self.cfg['image_information']['red_identifier'] = self.ui.red_identifier.text()
        else:
            err_msg += "red_identifier is empty.\n"

        if len(self.ui.green_identifier.text()) > 0:
            self.cfg['image_information']['green_identifier'] = self.ui.green_identifier.text()
        else:
            err_msg += "green_identifier is empty.\n"

        if len(self.ui.blue_identifier.text()) > 0:
            self.cfg['image_information']['blue_identifier'] = self.ui.blue_identifier.text()
        else:
            err_msg += "blue_identifier is empty.\n"

        if len(self.ui.image_height.text()) > 0:
            self.cfg['image_information']['image_height'] = self.ui.image_height.text()
        else:
            err_msg += "image_height is empty.\n"

        if len(self.ui.image_width.text()) > 0:
            self.cfg['image_information']['image_width'] = self.ui.image_width.text()
        else:
            err_msg += "image_width is empty.\n"

        if bg_mode != "3":
            if len(self.ui.red_bgst_identifier.text()) > 0:
                self.cfg['image_information']['red_bgst_identifier'] = self.ui.red_bgst_identifier.text()
            else:
                err_msg += "red_bgst_identifier is empty.\n"

            if len(self.ui.green_bgst_identifier.text()) > 0:
                self.cfg['image_information']['green_bgst_identifier'] = self.ui.green_bgst_identifier.text()
            else:
                err_msg += "green_bgst_identifier is empty.\n"

            if len(self.ui.blue_bgst_identifier.text()) > 0:
                self.cfg['image_information']['blue_bgst_identifier'] = self.ui.blue_bgst_identifier.text()
            else:
                err_msg += "blue_bgst_identifier is empty.\n"
        if bg_mode == "2":
            if len(r2_r1_diff) > 0:
                self.cfg['parameters']['R2_R1_DIFF'] = r2_r1_diff
            else:
                err_msg += "R2_R1_DIFF is empty.\n"

            if len(max_contours_area) > 0:
                self.cfg['parameters']['MAX_CONTOURS_AREA'] = max_contours_area
            else:
                err_msg += "MAX_CONTOURS_AREA is empty.\n"

        if len(pixel_threshold) > 0:
            self.cfg['parameters']['PRED_THRESHOLD'] = pixel_threshold
        else:
            err_msg += "Pixel_Possibility_Threshold is empty.\n"

        if len(golgi_threshold) > 0:
            self.cfg['parameters']['SELECTED_THRESHOLD'] = golgi_threshold
        else:
            err_msg += "Golgi_Possibility_Threshold is empty.\n"

        # if len(self.ui.red_mean5sd.text()) > 0:
        #     self.cfg['bg_information']['red_mean5sd'] = self.ui.red_mean5sd.text()
        # else:
        #     err_msg += "red_mean5sd is empty.\n"
        # if len(self.ui.green_mean5sd.text()) > 0:
        #     self.cfg['bg_information']['green_mean5sd'] = self.ui.green_mean5sd.text()
        # else:
        #     err_msg += "green_mean5sd is empty.\n"
        # if len(self.ui.blue_mean5sd.text()) > 0:
        #     self.cfg['bg_information']['blue_mean5sd'] = self.ui.blue_mean5sd.text()
        # else:
        #     err_msg += "blue_mean5sd is empty.\n"

        if len(self.ui.red_30sd.text()) > 0:
            self.cfg['bg_information']['red_30sd'] = self.ui.red_30sd.text()
        else:
            err_msg += "green_30sd is empty.\n"
        if len(self.ui.green_30sd.text()) > 0:
            self.cfg['bg_information']['green_30sd'] = self.ui.green_30sd.text()
        else:
            err_msg += "green_30sd is empty.\n"
        if len(self.ui.blue_30sd.text()) > 0:
            self.cfg['bg_information']['blue_30sd'] = self.ui.blue_30sd.text()
        else:
            err_msg += "blue_30sd is empty.\n"
        if self.ui.excel_cell_ref_ratio.isChecked():
            self.cfg['bg_information']['excel_cell_ref'] = "1"
        else:
            self.cfg['bg_information']['excel_cell_ref'] = "0"

        # Check identifier for each image folder
        folder_list = folder_path.split(";")

        # Check if excel cell exists in image folder when excel_cell_ref == 1
        if self.ui.excel_cell_ref_ratio.isChecked():
            for folder in folder_list:
                file_list = os_listdir(folder)
                flag = False
                for file in file_list:
                    if file.endswith("xlsx") or file.endswith("xls") or file.endswith("csv"):
                        flag = True
                        break
                if not flag:
                    err_msg += "Path {} has no excel file. Bue choose background data type as excel cell reference".format(
                        folder)
        if len(err_msg) > 0:
            n_enter = err_msg.count("\n")
            self.ui.next_btn.setDisabled(True)
            self.ui.error_browser.setVisible(True)
            self.ui.error_browser.setText(err_msg)
            if n_enter > 1:
                error_browser_height = 20 * n_enter + 2
            else:
                error_browser_height = 30

        else:
            with open(config_file, 'w') as configfile:
                self.cfg.write(configfile)
            self.ui.next_btn.setEnabled(True)
            self.ui.start_btn.setEnabled(True)
            self.ui.error_browser.setVisible(True)
            self.ui.error_browser.setText("PASS")
            error_browser_height = 30
        # self.ui.error_browser.setFixedHeight(error_browser_height)
        # self.ui.err_frame.setFixedHeight(error_browser_height + 10)

    def next_btn_handler(self):
        self.ui.tabWidget.setCurrentIndex(1)
        self.ui.tabWidget.setTabEnabled(1, True)
        # self.resize(1141, 900)
        # x, y = self.geometry().getCoords()[0:2]
        # self.move(x, y - 75)

    def show_golgi_result(self):
        valid_golgi, valid_lq = self.func.get_golgi_lq()

        qScrollLayout = QVBoxLayout(self.ui.scroll_golgi_content)
        qfigWidget = QWidget(self.ui.scroll_golgi_content)

        columns = 6
        rows = int(len(valid_golgi) / columns + 1)
        static_canvas = FigureCanvas(Figure(figsize=(1.7 * columns, 1.7 * rows)))
        static_canvas.mpl_connect('button_press_event', self.subplot_onclick)

        axes_dict = {}
        subplot_axes = static_canvas.figure.subplots(rows, columns)
        static_canvas.figure.tight_layout()
        static_canvas.figure.subplots_adjust(hspace=0.3)
        for i, axes in enumerate(subplot_axes.reshape(-1)):
            if i >= valid_golgi.shape[0]:
                axes.axis("off")
            else:
                # key = hash(id(axes))
                key = id(axes)
                axes_dict[key] = i
                axes.title.set_text("lq: %.4f" % valid_lq[i])
                axes.title.set_size(10)
                # calcualte max value in each channel
                img = valid_golgi[i] / np_amax(valid_golgi[i], axis=(0, 1))
                axes.imshow(img)

        self.axes_dict = axes_dict

        plotLayout = QVBoxLayout()
        plotLayout.addWidget(static_canvas)
        qfigWidget.setLayout(plotLayout)
        static_canvas.setMinimumSize(static_canvas.size())
        qScrollLayout.addWidget(qfigWidget)
        self.ui.scroll_golgi_content.setLayout(qScrollLayout)
        self.ui.scroll_golgi_content.show()

    def start_process(self):
        self.ui.progress_text.clear()
        # self.ui.beads_vector_scroll_content.hide()
        # self.ui.scroll_content.hide()

        self.ui.scroll_golgi_content = QtWidgets.QWidget()
        self.ui.scroll_golgi.setWidget(self.ui.scroll_golgi_content)

        self.ui.scroll_beads_content = QtWidgets.QWidget()
        self.ui.scroll_beads.setWidget(self.ui.scroll_beads_content)

        self.logger.info("start")
        try:
            self.logger.info('start doing stuff in: {}'.format(QThread.currentThread()))
            self.func = QtFunctions(self.saved_model_path, self.saved_model, self.logger)
            self.func.moveToThread(self.thread)
            self.start_backgroung_work.connect(self.func.pipeline)
            self.func.append_text.connect(self.update_message)
            self.func.process_finished.connect(self.show_golgi_result)
            self.func.beads_finished.connect(self.show_vector_map)

            self.thread.start()
            self.start_backgroung_work.emit()
            # func.pipeline(self.ui.progress_text)
        except Exception as e:
            self.ui.progress_text.append("{}".format(e))
        else:
            self.saved_model_path, self.saved_model = self.func.get_model()

    def show_hist(self):
        valid_golgi, valid_lq = self.func.get_golgi_lq()
        if self.ui.pick_btn.isChecked():
            # pick_select
            selected_lq = valid_lq[self.selected_list]
        else:
            # drop_select
            selected_lq = np_delete(valid_lq, self.selected_list)
        if len(selected_lq) > 0:
            self.lq_dialog = LqDialog(selected_lq)
            self.lq_dialog.show()

    def save_golgi_result(self):
        valid_golgi, valid_lq = self.func.get_golgi_lq()
        if self.ui.pick_btn.isChecked():
            # pick_select
            selected_golgi = valid_golgi[self.selected_list]
            selected_lq = valid_lq[self.selected_list]
        else:
            # drop_select
            selected_golgi = np_delete(valid_golgi, self.selected_list)
            selected_lq = np_delete(valid_lq, self.selected_list)
        save_path, save_type = QFileDialog.getSaveFileName(self, "Save File", "./Golgi Gallery & LQ Histogram.pdf",
                                                           'pdf(*.pdf)')
        folder, file = os_path_split(save_path)
        golgi_lq_2pdf(selected_golgi, selected_lq, folder, file)

    def show_vector_map(self):
        beads_df, pred_beads = self.func.get_beads_vector()

        arrow_df = pd_DataFrame({'diff_green_y': 100 * (beads_df['green_y'] - beads_df['red_y']),
                                 'diff_green_x': 100 * (beads_df['green_x'] - beads_df['red_x']),
                                 'diff_blue_y': 100 * (beads_df['blue_y'] - beads_df['red_y']),
                                 'diff_blue_x': 100 * (beads_df['blue_x'] - beads_df['red_x']),
                                 'pred_diff_green_y': 100 * (pred_beads['green_y'] - beads_df['red_y']),
                                 'pred_diff_green_x': 100 * (pred_beads['green_x'] - beads_df['red_x']),
                                 'pred_diff_blue_y': 100 * (pred_beads['blue_y'] - beads_df['red_y']),
                                 'pred_diff_blue_x': 100 * (pred_beads['blue_x'] - beads_df['red_x'])})

        qScrollLayout = QVBoxLayout(self.ui.scroll_beads_content)
        qfigWidget = QWidget(self.ui.scroll_beads_content)

        static_canvas = FigureCanvas(Figure(figsize=(7, 7)))
        text_size = 6
        r = static_canvas.get_renderer()

        subplot_axes = static_canvas.figure.subplots(2, 2)
        # static_canvas.figure.tight_layout()
        # static_canvas.figure.subplots_adjust(hspace=0.3)
        # 0,0 original arrow
        subplot_axes[0, 0].title.set_text("original arrow")
        subplot_axes[0, 0].title.set_size(10)

        x_lim = (min(beads_df["red_x"].min(), (beads_df["red_x"] + arrow_df['diff_green_x']).min(),
                     (beads_df["red_x"] + arrow_df['diff_blue_x']).min()) - 10,
                 max(beads_df["red_x"].max(), (beads_df["red_x"] + arrow_df['diff_green_x']).max(),
                     (beads_df["red_x"] + arrow_df['diff_blue_x']).max()) + 10)
        y_lim = (min(beads_df["red_y"].min(), (beads_df["red_y"] + arrow_df['diff_green_y']).min(),
                     (beads_df["red_y"] + arrow_df['diff_blue_y']).min()) - 10,
                 max(beads_df["red_y"].max(), (beads_df["red_y"] + arrow_df['diff_green_y']).max(),
                     (beads_df["red_y"] + arrow_df['diff_blue_y']).max()) + 10)
        subplot_axes[0, 0].set_xlim(x_lim)

        subplot_axes[0, 0].set_ylim(y_lim)

        subplot_axes[0, 1].set_xlim(x_lim)

        subplot_axes[0, 1].set_ylim(y_lim)

        for index, row in beads_df.iterrows():
            subplot_axes[0, 0].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["diff_green_x"],
                                     arrow_df.iloc[index]["diff_green_y"], color='green', head_width=6, lw=0.4)
            subplot_axes[0, 0].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["diff_blue_x"],
                                     arrow_df.iloc[index]["diff_blue_y"], color='blue', head_width=6, lw=0.4)
        subplot_axes[0, 0].add_patch(
            Rectangle((0, 0), int(self.ui.image_width.text()), int(self.ui.image_height.text()), fill=False,
                      linewidth=1))
        subplot_axes[0, 0].axis('off')

        # 0,1 shifted arrow
        subplot_axes[0, 1].title.set_text("shifted arrow")
        subplot_axes[0, 1].title.set_size(10)
        for index, row in pred_beads.iterrows():
            subplot_axes[0, 1].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["pred_diff_green_x"],
                                     arrow_df.iloc[index]["pred_diff_green_y"], color='green', head_width=6, lw=0.4)
            subplot_axes[0, 1].arrow(row['red_x'], row['red_y'], arrow_df.iloc[index]["pred_diff_blue_x"],
                                     arrow_df.iloc[index]["pred_diff_blue_y"], color='blue', head_width=6, lw=0.4)
        subplot_axes[0, 1].add_patch(
            Rectangle((0, 0), int(self.ui.image_width.text()), int(self.ui.image_height.text()), fill=False,
                      linewidth=1))
        subplot_axes[0, 1].axis('off')

        # scattor
        scatter_df = pd_DataFrame(
            {'green_x': beads_df['green_x'] - beads_df['red_x'], 'green_y': beads_df['green_y'] - beads_df['red_y'],
             'blue_x': beads_df['blue_x'] - beads_df['red_x'], 'blue_y': beads_df['blue_y'] - beads_df['red_y'],
             'pred_green_x': pred_beads['green_x'] - pred_beads['red_x'],
             'pred_green_y': pred_beads['green_y'] - pred_beads['red_y'],
             'pred_blue_x': pred_beads['blue_x'] - pred_beads['red_x'],
             'pred_blue_y': pred_beads['blue_y'] - pred_beads['red_y']})
        scatter_df = scatter_df.apply(lambda x: 67 * x)

        # 1,0 original related
        subplot_axes[1, 0].title.set_text("original related")
        subplot_axes[1, 0].title.set_size(10)
        x_max = max(abs(min(scatter_df[["green_x", "blue_x"]].min())), max(scatter_df[["green_x", "blue_x"]].max()))
        y_max = max(abs(min(scatter_df[["green_y", "blue_y"]].min())), max(scatter_df[["green_y", "blue_y"]].max()))
        lim = max(x_max, y_max)
        if lim % 10 != 0:
            lim = (int(lim / 10) + 1) * 10
        xy_lim = (-lim, lim)
        xy_ticks = np_arange(-lim, lim + 10, 10)
        xy_labels = [xy_ticks[i] if i % 2 == 0 else " " for i in range(len(xy_ticks))]
        subplot_axes[1, 0].set_xlim(xy_lim)
        subplot_axes[1, 0].set_ylim(xy_lim)
        subplot_axes[1, 0].set_xticks(xy_ticks, xy_labels)
        subplot_axes[1, 0].set_yticks(xy_ticks, xy_labels)
        subplot_axes[1, 0].scatter(scatter_df["green_x"], scatter_df["green_y"], c='g', s=5, alpha=0.3)
        subplot_axes[1, 0].scatter(scatter_df["blue_x"], scatter_df["blue_y"], c='b', s=5, alpha=0.3)
        subplot_axes[1, 0].axvline(c="black", lw=1)
        subplot_axes[1, 0].axhline(c="black", lw=1)
        subplot_axes[1, 0].spines['top'].set_visible(False)
        subplot_axes[1, 0].spines['right'].set_visible(False)
        subplot_axes[1, 0].set_xlabel("x position of center (nm)")
        subplot_axes[1, 0].set_ylabel("y position of center (nm)")

        # box information
        title_text = subplot_axes[1, 0].text(lim, lim, "mean±SD", ha="right", va="top", ma="center",
                                             size=text_size)
        bb_title = title_text.get_window_extent(renderer=r)
        bb_title = bb_title.transformed(subplot_axes[1, 0].transData.inverted())

        text_height = bb_title.ymax - bb_title.ymin

        green_text = subplot_axes[1, 0].text(bb_title.x1, bb_title.y0 - text_height / 2,
                                             "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(scatter_df[
                                                                                                         'green_x'].mean(),
                                                                                                     scatter_df[
                                                                                                         'green_x'].std(),
                                                                                                     scatter_df[
                                                                                                         'green_y'].mean(),
                                                                                                     scatter_df[
                                                                                                         'green_y'].std()),
                                             ha="right", va="top", ma="left", size=text_size, c='g')
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 0].transData.inverted())
        green_text.set_ha("left")
        green_width = bb_green.xmax - bb_green.xmin

        blue_text = subplot_axes[1, 0].text(bb_green.x0, bb_green.y0,
                                            "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(scatter_df[
                                                                                                        'blue_x'].mean(),
                                                                                                    scatter_df[
                                                                                                        'blue_x'].std(),
                                                                                                    scatter_df[
                                                                                                        'blue_y'].mean(),
                                                                                                    scatter_df[
                                                                                                        'blue_y'].std()),
                                            ha="left", va="top", ma="left", size=text_size, c='b')
        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 0].transData.inverted())
        blue_width = bb_blue.xmax - bb_blue.xmin

        max_width = max(blue_width, green_width)

        if blue_width > green_width:
            shift = bb_blue.x1 - bb_green.x1
        else:
            shift = 0

        new_x0 = bb_blue.x0 - shift - text_height / 2
        blue_text.set_x(new_x0)
        green_text.set_x(new_x0)

        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 0].transData.inverted())
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 0].transData.inverted())

        title_shift = bb_title.x0 - text_height / 2 - bb_blue.x0
        title_text.set_x(bb_title.x1 - title_shift / 2)
        title_text.set_y(bb_title.y1 - text_height / 2)

        count_text = subplot_axes[1, 0].text(bb_blue.x1, bb_blue.y0,
                                             "n={}".format(len(scatter_df)), ha="right", va="top",
                                             ma="center",
                                             size=text_size)
        bb_count = count_text.get_window_extent(renderer=r)
        bb_count = bb_count.transformed(subplot_axes[1, 0].transData.inverted())

        count_shift = bb_count.x0 - bb_blue.x0
        count_text.set_x(bb_count.x1 - count_shift / 2)

        bbox_shift = text_height / 3
        bbox_xy = (bb_blue.x0 - text_height / 3, bb_count.y0)
        bbox_width = max_width + 2 * bbox_shift
        bbox_height = bb_title.y1 - bb_count.y0

        subplot_axes[1, 0].add_patch(
            Rectangle(bbox_xy, bbox_width, bbox_height, fc=(1, 1, 1), ec=(0, 0, 0), lw=0.5, alpha=0.5))

        # 1,1 shifted related
        subplot_axes[1, 1].title.set_text("shifted related")
        subplot_axes[1, 1].title.set_size(10)
        subplot_axes[1, 1].set_xlim(xy_lim)
        subplot_axes[1, 1].set_ylim(xy_lim)
        subplot_axes[1, 1].set_xticks(xy_ticks, xy_labels)
        subplot_axes[1, 1].set_yticks(xy_ticks, xy_labels)

        subplot_axes[1, 1].scatter(scatter_df["pred_green_x"], scatter_df["pred_green_y"], c='g', s=5, alpha=0.3)
        subplot_axes[1, 1].scatter(scatter_df["pred_blue_x"], scatter_df["pred_blue_y"], c='b', s=5, alpha=0.3)

        subplot_axes[1, 1].axvline(c="black", lw=1)
        subplot_axes[1, 1].axhline(c="black", lw=1)
        subplot_axes[1, 1].spines['top'].set_visible(False)
        subplot_axes[1, 1].spines['right'].set_visible(False)
        subplot_axes[1, 1].set_xlabel("x position of center (nm)")
        subplot_axes[1, 1].set_ylabel("y position of center (nm)")

        # box information
        title_text = subplot_axes[1, 1].text(lim, lim, "mean±SD", ha="right", va="top", ma="center",
                                             size=text_size)
        bb_title = title_text.get_window_extent(renderer=r)
        bb_title = bb_title.transformed(subplot_axes[1, 1].transData.inverted())

        green_text = subplot_axes[1, 1].text(bb_title.x1, bb_title.y0 - text_height / 2,
                                             "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(scatter_df[
                                                                                                         'green_x'].mean(),
                                                                                                     scatter_df[
                                                                                                         'green_x'].std(),
                                                                                                     scatter_df[
                                                                                                         'green_y'].mean(),
                                                                                                     scatter_df[
                                                                                                         'green_y'].std()),
                                             ha="right", va="top", ma="left", size=text_size, c='g')
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 1].transData.inverted())
        green_text.set_ha("left")
        green_width = bb_green.xmax - bb_green.xmin

        blue_text = subplot_axes[1, 1].text(bb_green.x0, bb_green.y0,

                                            "X={:<2.1f}±{:<2.1f} nm\nY={:<2.1f}±{:<2.1f} nm".format(scatter_df[
                                                                                                        'blue_x'].mean(),
                                                                                                    scatter_df[
                                                                                                        'blue_x'].std(),
                                                                                                    scatter_df[
                                                                                                        'blue_y'].mean(),
                                                                                                    scatter_df[
                                                                                                        'blue_y'].std()),
                                            ha="left", va="top", ma="left", size=text_size, c='b')

        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 1].transData.inverted())
        blue_width = bb_blue.xmax - bb_blue.xmin

        if blue_width > green_width:
            shift = bb_blue.x1 - bb_green.x1
        else:
            shift = 0

        new_x0 = bb_blue.x0 - shift - text_height / 2
        blue_text.set_x(new_x0)
        green_text.set_x(new_x0)

        bb_blue = blue_text.get_window_extent(renderer=r)
        bb_blue = bb_blue.transformed(subplot_axes[1, 1].transData.inverted())
        bb_green = green_text.get_window_extent(renderer=r)
        bb_green = bb_green.transformed(subplot_axes[1, 1].transData.inverted())

        max_width = max(blue_width, green_width)

        title_shift = bb_title.x0 - text_height / 2 - bb_blue.x0
        title_text.set_x(bb_title.x1 - title_shift / 2)
        title_text.set_y(bb_title.y1 - text_height / 2)

        count_text = subplot_axes[1, 1].text(bb_blue.x1, bb_blue.y0,
                                             "n={}".format(len(scatter_df)), ha="right", va="top",
                                             ma="center",
                                             size=text_size)
        bb_count = count_text.get_window_extent(renderer=r)
        bb_count = bb_count.transformed(subplot_axes[1, 1].transData.inverted())

        count_shift = bb_count.x0 - bb_blue.x0
        count_text.set_x(bb_count.x1 - count_shift / 2)

        bbox_shift = text_height / 3
        bbox_xy = (bb_blue.x0 - text_height / 3, bb_count.y0)
        bbox_width = max_width + 2 * bbox_shift
        bbox_height = bb_title.y1 - bb_count.y0

        subplot_axes[1, 1].add_patch(
            Rectangle(bbox_xy, bbox_width, bbox_height, fc=(1, 1, 1), ec=(0, 0, 0), lw=0.5, alpha=0.5))

        subplot_axes[0, 0].set_aspect(1)
        subplot_axes[0, 1].set_aspect(1)
        subplot_axes[1, 0].set_aspect(1)
        subplot_axes[1, 1].set_aspect(1)
        plotLayout = QVBoxLayout()
        self.beads_maps = static_canvas
        plotLayout.addWidget(static_canvas)
        qfigWidget.setLayout(plotLayout)

        static_canvas.setMinimumSize(static_canvas.size())

        qScrollLayout.addWidget(qfigWidget)
        self.ui.scroll_beads_content.setLayout(qScrollLayout)

        self.ui.scroll_beads_content.show()
        self.ui.save_beads_vector_map.setEnabled(True)

    def save_beads_maps(self):
        save_path, save_type = QFileDialog.getSaveFileName(self, "Save File", "./beads vector maps",
                                                           'pdf (*.pdf);; png (*.png);;jpg (*.jpg)')
        print(save_path)
        if save_type == "pdf (*.pdf)":
            with PdfPages(save_path) as pdf:
                pdf.savefig(self.beads_maps.figure, dpi=120)
        else:
            self.beads_maps.figure.savefig(save_path)


# Main access
if __name__ == '__main__':
    app = QApplication(sys_argv)
    window = MainWindow()
    window.show()
    sys_exit(app.exec())
