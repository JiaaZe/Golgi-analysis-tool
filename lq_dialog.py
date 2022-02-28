from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from ui.lq_dialog import Ui_dialog
from numpy import (arange as np_arange, array as np_array, round as np_round)


class LqDialog(QDialog):
    def __init__(self, lq_list):
        super().__init__()
        self.ui = Ui_dialog()
        self.ui.setupUi(self)

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

        static_canvas = FigureCanvas(Figure(figsize=(10, 10)))
        ax = static_canvas.figure.subplots()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlim(x_tick_min, x_tick_max)
        ax.set_xticks(x_ticks_label)

        count_arr, data, _ = ax.hist(lq, bins=len(x_ticks), color='gray', rwidth=0.8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        y = count_arr.max()
        ax.set_ylim(0, y)
        ax.set_title("{}±{}(n={})".format(mean, std, len(lq)), verticalalignment='bottom', horizontalalignment='center')

        plotLayout = QVBoxLayout()
        plotLayout.addWidget(static_canvas)
        self.ui.frame.setLayout(plotLayout)
