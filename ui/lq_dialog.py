# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'lq_dialog.ui'
##
## Created by: Qt User Interface Compiler version 6.2.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
                          QMetaObject, QObject, QPoint, QRect,
                          QSize, QTime, QUrl, Qt)
from PyQt5.QtWidgets import (QApplication, QDialog, QFrame, QGridLayout,
                             QSizePolicy, QWidget)


class Ui_dialog(object):
    def setupUi(self, dialog):
        if not dialog.objectName():
            dialog.setObjectName(u"dialog")
        dialog.resize(400, 400)
        dialog.setMinimumSize(QSize(400, 400))
        self.gridLayout = QGridLayout(dialog)
        self.gridLayout.setObjectName(u"gridLayout")
        self.frame = QFrame(dialog)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)

        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)

        self.retranslateUi(dialog)

        QMetaObject.connectSlotsByName(dialog)

    # setupUi

    def retranslateUi(self, dialog):
        dialog.setWindowTitle(QCoreApplication.translate("dialog", u"LQ histogram", None))
    # retranslateUi
