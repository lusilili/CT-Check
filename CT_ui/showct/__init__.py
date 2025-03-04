import random
import sys
from PySide6 import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
# from charts.barchart import Barchart
# from charts.linechart import Linechart
# from charts.indicator import Indicator


class Ctwid(QWidget):
    def __init__(self,pic_path):
        super().__init__()
        self.setObjectName('Ctwid')
        self.layout_QV_group = QGroupBox()
        self.layout_QH = QGridLayout()
        self.pic_area = QLabel()
        # self.pic_area.setText('hello')
        pic = QPixmap(pic_path).scaled(self.pic_area.size(),
                                                                                          aspectMode=Qt.KeepAspectRatio)  # 图片自适应
        self.pic_area.setPixmap(pic)
        self.layout_QH.addWidget(self.pic_area)

        self.layout_QV_group.setLayout(self.layout_QH)