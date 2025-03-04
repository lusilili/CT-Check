import random
import sys
from PySide6 import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
# from charts.barchart import Barchart
# from charts.linechart import Linechart
# from charts.indicator import Indicator


class Ct_output(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName('Ct_output')
        self.output_group = QGroupBox()
        self.layout_QH = QGridLayout()
        self.edit=QPlainTextEdit()
        self.edit.setReadOnly(True)
        self.layout_QH.addWidget(self.edit)

        self.output_group.setLayout(self.layout_QH)