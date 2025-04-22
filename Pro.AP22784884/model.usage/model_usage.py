#Qt________________________________________
from PyQt5 import QtGui, QtCore, QtWidgets#
from PyQt5.QtWidgets import* ##############
from PyQt5.QtCore import* #################
from PyQt5.QtGui import* ##################

#OTHERS____________________________________
from pathlib import Path ##################
from PIL import Image #####################
from enum import Enum #####################
import qrc_resources ######################
from math import* #########################
import numpy as np#########################
import sys#################################
import os #################################
###########################################
#ui-components files:



class model_usage(QObject):

#Initializer:
	def __init__(self):
		super().__init__()
        
		#ui_components:
		self.__left = Left()
		self.__right= Right()


#Get instances Right & Left:
	def getLeftWidget(self):
		return self.__left
		
	def getRightLayout(self):
		return self.__right
	#........................




    



#Document presenter.....................................................................
class Left(QWidget):

#Initializer:
    def __init__(self):
        super().__init__()
        
    #main widget of presenter:
        widget = QGroupBox() ###############
        widget.setObjectName('gbox_usage_left') #
        ####################################

    #main layout of main widget of presenter:
        layout = QHBoxLayout() #############
        layout.addWidget(widget) ###########
        self.setLayout(layout) #############
        self.setContentsMargins(0, 0, 0, 0)
    #End of initializer....................




#to hide from user:
    def hide(self):
        self.setVisible(False)

#to show for user:
    def show(self):
        self.setVisible(True)
#End of Left presenter..................................................................




class Right(QVBoxLayout):

#Initializer:
	def __init__(self):
		super().__init__()

		self.setAlignment(Qt.AlignTop)
		self.setContentsMargins(20, 60, 20, 20)
		self.setSpacing(10)

		message_label = QLabel()

		message_label.setText(
            """
            <p style="text-align: center;"><b>Welcome to FAI-tools!</b></p>
            <p style="text-align: justify;">This software was developed at the</p>
            <p style="text-align: justify;"><b>FESENKOV ASTROPHYSICAL INSTITUTE</b></p>
            <p style="text-align: justify;">
                as part of the project funded by the Science Committee of the Ministry
                of Science and Higher Education of the Republic of Kazakhstan
                (Grant No. AP14869876).
            </p>
            <p style="text-align: justify;">Please review the</p>
            <p style="text-align: justify;">
                <a href='https://fai.kz/ru/about' style="color: #6988E7; text-decoration: none;"
                   onmouseover="this.style.color='#6988E7';"
                   onmouseout="this.style.color='#6988E7';">
                   https://fai.kz
                </a>.
            </p>
            """
        )

		message_label.setAlignment(Qt.AlignCenter)
		message_label.setWordWrap(True)
		message_label.setOpenExternalLinks(True)
		message_label.setObjectName("usage_right_label")
		message_label.setFixedSize(700, 800) 

		agreement_checkbox = QCheckBox("I accept the agreement.")
		agreement_checkbox.setObjectName("usage_right_checkbox")

		accept_button = QPushButton("Continue")
		accept_button.setObjectName("right_btn_accept")
		accept_button.setEnabled(False)
		accept_button.setFixedSize(240, 70)
		# Связываем чекбокс с активацией кнопки
		agreement_checkbox.stateChanged.connect(
            lambda state: accept_button.setEnabled(state == Qt.Checked)
        )

		self.addWidget(message_label, alignment=Qt.AlignCenter)
		self.addWidget(agreement_checkbox, alignment=Qt.AlignBottom)
		self.addWidget(accept_button, alignment=Qt.AlignBottom)






    
