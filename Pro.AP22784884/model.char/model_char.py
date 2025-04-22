#Qt________________________________________
from PyQt5 import QtGui, QtCore, QtWidgets#
from PyQt5.QtWidgets import* ##############
from PyQt5.QtCore import* #################
from PyQt5.QtCore import pyqtSignal########
from PyQt5.QtGui import* ##################
from PyQt5.QtGui import QResizeEvent ######
from PyQt5.QtGui import QIcon, QPixmap ####
from PyQt5.QtCore import QSize ############

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

#FIA LIBS__________________________________
from astropy.io import fits################
from shapely.geometry import LineString
from shapely import affinity
###########################################

#OPENCV LIBS_______________________________
import cv2, imutils########################
###########################################
#from skimage import io, color, transform
#ui-components files_______________________
###########################################

#MATPLOTLIBS_______________________________
import matplotlib.pyplot ##################
matplotlib.use('Qt5Agg')###################
from matplotlib.figure import Figure#######
from matplotlib.patches import Circle #####
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas #######
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
########################################################################################

import sys
import os

# Добавляем родительскую папку в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ArxSR import* 

class model_char(object):

#Initializer:
	def __init__(self,parent=None):
		super().__init__()
        
	#QMainWindow as parent:
		self.main_parent = parent

	#The sizes of MainWindow:
		self.win_w = parent.window_width
		self.win_h = parent.window_height

	#data holder:
		self.arx_data = None####
		self.data = None #######
		self.data_header = None#

	#ui_components:
		self.__left = Left(self)
		self.__right= Right(self)
#End of Initializer................


#SETTERS:
#Set MainWindow as parent for dialog window:
	def set_dialog_parent(self, parent):
		self.main_parent = parent	
#..........................................

#GETTERS:
#Get instances Right & Left:
	def getLeftWidget(self):
		return self.__left
#...........................

	def getRightLayout(self):
		return self.__right
#............................

	def getMainParent(self):
		return self.main_parent
#..............................


#METHODS:
#Load Img:
	def dataLoad(self):
		try:
		#get file path:		
			self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_parent , 'Single File', QtCore.QDir.rootPath() , '(*.fits *.fit)')
			
		#If user canceled dialog:
			if not self.fileName:
				print("Err: fits file has not been choosen")
			try:
				self.arx_data = ArxData(self.fileName)
			#save data from file as data:
				self.data = self.arx_data.get_data() ########
				self.data_header = self.arx_data.get_header()
				#############################################
				print(f"data header: {self.data_header}")
				print(f"fits data: {self.data}")
			except Exception as err:
				print(f"err in reading {self.fileName}")
				return
			
			#show png in the label not data:
			#change this point to plt shower
			#self.__left.imgLoad()
			
			#set data into QDataTable:
			self.__right.setData(ArxDataEditor(self.arx_data).get_ArxData_xy())

		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
			return
#...............................................................................

	def plotData(self):
		self.__left.Clear()

	

#End of model ###############################################



#CLass Left of model cropper:
class Left(QWidget):

#Initializer:
	def __init__(self, parent=None):
		super(Left, self).__init__(None)
		from matplotlib import rcParams
		# a figure instance to plot on
		self.figure = Figure()

		# this is the Canvas Widget that displays the `figure`
		# it takes the `figure` instance as a parameter to __init__
		self.canvas = FigureCanvas(self.figure)

		# this is the Navigation widget
		# it takes the Canvas widget and a parent
		self.toolbar = NavigationToolbar(self.canvas, self)
		
		# set the layout
		layout = QVBoxLayout()
		layout.setContentsMargins(0, 0, 0, 0)
		
		layout.addWidget(self.toolbar)
		layout.addWidget(self.canvas)
		self.setLayout(layout)		

		#Init it's parent - model cropper:
		self.model = parent
		
		#w,h - the size of main QLabel:
		self.w = self.model.win_w
		self.h = self.model.win_h
		print(f"win w = {self.w}")
		print(f"win h = {self.h}")
		###########################

#End of Init ..............................................................


#Clear canvas
	def Clear(self):
		self.figure.clf()


		ax = self.figure.add_subplot(111)
		ax.clear()

		ax.plot([1,2,3])
		self.canvas.draw()
		#................................
	
#Load image with cv2:
	def dataLoad(self):
		pass	
	#................................................

#End of Left ########################################################


#CLass Right of model cropper:
class Right(QVBoxLayout):

#Initializer:
	def __init__(self, parent=None):

	#Init Main Layout:
		super().__init__() ##################
		self.setAlignment(Qt.AlignTop) ######
		self.setContentsMargins(5, 5, 5, 5) #
		#####################################

	#model as parent:
		self.model = parent

		#w,h - the size of main QLabel:
		self.w = self.model.win_w
		self.h = self.model.win_h
		###########################

	#Browsing__________________________________________________
	#This is label to set Layout with widgets (text & btn):
		browse_label = QLabel() ###############################
		browse_label.setAlignment(Qt.AlignCenter) #############
		#перенос текста #######################################
		browse_label.setWordWrap(True) ########################
		browse_label.setObjectName("cropper_right_browse_label") 
	#Size of dark label #######################################
		browse_label.setFixedSize(self.w/5.3, self.h/16) ###################
		#######################################################

	#text above the button:
		browse_text = """
            <p style="text-align: justify;"><b>Upload fits format file</b></p>
            """
		browse_text_label = QLabel(browse_text, browse_label)
		#######################################################

	#Layout to hold text and button:
		layout = QVBoxLayout() #############################
		layout.setSpacing(7) ##############################
		layout.setContentsMargins(0, 0, 0, 0) ##############
		layout.setAlignment(Qt.AlignTop) ###################
		
	#label for the text:
		browse_text_label.setWordWrap(True) ################
		browse_text_label.setAlignment(Qt.AlignCenter) #####
		browse_text_label.setObjectName("cropper_browse_text")
	#Size of label for the text:
		browse_text_label.setFixedSize(self.w/6, self.h/44) ############
		####################################################

	#Button to browse:
		self.browse_button = QPushButton("Browse") ###########
		self.browse_button.setObjectName("cropper_browse_btn")
		self.browse_button.setEnabled(True) ##################
		self.browse_button.setFixedSize(self.w/12, self.h/38) #############
		######################################################

	#Set btn & text(lbl) into layout then set layout into browse label widget:
		browse_label.setLayout(layout) #######################################
		layout.addWidget(browse_text_label, 10, Qt.AlignHCenter) #############
		layout.addWidget(self.browse_button, 0, Qt.AlignHCenter) #############

	#Set browse label into main QVBoxLayout:
		self.addWidget(browse_label, alignment=Qt.AlignCenter)
		######################################################################

	#Init connections:
		self.browse_button.clicked.connect(self.model.dataLoad)

	
	#For tabs controller______________________________________________________
		#This is label to hold tab_layout:
		main_tabs_label = QLabel() ################################
		main_tabs_label.setAlignment(Qt.AlignCenter) ##############
		main_tabs_label.setWordWrap(True) #########################
		main_tabs_label.setObjectName("cropper_right_browse_label") 
		#main_tabs_label.setFixedSize(720, 1000) ##################
		main_tabs_label.setMinimumSize(self.w/5.3, self.h/2.2) ####
		main_tabs_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		###########################################################

		#Text of tabs controller
		tab_text = """
        <p style="text-align:justify;"><b>FITS Data Points</b></p>
            """

		#label for text of tabs controller
		tab_text_label = QLabel(tab_text)##################
		tab_text_label.setWordWrap(True) ##################
		tab_text_label.setAlignment(Qt.AlignCenter) #######
		tab_text_label.setObjectName("cropper_browse_text")
		#Size of label for the text:
		#tab_text_label.setFixedSize(600, 70) ##############
		tab_text_label.setMinimumSize(self.w/6, self.h/44)
		tab_text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		###################################################

		#Layout to hold text and tabs:
		tab_layout = QVBoxLayout() #############################
		#tab_layout.setSpacing(10) ##############################
		tab_layout.setContentsMargins(0, 0, 0, 0) ##############
		tab_layout.setAlignment(Qt.AlignTop) ###################

		#Main Tab:
		tabs = QTabWidget() ################
		tabs.setMinimumSize(self.w/5.3, self.h/2.7)#2.4)
		#tabs.setMinimumSize(self.w/6.5, self.h/3)
		tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

		#Number-1 tab page:
		tab_1 = QWidget() ##################
		tab_layout_data = QVBoxLayout() ##
		tab_layout_data.setAlignment(Qt.AlignTop)
		tab_1.setLayout(tab_layout_data) #
		tab_layout_data.setSpacing(1)  # Убирает вертикальные отступы между виджетами
		tab_layout_data.setContentsMargins(2, 5, 30, 20)  # Убирает внешние отступы

		#Number-2 tab page:
		tab_2 = QWidget()  #################
		tab_layout_peaks = QVBoxLayout() ####
		tab_2.setLayout(tab_layout_peaks) ###
		tab_layout_peaks.setSpacing(1)
		#tab_layout_crop.setContentsMargins(10, 0, 10, 0)  # Убирает внешние отступы

		tabs.addTab(tab_1, "Data") ###################################
		tabs.addTab(tab_2, "Peaks") #####################################
		tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) 
		
		################################################################
	
		#Install text & tabs into layout then set layout into main label:
		tab_layout.addWidget(tab_text_label, 10, Qt.AlignHCenter) #######
		tab_layout.addWidget(tabs, alignment=Qt.AlignLeft) ##############
		#################################################################
		main_tabs_label.setLayout(tab_layout) ###########################
		#Set main_tabs_label into Right:
		self.addWidget(main_tabs_label, alignment=Qt.AlignCenter)

		#stretch_box = QToolBox()
		#stretch_box.addItem(main_tabs_label, "Data")
		#self.addWidget(stretch_box, alignment=Qt.AlignCenter)

		#data Table:
		self.dataTable = QTableWidget()
		self.dataTable.setObjectName("dataTable")
		self.dataTable.setColumnCount(2)
		self.dataTable.setHorizontalHeaderLabels([" Х "," Y "])
		self.dataTable.setShowGrid(True)
		self.dataTable.horizontalHeader().setStretchLastSection(True)
		self.dataTable.setAlternatingRowColors(False)
		self.dataTable.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
		self.dataTable.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)
		self.dataTable.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
		
		#set data as a next row:
		self.setEmptyData(self.dataTable)
		#Set Widgets into Tabs:
		tab_layout_data.addWidget(self.dataTable) ###

		#Button to plot data:
		self.data_plot_button = QPushButton("PLOT") ###########
		self.data_plot_button.setObjectName("cropper_browse_btn")
		self.data_plot_button.setEnabled(True) ##################
		self.data_plot_button.setFixedSize(self.w/16, self.h/38) #############
		self.data_plot_button.clicked.connect(self.model.plotData)
		tab_layout_data.addWidget(self.data_plot_button)

		#peaks Table:
		self.peakTable = QTableWidget()
		self.peakTable.setObjectName("dataTable")
		self.peakTable.setColumnCount(2)
		self.peakTable.setHorizontalHeaderLabels([" Х "," Y "])
		self.peakTable.setShowGrid(True)
		self.peakTable.horizontalHeader().setStretchLastSection(True)
		self.peakTable.setAlternatingRowColors(False)
		self.peakTable.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
		self.peakTable.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)
		self.peakTable.verticalHeader().setDefaultAlignment(Qt.AlignCenter)
		
		#set data as a next row:
		self.setEmptyData(self.peakTable)
		#Set Widgets into Tabs:
		tab_layout_peaks.addWidget(self.peakTable) ###



	#End of Main Tab###########################################################



	#End of Browsing__________________________________________________________

	def setEmptyData(self, datatable):
		for _ in range(10):
			row_position = datatable.rowCount()
			datatable.insertRow(row_position)
			datatable.setItem(row_position, 0, QTableWidgetItem(""))
			datatable.setItem(row_position, 1, QTableWidgetItem(""))

	def resetData(self):
		self.setEmptyData(self.dataTable)
		self.setEmptyData(self.peakTable)
	#End of resetData____________________


#	def setData(self, data):
#    
#		#if not isinstance(data, np.ndarray) or data.ndim != 1 or data.size != 2:
#		#	raise ValueError("The array should be numpy's object...")
#		
#			# Очищаем таблицу
#		self.dataTable.setRowCount(0)
#
#		# Вставляем одну строку
#		self.dataTable.insertRow(0)
#		for col in range(2):
#			item = QTableWidgetItem(str(data[col]))
#			item.setTextAlignment(Qt.AlignCenter)
#			self.dataTable.setItem(0, col, item)

	def setData(self, data):
		# Очистить таблицу
		self.dataTable.setRowCount(0)

		# Проверка: data должен быть 2D numpy-массивом с 2 колонками
		if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
			raise ValueError("waiting for 2D array (N, 2)")

		# Заполнение таблицы
		for i, (x, y) in enumerate(data):
			self.dataTable.insertRow(i)

			x_item = QTableWidgetItem(str(x))
			x_item.setTextAlignment(Qt.AlignCenter)
			self.dataTable.setItem(i, 0, x_item)

			y_item = QTableWidgetItem(str(y))
			y_item.setTextAlignment(Qt.AlignCenter)
			self.dataTable.setItem(i, 1, y_item)


#End of Initializer..........................
	
#End of Right ########################################################





