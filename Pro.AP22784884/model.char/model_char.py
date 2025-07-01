#Qt________________________________________
from asyncio.format_helpers import extract_stack
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
from mp_viewer_app import*

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
		self.data_header = None
		self.data_editor = None
		self.polynom = None

		self.exp_time = 0
		self.max_fname = "None"
		self.date = None
	#peaks holder:
		self.current_Peaks = None
		self.listOfPeaks = {}
		self.listofExTimes = {}

	#ui_components:
		self.__left = Left(self)
		self.__right= Right(self)

	#message box:
		self.msg = QMessageBox()
		self.msg.setWindowTitle("Error:")
		self.msg.setStandardButtons(QMessageBox.Ok)

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
#fits data Load:
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

				self.data_editor = ArxDataEditor(self.arx_data)
				#############################################
				print(f"data header: {self.data_header}")
				print(f"fits data: {self.data}")
			except Exception as err:
				print(f"err in reading {self.fileName}")
				return
			
			#set data into QDataTable:
			self.__right.setData(ArxDataEditor(self.arx_data).get_ArxData_xy())

		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
			return
#...............................................................................

#PLot data figure:
	def plotData(self):
		self.__left.dataPlot(self.data_editor.get_ArxData_xy())
#..............................................................

	def setPolynom(self, polynom):
		self.polynom = polynom
#.................................


#PLot Peak figure:
	def plotPeak(self):
		if self.arx_data is None:
			self.msg.setText("No Data to plot Peaks!!!")
			self.msg.exec_()
			return
		else:
			try:
			#if checkBox is not trigged then use order value:
				if self.__right.peak_order_spinbox.isEnabled():   
				#get value from spinBox to use as order:
					order = self.__right.peak_order_spinbox.value()
				# HERE CODE BREAKS:*************************************************
					peak_x, peak_y = ArxCollibEditor(self.arx_data).get_peaks(order)
				# ******************************************************************
				else:
				#if checkBox is trigged then use None as order:
					peak_x, peak_y = ArxCollibEditor(self.arx_data).get_peaks( None)

			#convert peaks & set to peakTable in right:
				peaks = np.column_stack((peak_x, peak_y))#
				self.__right.setPeak(peaks)###############
				self.current_Peaks = peaks ###############

			#set peaks into left onto plot:
				self.__left.peakPlot(peaks)
			#............................
			except Exception as err:
				print(err)
#...................................................................................

#Clear plot data:
	def clearPlotData(self):
		self.__left.Clear()
#...........................

#Clear plot Peaks:
	def clearPlotPeaks(self):
		pass
		#self.__left.Clear()
#..........................

#send the peak to list:
	def sendToList(self):
		#create List of peaks holder in the init:
	#Message Box for Users:
		msg = QMessageBox()
		msg.setWindowTitle("Message:")
		msg.setStandardButtons(QMessageBox.Ok)

		PeakfName = self.arx_data.get_fname()
		ExposeTime = self.arx_data.get_exptime() #CHECK
		date = self.arx_data.get_dateObs() #CHECK

		#CHECK
		if ExposeTime is None:  
			print("None exp-time")
			return

		#CHECK
		if date is None:
			print("None date time")
			return

		# Проверяем listOfPeaks на None и создаём словарь если нужно
		if self.listOfPeaks is None:
			self.listOfPeaks = {}

		# Проверяем, что current_Peaks не None, иначе смысла нет
		if self.current_Peaks is not None:
			self.setMaxExpTime(ExposeTime, PeakfName, date)
			# Добавляем в listOfPeaks
			#MAKE IT RETURNABLE & ADD PARAMS exptime and date:
			

			self.listOfPeaks[PeakfName] = self.current_Peaks
			self.__right.updateComboListOfPeaks(self.listOfPeaks)
			msg.setText(f"The peaks PeakfName has been sent into Peaks List!!!")
			
		else:
			msg.setText(f"Warning: current_Peaks is None, nothing to add.")

		msg.exec_()
		
	#............................................................
	

	def setMaxExpTime(self,exp_time, fname, date):
		if self.exp_time < exp_time:
			self.exp_time = exp_time
			self.max_fname = fname
			self.date = date
	#......................................

	def runPolyCreator(self):
		try:

			viewer = CalibPolyDlg()#mp_viewer()
			viewer.setDialogParent(self.main_parent)
			viewer.setModule(self)
			#viewer.setAllPeaks(self.listOfPeaks,self.listofExTimes)
			viewer.setAllPeaks(self.listOfPeaks,self.exp_time, self.max_fname, self.date)
			print("ssss-"+self.max_fname)
			viewer.setModeValue(0)
			viewer.show()
			viewer.exec_()
			
			# Потом:
			#saved = viewer.saved_poly_coeffs
			#print(saved)
		except Exception as err:
			print(err)


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
		self.peaks_plot = None 
		
		#w,h - the size of main QLabel:
		self.w = self.model.win_w
		self.h = self.model.win_h
		print(f"win w = {self.w}")
		print(f"win h = {self.h}")
		###########################
#End of Initializer..............................................................


#Clear canvas
	def Clear(self):
		self.figure.clf()          
		self.canvas.draw() 
		self.peaks_plot = None 
		#self.canvas.draw_idle()
		#................................
	
#Load image with cv2:
	def dataPlot(self,data):

		self.figure.clf()
		ax = self.figure.add_subplot(111)
		ax.clear()

		x = data[:, 0]
		y = data[:, 1]

		ax.plot(x, y)
		self.canvas.draw()
		#self.canvas.draw_idle()
	#................................................

#Plot peak on the left:
	def peakPlot(self,peaks):
		#Не работает!!!!!!!!!!!!!!!!!!!!!!!!!!!
		import numpy as np
		peaks = np.asarray(peaks, dtype=float)      # гарантируем float
		if peaks.size == 0:                         # нечего рисовать
			print("peakPlot: пустой массив.")
			return

		# 1. Берём текущие оси (создаём при необходимости)
		ax = self.figure.gca() if self.figure.axes else self.figure.add_subplot(111)

		# 2. Удаляем старый scatter, если он был
		if getattr(self, "peaks_plot", None) is not None:
			try:
				self.peaks_plot.remove()
			except ValueError:
				pass  # ось могла быть уже стёрта
			self.peaks_plot = None
			
		# 3. Рисуем новые маркеры
		self.peaks_plot = ax.scatter(
		    peaks[:, 0],          # X
		    peaks[:, 1],          # Y
		    s=60,                 # размер маркера; сделайте больше, если не видно
		    c="red",              # цвет
		    marker="o",
		    linewidths=0.8,
		    edgecolors="black",
		    zorder=5,
		    label="Peaks"
		)

		# 4. Освежаем масштаб осей (чтобы точки точно попали в кадр)
		ax.relim()               # пересчитать лимиты по данным
		ax.autoscale_view()      # применить лимиты

		# 5. Перерисовываем только когда Qt сочтёт нужным
		self.canvas.draw()


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


	#Init right ui components:
		self.InitUi()

	#Init Connections:
		self.InitConnects()

#End of Initializ............................


	def InitUi(self):

	#    %%%%%%%%%%%%%     BROWSING     %%%%%%%%%%%%%%%%%%%%%%%%:	
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#This is label to set Layout with widgets (text & btn):
		browse_label = QLabel() ################################
		browse_label.setAlignment(Qt.AlignCenter) ##############
		#перенос текста ########################################
		browse_label.setWordWrap(True) #########################
		browse_label.setObjectName("cropper_right_browse_label") 
	#Size of dark label ########################################
		#MAC CHANGE
		#browse_label.setFixedSize(self.w/5.3, self.h/16)
		browse_label.setFixedSize(int(self.w/5.3), int(self.h/16)) #######
		########################################################

	#Mian Big Text on the Label:
		browse_text = """
            <p style="text-align: justify;"><b>
			Upload fits format file</b></p>
            """
		browse_text_label = QLabel(browse_text, browse_label)
		#####################################################

	#Layout to hold Main Big text and button:
		layout = QVBoxLayout() ##############
		layout.setSpacing(7) ################
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop) ####
		
	#label for the text:
		browse_text_label.setWordWrap(True) ##################
		browse_text_label.setAlignment(Qt.AlignCenter) #######
		browse_text_label.setObjectName("cropper_browse_text")
		#MAC CHANGE
		#browse_text_label.setFixedSize(self.w/6, self.h/44)	
		browse_text_label.setFixedSize(int(self.w/6), int(self.h/44))###
		######################################################

	#Button to browse:
		self.browse_button = QPushButton("Browse") ###########
		self.browse_button.setObjectName("cropper_browse_btn")
		self.browse_button.setEnabled(True) ##################
		#MAC CHANGE
		#self.browse_button.setFixedSize(self.w/12, self.h/38)
		self.browse_button.setFixedSize(int(self.w/12), int(self.h/38))#

	#Set btn&text(lbl) into layout then set it into browse QLbl:
		browse_label.setLayout(layout) #########################
		layout.addWidget(browse_text_label, 10, Qt.AlignHCenter) 
		layout.addWidget(self.browse_button, 0, Qt.AlignHCenter) 

	#Set browse label into Main QVBoxLayout:
		self.addWidget(browse_label, alignment=Qt.AlignCenter)
		########################################################
	#   %%%%%%%%%%%%%%   END OF BROWSING   %%%%%%%%%%%%%%%%%%%%%

	#    %%%%%%%%%%%%%     TABS AREA ()     %%%%%%%%%%%%%%%%%%%%%%%	
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#QLabel for Tabs (Data, Peaks and its components):
		main_tabs_label = QLabel() ################################
		main_tabs_label.setAlignment(Qt.AlignCenter) ##############
		main_tabs_label.setWordWrap(True) #########################
		main_tabs_label.setObjectName("cropper_right_browse_label") 
		#MAC CHANGE
		#main_tabs_label.setMinimumSize(self.w/5.3, self.h/2.2)
		main_tabs_label.setMinimumSize(int(self.w/5.3), int(self.h/2.2)) ####
		main_tabs_label.setSizePolicy(QSizePolicy.Expanding, 
								QSizePolicy.Expanding) ############
		###########################################################

	#Main Big Text of tabs Label
		tab_text = """
        <p style="text-align:justify;"><b>FITS Data Points</b></p>
            """
		##########################################################

	#QLabel to hold Main Big Text:
		tab_text_label = QLabel(tab_text)##################
		tab_text_label.setWordWrap(True) ##################
		tab_text_label.setAlignment(Qt.AlignCenter) #######
		tab_text_label.setObjectName("cropper_browse_text")
		#MAC CHANGE
		#tab_text_label.setMinimumSize(self.w/6 , self.h/44)
		tab_text_label.setMinimumSize(int(self.w/6) , int(self.h/44))
		tab_text_label.setSizePolicy(QSizePolicy.Expanding, 
							   QSizePolicy.Fixed) #########
		###################################################

	#Layout to hold text and tabs:
		tab_layout = QVBoxLayout() #################
		#tab_layout.setSpacing(10) #################
		tab_layout.setContentsMargins(0, 0, 0, 0) ##
		tab_layout.setAlignment(Qt.AlignTop) #######

	#Main Tab:
		tabs = QTabWidget() ######################## 
		#MAC CHANGE
		#tabs.setMinimumSize(self.w/5.3, self.h/2.45)
		tabs.setMinimumSize(int(self.w/5.3), int(self.h/2.45))
		tabs.setSizePolicy(QSizePolicy.Expanding, 
					 QSizePolicy.Fixed) ############
		############################################

	#Number-1 tab page (Data Table):
		tab_1 = QWidget() ##########################
		tab_layout_data = QVBoxLayout() ############
		tab_layout_data.setAlignment(Qt.AlignTop)###
		tab_1.setLayout(tab_layout_data) ###########
		tab_layout_data.setSpacing(15) #############
		tab_layout_data.setContentsMargins(2,5,30,5) 
		############################################

	#Number-2 tab page (Peak Table):
		tab_2 = QWidget()  ###############
		tab_layout_peaks = QVBoxLayout() #
		tab_2.setLayout(tab_layout_peaks)#
		tab_layout_peaks.setSpacing(15)###
		##################################

	#Set data-tab and peak-tab into main tabs:
		tabs.addTab(tab_1, "Data") ##############
		tabs.addTab(tab_2, "Peaks") #############
		tabs.setSizePolicy(QSizePolicy.Expanding,
					QSizePolicy.Fixed) ##########
		#########################################
		
	#Install txt & tabs into layout then set layout into main lbl:
		tab_layout.addWidget(tab_text_label, 10, Qt.AlignHCenter)#
		tab_layout.addWidget(tabs, alignment=Qt.AlignLeft) #######
		##########################################################
		main_tabs_label.setLayout(tab_layout) ####################
		#Set main_tabs_label into Right: #########################
		self.addWidget(main_tabs_label, alignment=Qt.AlignCenter)#
		##########################################################

	#Data Table:
		self.dataTable = QTableWidget() ########################################
		self.dataTable.setObjectName("dataTable") ##############################
		self.dataTable.setColumnCount(2) #######################################
		self.dataTable.setHorizontalHeaderLabels([" Х "," Y "]) ################
		self.dataTable.setShowGrid(True) #######################################
		self.dataTable.horizontalHeader().setStretchLastSection(True) ##########
		self.dataTable.setAlternatingRowColors(False) ##########################
		self.dataTable.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
		self.dataTable.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)
		self.dataTable.verticalHeader().setDefaultAlignment(Qt.AlignCenter) ####
		########################################################################
		
	#Reset data in self.dataTable:
		self.setEmptyData(self.dataTable)

	#Set dataTable into data-tab layout:
		tab_layout_data.addWidget(self.dataTable) 

	#Horizontal Layout of plot data & cancel buttons:
		data_buttons_layout = QHBoxLayout() ##############
		data_buttons_layout.setContentsMargins(0, 0, 0, 0)  
		#minimum distance between buttons ################
		data_buttons_layout.setSpacing(15) ###############
		##################################################

	#Plot data button in horizontal buttons Layout: 
		self.data_plot_button = QPushButton("") ##########################
		self.data_plot_button.setObjectName("transround") ################
		self.data_plot_button.setEnabled(True) ###########################
		#MAC CHANGE
		# #self.data_plot_button.setFixedSize(self.w/60, self.h/40)
		self.data_plot_button.setFixedSize(int(self.w/60), int(self.h/40)) 
		##################################################################
		
	#Set icon for the data plot button:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "plot_data.png") #
		self.data_plot_button.setIcon(QIcon(icon_path)) #######
		self.data_plot_button.setIconSize(QSize(300, 65)) #####
		data_buttons_layout.addWidget(self.data_plot_button) ##
		#######################################################
		
	#Cancel data button in horizontal buttons Layout:
		self.cancel_plot_button = QPushButton("") ##########################
		self.cancel_plot_button.setObjectName("transround") ################
		self.cancel_plot_button.setEnabled(True) ###########################
		self.cancel_plot_button.setFixedSize(int(self.w/60),int(self.h/40))#  
		
	#Set icon for the cancel data plot button:
		icon_path = os.path.join(script_dir, "cancel_plot.png")
		self.cancel_plot_button.setIcon(QIcon(icon_path)) #####
		self.cancel_plot_button.setIconSize(QSize(300, 65)) ###
		data_buttons_layout.addWidget(self.cancel_plot_button)#
		#######################################################
		
	#Widget to hold data plot&cancel buttons and add it into tab data layout:
		container_widget = QWidget() #######################################
		container_widget.setLayout(data_buttons_layout)#####################
		tab_layout_data.addWidget(container_widget, alignment=Qt.AlignLeft)#
		####################################################################

		#######################    PEAK BLOCK    ###########################

	#Peaks Table:
		self.peakTable = QTableWidget() ########################################
		self.peakTable.setObjectName("dataTable") ##############################
		self.peakTable.setColumnCount(2) #######################################
		self.peakTable.setHorizontalHeaderLabels([" Х "," Y "]) ################
		self.peakTable.setShowGrid(True) #######################################
		self.peakTable.horizontalHeader().setStretchLastSection(True) ##########
		self.peakTable.setAlternatingRowColors(False) ##########################
		self.peakTable.horizontalHeaderItem(0).setTextAlignment(Qt.AlignHCenter)
		self.peakTable.horizontalHeaderItem(1).setTextAlignment(Qt.AlignHCenter)
		self.peakTable.verticalHeader().setDefaultAlignment(Qt.AlignCenter) ####
		########################################################################
		
	#Clear Peak Table:
		self.setEmptyData(self.peakTable)
	#Set Peak Table into Tabs:
		tab_layout_peaks.addWidget(self.peakTable) 

	#Horizontal Layout of Peak tab widgets:
		peak_widgets_layout = QHBoxLayout()############
		peak_widgets_layout.setContentsMargins(0,0,0,0) 
		peak_widgets_layout.setSpacing(15)#############
		###############################################

	#Label for ward "Order":
		peak_undertext_label = QLabel("Order:") #########################
		peak_undertext_label.setAlignment(Qt.AlignLeft) #################
		peak_undertext_label.setWordWrap(True) ##########################
		peak_undertext_label.setObjectName("cropper_right_browse_label")# 
		peak_undertext_label.setFixedSize(int(self.w/30), int(self.h/30))
		peak_widgets_layout.addWidget(peak_undertext_label)##############
		#################################################################

	#Order spin box:
		self.peak_order_spinbox = QSpinBox()##################
		self.peak_order_spinbox.setMinimum(1)#################       
		self.peak_order_spinbox.setMaximum(21)################     
		self.peak_order_spinbox.setValue(2)###################        
		self.peak_order_spinbox.setSingleStep(1)##############  
		peak_widgets_layout.addWidget(self.peak_order_spinbox)
		######################################################

	#Label for ward "Auto":
		order_undertext_label = QLabel("  Auto:") #######################
		order_undertext_label.setAlignment(Qt.AlignLeft) ################
		order_undertext_label.setWordWrap(True) #########################
		order_undertext_label.setObjectName("cropper_right_browse_label") 
		order_undertext_label.setFixedSize(int(self.w/30),int(self.h/30)) 
		peak_widgets_layout.addWidget(order_undertext_label)#############
		#################################################################

	#CheckBox Auto mode for order:
		self.auto_checkBox = QCheckBox("  ") ############
		peak_widgets_layout.addWidget(self.auto_checkBox)
		#################################################

	#Plot Peaks Button:
		self.peak_plot_button = QPushButton("") ##########################
		self.peak_plot_button.setObjectName("transround") ################
		self.peak_plot_button.setEnabled(True) ###########################
		self.peak_plot_button.setFixedSize(int(self.w/60), int(self.h/40)) 
		##################################################################
		
	#Icons for Button (plot peaks):
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "plot_data.png") #
		self.peak_plot_button.setIcon(QIcon(icon_path)) #######
		self.peak_plot_button.setIconSize(QSize(300, 65)) #####
	#Set button into layout: ##################################
		peak_widgets_layout.addWidget(self.peak_plot_button) ##
		#######################################################
		
	#Send Peaks Button into list:
		self.peak_tolist_button = QPushButton("") #########################
		self.peak_tolist_button.setObjectName("transround")################
		self.peak_tolist_button.setEnabled(True)###########################
		self.peak_tolist_button.setFixedSize(int(self.w/60),int(self.h/40))  
		###################################################################
		
	#Set icon for Button (Peaks into list)
		icon_path = os.path.join(script_dir, "peak_tolist.png")#
		self.peak_tolist_button.setIcon(QIcon(icon_path)) ######
		self.peak_tolist_button.setIconSize(QSize(300, 65))#####
		peak_widgets_layout.addWidget(self.peak_tolist_button)##
		
	#Set horizontal layout into container:
		container_peakwidget = QWidget() ##################
		container_peakwidget.setLayout(peak_widgets_layout)
		###################################################

	#Set Container into peak tab layout:
		tab_layout_peaks.addWidget(container_peakwidget, alignment=Qt.AlignLeft)
	#   %%%%%%%%%%%%%%%%%%%%   END OF TAB AREA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	#   %%%%%%%%%%%%%%%%%%%%   CREATE CHARACTERISTICS & LIST OF PEAKS AREA %%%%%

	#Label for Characteristic widgets:
		Character_label = QLabel() #################################
		Character_label.setAlignment(Qt.AlignCenter) ###############
		#перенос текста ############################################
		Character_label.setWordWrap(True) ##########################
		Character_label.setObjectName("cropper_right_browse_label")# 
		Character_label.setFixedSize(int(self.w/5.3),int(self.h/10)) 
		############################################################

	#Text for Characteristic Label:
		Character_text = """
            <p style="text-align: justify;"><b>List Of Peaks</b></p>
            """
		##############################################################

	#Verticale Layout to hold text list of peaks and 2 buttons:
		layout = QVBoxLayout() ############
		layout.setSpacing(15) #############
		layout.setContentsMargins(0,0,0,0)#
		layout.setAlignment(Qt.AlignTop)###

	#QLabel for the text of Characteristic Label:
		Character_text_label = QLabel(Character_text, Character_label)####
		Character_text_label.setWordWrap(True) ###########################
		Character_text_label.setAlignment(Qt.AlignCenter) ################
		Character_text_label.setObjectName("cropper_browse_text")#########
		Character_text_label.setFixedSize(int(self.w/6), int(self.h/44))##
		##################################################################

	#List of Peaks holder - QComboBox:
		self.combolistOfPeaks = QComboBox()
		###################################

	#Horizontal layout for buttons
	#"Delete Peaks" & "Create Polynom":
		char_buttons_layout = QHBoxLayout()
		###################################

	#Button to delete Peaks:
		self.Character_Delete_button = QPushButton("Delete Peaks") #############
		self.Character_Delete_button.setObjectName("cropper_browse_btn") #######
		self.Character_Delete_button.setEnabled(True) ##########################
		self.Character_Delete_button.setFixedSize(int(self.w/14),int(self.h/38)) 
		########################################################################

	#Button to Create Polynom:
		self.Character_Create_button = QPushButton("Create Polynom") ###########
		self.Character_Create_button.setObjectName("cropper_browse_btn") #######
		self.Character_Create_button.setEnabled(True) ##########################
		self.Character_Create_button.setFixedSize(int(self.w/14),int(self.h/38)) 
		########################################################################

    #Set Buttons into Horizontal layout:
		char_buttons_layout.addWidget(self.Character_Delete_button,10,Qt.AlignHCenter)
		char_buttons_layout.addWidget(self.Character_Create_button,10,Qt.AlignHCenter)
		##############################################################################

	#Set text(lbl), comboBox and buttons hor layout into ver layout
	#then set layout into browse label widget:
		layout.addWidget(Character_text_label, 10, Qt.AlignHCenter)#
		layout.addWidget(self.combolistOfPeaks,  stretch=1) ########
		layout.addLayout(char_buttons_layout,0) ####################
		Character_label.setLayout(layout) ##########################

	#Set browse label into main QVBoxLayout:
		self.addWidget(Character_label, alignment=Qt.AlignCenter)###
		############################################################
	#   %%%%%%%%%%%%%%%%%%%%   END OF CHARACTERISTICS AREA  %%%%%%%%%%%%%%%%%%%%%%%%%%


	#   %%%%%%%%%%%%%%%%%%%%   Watch Polynom AREA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#Big label to hold Layout with widgets (text & btn):
		watch_label = QLabel() #################################
		watch_label.setAlignment(Qt.AlignCenter) ###############
		watch_label.setWordWrap(True) ##########################
		watch_label.setObjectName("cropper_right_browse_label")# 
		watch_label.setFixedSize(int(self.w/5.3),int(self.h/16))
		########################################################

	#Text of Big label for Watch layer:
		watch_text = """
            <p style="text-align: justify;"><b>Watch Obtained Polynom</b></p>
            """
		#####################################################################

	#Vertical Layout to hold text and watch button:
		layout = QVBoxLayout() ####################
		layout.setSpacing(7) ######################
		layout.setContentsMargins(0,0,0,0) ########
		layout.setAlignment(Qt.AlignTop) ##########
		
	#Label for the watch text:
		watch_text_label = QLabel(watch_text, watch_label)#########
		watch_text_label.setWordWrap(True) ########################
		watch_text_label.setAlignment(Qt.AlignCenter) #############
		watch_text_label.setObjectName("cropper_browse_text")######
		watch_text_label.setFixedSize(int(self.w/6),int(self.h/44)) 
		###########################################################

	#Button to browse:
		self.watch_button = QPushButton("Watch") ####################
		self.watch_button.setObjectName("cropper_browse_btn")########
		self.watch_button.setEnabled(True) ##########################
		self.watch_button.setFixedSize(int(self.w/12),int(self.h/38)) 
		#############################################################

	#Set widgets into layout then set layout into Watch label:
		layout.addWidget(watch_text_label,10,Qt.AlignHCenter) #
		layout.addWidget(self.watch_button,0,Qt.AlignHCenter) #
		watch_label.setLayout(layout) #########################

	#Set Watch label into main Right QVBoxLayout: #############
		self.addWidget(watch_label, alignment=Qt.AlignCenter) #
		#######################################################

	#   %%%%%%%%%%%%%%%%%%%%   END OF WATCH POLYNOM AREA  %%%%%%%%%%%%%%%%%%%%%


	#   %%%%%%%%%%%%%%%%%%%%   SAVE POLYNOM AREA  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#Big label to hold Layout with widgets (text & btn):
		save_label = QLabel() ##################################
		save_label.setAlignment(Qt.AlignCenter) ################
		save_label.setWordWrap(True) ###########################
		save_label.setObjectName("cropper_right_browse_label")##  
		save_label.setFixedSize(int(self.w/5.3),int(self.h/16))#
		########################################################

	#text above the button:
		save_text = """
            <p style="text-align: justify;"><b>Save Obtained Polynom</b></p>
            """
		
		#######################################################

	#Ver Layout to hold text and button:
		layout = QVBoxLayout() #################
		layout.setSpacing(7) ###################
		layout.setContentsMargins(0, 0, 0, 0)###
		layout.setAlignment(Qt.AlignTop) #######
		
	#label to hold save_text:
		save_text_label = QLabel(save_text, save_label)############
		save_text_label.setWordWrap(True) #########################
		save_text_label.setAlignment(Qt.AlignCenter) ##############
		save_text_label.setObjectName("cropper_browse_text")#######
		save_text_label.setFixedSize(int(self.w/6), int(self.h/44)) 
		###########################################################

	#Button to save polynom:
		self.save_button = QPushButton("Save") #####################
		self.save_button.setObjectName("cropper_browse_btn")########
		self.save_button.setEnabled(True) ##########################
		self.save_button.setFixedSize(int(self.w/12),int(self.h/38)) 
		############################################################

	#Set text label & button into layout then set layout into save label:
		layout.addWidget(save_text_label,10,Qt.AlignHCenter) ############
		layout.addWidget(self.save_button,0,Qt.AlignHCenter) ############
		save_label.setLayout(layout) ####################################

	#Set save label into main Right QVBoxLayout:
		self.addWidget(save_label, alignment=Qt.AlignCenter)
#End of Ui components....................................................


	def InitConnects(self):

		#CONNECTION OF - browse_button:========================
		self.browse_button.clicked.connect(self.model.dataLoad)
		#======================================================

		#CONNECTION OF - Data plot button:========================
		self.data_plot_button.clicked.connect(self.model.plotData)
		#=========================================================

		#CONNECTION OF - Cancel plot button:=============================
		self.cancel_plot_button.clicked.connect(self.model.clearPlotData)
		#================================================================

		#CONNECTION OF - QCheckBox with QSpinBox enable:======================
		self.auto_checkBox.stateChanged.connect(self.on_auto_checkbox_changed)
		#=====================================================================

		#CONNECTION OF - Button (plot peaks) with model:==========
		self.peak_plot_button.clicked.connect(self.model.plotPeak)
		#=========================================================

		#CONNECTION OF - Button (Peaks into list) with model:==============
		self.peak_tolist_button.clicked.connect(self.model.sendToList)
		#==================================================================

		#CONNECTION OF - Button (Peaks into list) with model:==============
		#self.Character_Delete_button.clicked.connect(self.model)
		#==================================================================

		#CONNECTION OF - Button (Peaks into list) with model:==============
		self.Character_Create_button.clicked.connect(self.model.runPolyCreator)
		#==================================================================

		#CONNECTION OF - Button (watch polynom) with model:================
		#self.watch_button.clicked.connect(self.model.)
		#==================================================================

		#CONNECTION OF - Button (save polynom) with model:=================
		#self.save_button.clicked.connect(self.model.)

#......................................................................................

	def setEmptyData(self, datatable):
		for _ in range(10):
			row_position = datatable.rowCount()
			datatable.insertRow(row_position)
			datatable.setItem(row_position, 0, QTableWidgetItem(""))
			datatable.setItem(row_position, 1, QTableWidgetItem(""))
#End of setEmptyData................................................

	def resetData(self):
		self.setEmptyData(self.dataTable)
		self.setEmptyData(self.peakTable)
#End of resetData........................

	def setData(self, data):
		#Clear table
		self.dataTable.setRowCount(0)

		#Check: if data is 2D numpy-array with 2 column
		if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
			raise ValueError("waiting for 2D array (N, 2)")

		#Complete the table
		for i, (x, y) in enumerate(data):
			self.dataTable.insertRow(i)

			x_item = QTableWidgetItem(str(x))######
			x_item.setTextAlignment(Qt.AlignCenter)
			self.dataTable.setItem(i, 0, x_item)###

			y_item = QTableWidgetItem(str(y))######
			y_item.setTextAlignment(Qt.AlignCenter)
			self.dataTable.setItem(i, 1, y_item)###
#End of setData....................................

	def setPeak(self, data):
		#Clear table
		self.peakTable.setRowCount(0)

		#Check: if data is 2D numpy-array with 2 column
		if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] != 2:
			raise ValueError("waiting for 2D array (N, 2)")

		#Complete the table
		for i, (x, y) in enumerate(data):
			self.peakTable.insertRow(i)

			x_item = QTableWidgetItem(str(x))######
			x_item.setTextAlignment(Qt.AlignCenter)
			self.peakTable.setItem(i, 0, x_item)###

			y_item = QTableWidgetItem(str(y))######
			y_item.setTextAlignment(Qt.AlignCenter)
			self.peakTable.setItem(i, 1, y_item)###
#End of setPeak....................................


	def on_auto_checkbox_changed(self, state):
		if state == Qt.Checked:
			self.peak_order_spinbox.setEnabled(False)
		else:
			self.peak_order_spinbox.setEnabled(True)
#End of checkbox_changed.............................

	def updateComboListOfPeaks(self, listOfPeaks):
		# Очищаем старое содержимое
		self.combolistOfPeaks.clear()

		# Проверка на None и пустоту
		if listOfPeaks is not None and len(listOfPeaks) > 0:
		    # Добавляем имена в ComboBox
			for name in listOfPeaks.keys():
				self.combolistOfPeaks.addItem(name)
		else:
			print("listOfPeaks is None or empty — ComboBox cleared.")

	def addIntoPeakList(self, peaksName):
		self.combolistOfPeaks.addItem(peaksName)
#End of add into list...........................

#End of Right ########################################################


#+++++++++++++++++++++++++++++ADDITIONA MAIN PACKEDGE++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox,
    QSlider, QPushButton, QInputDialog, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import sys
from functools import partial
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#CLass for creating polynom by Peaks List:
class CalibPolyDlg(QDialog): #QWidget):

#Initializer:
	def __init__(self, parent=None):
	#base class initializer:
		super(CalibPolyDlg, self).__init__(parent)

	#interface Layout to seperate left-right part:
		interfaceLayout = QHBoxLayout() ###########
		interfaceLayout.setContentsMargins(0,0,0,0)
		###########################################

	#for plotting:
		self.figure = Figure()#############################
		self.canvas = FigureCanvas(self.figure)############
		self.toolbar = NavigationToolbar(self.canvas, self)
	#left layout to hold plotting tools:
		pltLayout = QVBoxLayout() #########################
		pltLayout.addWidget(self.toolbar)##################
		pltLayout.addWidget(self.canvas)###################
		###################################################

	#main widget to hold other widgets on right side:
		self.controlWidget = QWidget() ###################
		self.controlWidget.setObjectName("controlPanel")##
		#self.rightLayout is inside the self.controlWidget:
		self.rightLayout = QVBoxLayout(self.controlWidget)
		self.rightLayout.setAlignment(Qt.AlignTop) #######  
		##################################################

	#interface setup:1.set left side;2.set right side:
		interfaceLayout.addLayout(pltLayout)#########
		interfaceLayout.addWidget(self.controlWidget)  
		self.setLayout(interfaceLayout)##############
		#############################################

	#dialog window parent - MainWindow:
		self.mainWinParent = None #####

	#0 - if before 1972; 1 - if after 1972:
		self.modeValue = 0 ################

	#for if/else:
		self.plt_check = None

	#contains peaks data without names:
		self.allPeaks = {} ########
		self.allTimes = {}
		self.allPeaksOrign = {}
		#PEAKS_CHECK
		self.values_before_1972 = [0,0.61,1.1,1.47,1.84,2.25,2.66,3.04,0]
		#self.values_before_1972 = [0,0.61,1.1,1.47,1.84,2.25,2.66,3.04,0][:-1]
		self.values_after_1972 = [0,0.5,0.97,1.44,1.93,2.43,2.69,3.04,0]
		#self.values_after_1972 = [0,0.5,0.97,1.44,1.93,2.43,2.69,3.04,0][:-1]

	#The file name which has maximum exposition:
		self.maxExposefname = None

#.........................................................

#set value representing the date:
	def setModeValue(self, value):
		self.modeValue = value####
#.................................

	def setModule(self,module):
		self.module = module

#set dialog window as a parent:
	def setDialogParent(self, parent):
		self.mainWinParent = parent##	
#.......................................

#set Peaks & Name from MainWindow:
	def setAllPeaks(self, all_peaks, expTime, maxFName, date):
		
		self.EXP_TIME = expTime
		self.MAXFNAME = maxFName
		self.DATE = date

		print(f"ALL PARAMS IS GIVEN: ex-time = {self.EXP_TIME}; MAXFNAME = {self.MAXFNAME};date = {self.DATE};")


		#self.allTimes = all_times
		self.allPeaks.clear() ####
		self.allPeaksOrign.clear()
		for name, peaks in all_peaks.items():
			self.allPeaks[name] = [] ########
			self.allPeaksOrign[name] = [] ###
			for i, (x, y) in enumerate(peaks):
				try:
					if self.modeValue == 0:
						mag = self.values_after_1972[i] 
					else: 
						self.values_before_1972[i]#####
				except IndexError:
					mag = x
				self.allPeaks[name].append((mag, y)) ####
				self.allPeaksOrign[name].append((mag, y))
#........................................................


#draws all peaks in plt:
	def pltDraw(self):
		if self.plt_check:
			self.plt_check.remove()

		ax = self.figure.add_subplot(111)
		self.plt_check = ax
		ax.clear()
		colors = ['b','g','r','c','m','k']

		for idx, (id, values) in enumerate(self.allPeaks.items()):
			x, y = zip(*values)
			ax.scatter(x, y, c=colors[idx % len(colors)])
			ax.set_xlim(0, 20)

		self.canvas.draw()
#.....................................................................

#to start dialog window:
	def show(self):

	#Label and spinBox:
		lbl = QLabel("""
            <p style="text-align: justify;"><b>
			Poly-Expert:</b></p>
            """)
		lbl.setObjectName("cropper_right_browse_label")
		#self.expositionBox = QSpinBox()################
		#self.expositionBox.setValue(1)#################
		#self.expositionBox.setRange(0,150)#############
		self.rightLayout.addWidget(lbl)################
		#self.rightLayout.addWidget(self.expositionBox)#
		###############################################


		#get fname of max exp time from list of self.allTimes

		self.maxExposefname = self.MAXFNAME


		#Dynamic adding labels and sliders:
		colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black']

		# loop allPeaks with index:
		for idx, fname in enumerate(self.allPeaks.keys()):
			color = colors[idx % len(colors)]  # colors of "---"
		
		    # HTML format with appropriate colors:
			html_text = (
				f'<span style="color: {color};">--- </span>'
		        f'<span style="color: white;">file name: {fname} </span>'
		    )
		
			label = QLabel(html_text, self.controlWidget)######
			self.rightLayout.addWidget(label) #################
			slider = QSlider(Qt.Horizontal, self.controlWidget)
			slider.setRange(0, 200)#########################
			slider.setValue(0)#################################
			slider.valueChanged.connect(partial(self.sliderValueChange, fname))
			self.rightLayout.addWidget(slider)#################

			if fname == self.maxExposefname:
				slider.setEnabled(False)

			###################################################



		#Static widgets:
		btnCurve = QPushButton("Get Curve") #########
		btnCurve.setObjectName("cropper_browse_btn")
		btnCurve.clicked.connect(self.getCharacteristicCurve)
		self.rightLayout.addWidget(btnCurve)

		layout = QHBoxLayout()########
		#self.input_box = QSpinBox()###
		#self.input_box.setValue(1)####
		#self.input_box.setRange(1, 20)

		btnPolynom = QPushButton("Polynom")
		btnPolynom.setObjectName("cropper_browse_btn")
		btnPolynom.clicked.connect(self.handle_polynom_button_click)
		self.rightLayout.addWidget(btnPolynom)

		btnSave = QPushButton("Save")
		btnSave.setObjectName("cropper_browse_btn")
		btnSave.clicked.connect(self.save_polynom)

		#layout.addWidget(self.input_box)
		layout.addWidget(btnPolynom)
		layout.addWidget(btnSave)
		self.rightLayout.addLayout(layout)


		super(CalibPolyDlg, self).show()
		self.pltDraw()
#.....................................................

#i-Slider's value changed:
	def sliderValueChange(self,fname, value):

		if self.maxExposefname == None:
			return
		if self.maxExposefname == fname:
			return
			

		originData = self.allPeaksOrign[fname]
	
		# Применяем сдвиг по X:
		value = value*0.1
		shiftData = [[x + value, y] for x, y in originData]

		# Обновляем отображение пиков (если нужно сохранить оригинал — сделай копию)
		self.allPeaks[fname] = shiftData

		# Обнови график, если нужно:
		self.pltDraw()



#Create Cheracteristic Curve:
	def getCharacteristicCurve(self):
		#get characteristic curve from ArxSR moduls:
		#self.allPeaks[fname] 

		mag_all = []
		y_all = []
		print("self.allPeaks.values()", self.allPeaks.values())
		for peaks in self.allPeaks.values():
			for mag, y in peaks:
				mag_all.append(mag)
				y_all.append(y)

		mag_all = np.array(mag_all)
		y_all = np.array(y_all)

		sorted_indices = np.argsort(mag_all)
		mag_all_sorted = mag_all[sorted_indices]
		y_all_sorted = y_all[sorted_indices]

		self.x_curve, self.y_curve= ArxCollibEditor.clean_peaks_convert_flux(mag_all_sorted,y_all_sorted)#do not forget params

		self.figure.clf()
		ax = self.figure.add_subplot(111)
		self.plt_check = ax
		ax.clear()
		ax.scatter(self.x_curve, self.y_curve)
		self.canvas.draw()
		#...........................................

	def save_polynom(self):
		pass
		#polynom
		#self.module.setPolynom(polynom)
		
#Create Polynom & draw :
	def handle_polynom_button_click(self):
		print("Check 1")
		try:
			best_poly,extra_poly_coef,root = ArxCollibEditor.poly_mse_new(self.y_curve, self.x_curve)
			extra_poly = np.poly1d(extra_poly_coef[::-1])
			if best_poly:
				print("Полином найден:",best_poly)
				print("ExtraPoly", extra_poly)
				print("Root", root)
			else:
				print("Не удалось построить хороший полином.")
		except Exception as e:
			print(f"Exception: {e}")

		try:
			print("Check 3")
		    
			if best_poly is None:
				print(" best_poly is None. Невозможно построить график.")
				return
		
			if root is None or extra_poly is None:
				print(" root или extra_poly отсутствует — используем только best_poly")
				x_full = np.linspace(0, np.max(self.y_curve), 500)
				y_full = best_poly(x_full)
			else:
				# 1. x_test для основной части
				x_test = np.linspace(root, np.max(self.y_curve), 300)
				y_test = best_poly(x_test)
	
					# 2. экстраполяция до root
				x_extrapolate = np.linspace(0, root, 200)
				y_extrapolate = extra_poly(x_extrapolate)  # extra_poly уже poly1d
	
					# 3. Объединяем
				x_full = np.concatenate([x_extrapolate, x_test])
				y_full = np.concatenate([y_extrapolate, y_test])
		
		    # Отрисовка
			self.figure.clf()
			ax = self.figure.add_subplot(111)
			self.plt_check = ax
			ax.clear()
			ax.scatter(self.y_curve, self.x_curve, color="steelblue", label="Input Data")
			ax.plot(x_full, y_full, color="orange", linestyle="--", label="Fitted+Extrapolated")
			ax.legend()
			self.canvas.draw()
		
		except Exception as e:
		    print(f" Ошибка при построении графика: {e}")
#		try:
#			max_value = np.max(self.x_curve)
#			x_poly = np.arange(0, max_value)
#			y_poly = []
#
#			print("Check 3")
#
#		   # Проверка наличия best_poly и extra_poly
#			if best_poly is None:
#				print("best_poly is None. Невозможно построить график.")
#				return
#
#			if root is None:
#				print("root is None — используем только best_poly")
#				y_poly = [best_poly(x) for x in x_poly]
#			else:
#				if extra_poly is None:
#					print("extra_poly is None при наличии root. Невозможно построить y_poly.")
#					return
#				for x in x_poly:
#					if x > root:
#						y_poly.append(best_poly(x))
#					else:
#						#y_poly.append(ArxCollibEditor.polynomial_extrapolate(extra_poly, x))
#						y_poly.append(extra_poly(x))
#
#			    # Отрисовка
#			self.figure.clf()
#			ax = self.figure.add_subplot(111)
#			self.plt_check = ax
#			ax.clear()
#			ax.scatter(self.x_curve, self.y_curve, color="steelblue")
#			ax.plot(x_poly, y_poly, color="orange")
#			self.canvas.draw()
#			
#		except Exception as e:
#			    print(f"Ошибка при построении графика: {e}")
	
	#print("poly",poly)
	#	print("extra",extra)
	#	print("root",root)
#	
#		max_value = np.max(self.x_curve)
#		x_poly = np.arange(0,max_value)
#		y_poly = []
#		print("Check 3")
#	
#		if root is None:
#			print("Root is None. Cannot continue processing.")
#			y_poly = [best_poly(x) for x in x_poly]  # без переключения
#		else:
#			y_poly = []
#			for x in x_poly:
#				if x > root:
#					y_poly.append(best_poly(x))
#				else:
#					y_poly.append(extra_poly(x))
#	
#		self.figure.clf()
#	
#		ax = self.figure.add_subplot(111)
#		self.plt_check = ax
#		ax.clear()
#		ax.scatter(self.x_curve, self.y_curve,color="steelblue")
#		ax.plot(x_poly, y_poly,color="orange")
#		self.canvas.draw()
#		#....................................

	
		#print("Check 4")



	   

