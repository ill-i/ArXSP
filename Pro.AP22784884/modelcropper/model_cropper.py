#Qt________________________________________
from PyQt5 import QtGui, QtCore, QtWidgets#
from PyQt5.QtWidgets import* ##############
from PyQt5.QtCore import* #################
from PyQt5.QtCore import pyqtSignal########
from PyQt5.QtGui import* ##################
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

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
import matplotlib.pyplot as plt############
from astropy.io import fits################
from shapely.geometry import LineString
from shapely import affinity
###########################################

#OPENCV LIBS_______________________________
import cv2, imutils########################
###########################################
#from skimage import io, color, transform
#ui-components files_______________________
from PyQt5.QtWidgets import QLCDNumber ####
from PyQt5.QtWidgets import QWidget #######
from PyQt5.QtWidgets import QVBoxLayout ###
from PyQt5.QtWidgets import QSlider #######
from PyQt5.QtWidgets import QGroupBox #####
from PyQt5.QtWidgets import QSizePolicy ###
from PyQt5.QtWidgets import QMessageBox ###
from PyQt5.QtWidgets import QSpinBox ######
###########################################

import matplotlib.pyplot as plt
from datetime import datetime
import os

# Добавляем родительскую папку в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ArxSR import* 
from HeaderEditWindow import FileHeaderEditor

class model_cropper(object):

#Initializer:
	def __init__(self,parent=None):
		super().__init__()
		
		self.win_w = parent.window_width
		self.win_h = parent.window_height
		
	#ui_components:
		self.__left = Left(self)
		self.__right= Right(self)

	#data holder:
		self.arx_data = None
		self.colormap = cv2.COLORMAP_VIRIDIS

	#manipulation holder:
		self.degree = 0
		self.horval = 0

		self.lefval = 0
		self.rigval = 0
		self.topval = 0
		self.dowval = 0

		self.hor_thickness = 6
		self.deg_thickness = 6
		self.hor_color = (0,79,0)
		self.deg_color = (76,0,0)

		self.crop_thickness = 6
		self.crop_color = (255,79,0)

	#Checking of img loading:
		self.isImgLoaded = False

	#QMainWindow as parent:
		self.dialog_parent = None

	#Cancel rotation:
		self.prev_rotation = None
		self.prev_crop = None
#End of Initializer


#Set MainWindow as parent for dialog window:
	def set_dialog_parent(self, parent):
		self.dialog_parent = parent	####
	#...................................

	def set_ColorMap(self, colormap):
		self.colormap = colormap
		self.ImgReLoad()
	#................................


	def set_HorThickness(self, hor_thick):
		self.hor_thickness = hor_thick

	def set_HorColor(self, hor_color):
		self.hor_color = hor_color

	def set_DegThickness(self, deg_thick):
		self.deg_thickness = deg_thick

	def set_DegColor(self, deg_color):
		self.deg_color = deg_color

	def set_CropThickness(self, crop_thick):
		self.crop_thickness = crop_thick

	def set_CropColor(self, crop_color):
		self.crop_color = crop_color
	#......................................

	def ApplyLineStyle(self):
		self.__left.set_CropStyle(self.crop_thickness, self.crop_color )
		self.__left.set_RotateStyle(self.hor_thickness, self.hor_color, self.deg_thickness, self.deg_color)

#Get instances Right & Left:
	def getLeftWidget(self):
		return self.__left
	#.......................
	def getRightLayout(self):
		return self.__right
	#........................

	
#Get fits path open,read & save as png:
	def ImgLoad(self):
		
		try:
		#get file path:		
			self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.dialog_parent , 'Single File', QtCore.QDir.rootPath() , '(*.fits *.fit)')
			
		#If user canceled dialog:
			if not self.fileName:
				print("Err: fits file has not been choosen")
			else:
				try:
					self.arx_data = ArxData(self.fileName)
					self.data = self.arx_data.get_data() ########
					self.data_header = self.arx_data.get_header()
					self.isImgLoaded = True
					print(f"data header: {self.data_header}")
					print(f"fits data: {self.data}")
				except Exception as err:
					self.isImgLoaded = False
					print(f"err in reading {self.fileName}")
					return

				#show png in the label not data:
				self.__left.imgLoad()
				self.__right.reset_sliders()
			
		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
			return
#End of ImgLoad__________________________________________

	#ReLoad only png not Data:
	def ImgReLoad(self):
		
		if self.fileName is None:
			return
		if self.isImgLoaded is False:
			return
		if not self.fileName:
			return
		#show png in the label not data:
		self.__left.imgLoad()
		self.__right.reset_sliders()
#End of ImgReLoad________________________________________

	#Save fits file as a result:
	def fitsSave(self):
		print("going to save fits file")
		msg = QMessageBox()
		print(f"request to save fits file")


		try:
			data = self.arx_data.get_data()
			header = self.arx_data.get_header()

			if data is None or header is None:
				msg.setWindowTitle("Error:")
				msg.setText("data or header is None!!!")
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec_()
				return

			# Get file name without path and extensions:
			base_name = os.path.splitext(os.path.basename(self.fileName))[0]
			
			# Add data time:
			today_str = datetime.today().strftime('%Y-%m-%d')
			suggested_name = f"{base_name}_{today_str}.fits"

			# Open Dialog "Save as"
			save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
				self.dialog_parent,
				"Save FITS File As",
				QtCore.QDir.homePath() + "/" + suggested_name,
				"FITS files (*.fits *.fit)"
			)

			if save_path:
				# Extension check:
				if not save_path.lower().endswith(('.fits', '.fit')):
					save_path += ".fits"

				#self.arx_data.save_as_fits(save_path)

				hdu = fits.PrimaryHDU(data, header=header)
				hdulist = fits.HDUList([hdu])
				hdulist.writeto(save_path, overwrite=True)
				print(f"File saved as: {save_path}")

				msg.setWindowTitle("Success")
				msg.setText(" File has been saved!!!")
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec_()
			else: 
				msg.setWindowTitle("Cancelled")
				msg.setText("Operation was cancelled by user.")
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec_()

		except Exception as e:
			msg.setWindowTitle("Error...")
			msg.setText("Error saving FITS file!")
			msg.setStandardButtons(QMessageBox.Ok)
			msg.exec_()


	def set_degree(self, degree):
		self.degree = degree
		self.rotationChanged(self.horval,self.degree)

	def set_horval(self,horval):
		self.horval = horval
		self.rotationChanged(self.horval,self.degree)


	def set_rightcrop(self, value):
		if self.isImgLoaded is True:
			self.rigval = value
			self.__left.PaintCropLines(value, self.lefval, self.topval, self.dowval)

	def set_leftcrop(self, value):
		if self.isImgLoaded is True:
			self.lefval = value
			self.__left.PaintCropLines(self.rigval, value, self.topval, self.dowval)

	def set_topcrop(self, value):
		if self.isImgLoaded is True:
			self.topval = value
			self.__left.PaintCropLines(self.rigval, self.lefval, value, self.dowval)

	def set_downcrop(self,value):
		if self.isImgLoaded is True:
			self.dowval = value
			self.__left.PaintCropLines(self.rigval, self.lefval, self.topval, value)

#Rotate Horizone SLider changed:
	def rotationChanged(self, value, degree):
		if self.isImgLoaded is True:
			self.__left.paintLineRotateHorizont(value, degree)
#End of Rotate Horizone SLider changed___________________

#Run additional page to make settings:
	def run_Settings(self):
		print("setting window has been run")
		dialog = SettingDialog(self.dialog_parent, self)
		dialog.show()
		#..................
#End of setting


#Edit button clicked:
	def EditHeader(self):
		print("Header Editor has been run")
		dialog = FileHeaderEditor(self.arx_data, self, self.dialog_parent)
		dialog.show()


	def set_header(self, header):
		self.arx_data.set_header(header)
		print("Получен header:", header)


#End of Edit button clicked______________________________

	def Rotate_fits(self):

		if self.isImgLoaded is False:
			return

		editor= ArxDataEditor(self.arx_data)

		self.prev_rotation = self.arx_data


		rot_data = editor.rotate(-1*self.degree)
		self.arx_data = rot_data
		#self.data = rot_data.get_data() ########
		#self.data_header = rot_data.get_header()

		#show png in the label not data:
		self.__left.imgLoad()

#End of rotate fits______________________________

	def rotate_Cancel(self):

		if self.prev_rotation is None:
			return

		if self.isImgLoaded is True:

			self.arx_data = self.prev_rotation
			self.data = self.arx_data.get_data() ########
			self.data_header = self.arx_data.get_header()

			#Checking:
			self.isImgLoaded = True
			self.__left.imgLoad()		
#End of cancel rotate____________________________

	def Crop_fits(self):

		if self.isImgLoaded is False:
			return

		editor= ArxDataEditor(self.arx_data)

		self.prev_crop = self.arx_data

		crop_data = editor.crop(self.rigval, self.lefval, self.topval, self.dowval)
		self.arx_data = crop_data

		self.__left.imgLoad()
#End of crop fits__________________________________________________________________

	def crop_Cancel(self):

		if self.prev_crop is None:
			return

		if self.isImgLoaded is True:

			self.arx_data = self.prev_crop
			self.data = self.arx_data.get_data() ########
			self.data_header = self.arx_data.get_header()

			#Checking:
			self.isImgLoaded = True
			self.__left.imgLoad()	
#End of cancel crop______________________________________


#End of model cropper####################################



#CLass Left of model cropper:
class Left(QLabel):

#Constructor:
	def __init__(self, parent=None):

	#Main QLabel setting:
		QLabel.__init__(self, None) #######################################
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) ##
		self.setAlignment(QtCore.Qt.AlignCenter) ##########################
		self.setText('BROWSE & LOAD FITS FILE TO SEE THE IMAGE') ##########
		self.setObjectName("cropper_left_mainlabel") ######################
		#self.setScaledContents(True) #####################################
		###################################################################
		
	#Init it's parent - model cropper:
		self.model = parent
		
	#w,h - the size of main QLabel:
		self.w = self.width() #####
		self.h = self.height() ####
		###########################
		
	#the image to draw:
		self.loaded_image = None
		self.tmp_image = None

		self.hor_thickness = 6
		self.deg_thickness = 6
		self.hor_color = (0,79,0)
		self.deg_color = (76,0,0)

		self.crop_thickness = 6
		self.crop_color = (255,79,0)
		########################

	#curr row for the animation:
		#self.current_row = 0 ###########################
	#tim#er for rendering:
		#self.timer = QTimer() ##########################
		#self.timer.timeout.connect(self.updateAnimation)  
	#to #hold partial img:
		#self.partial_image = None ######################
		
	#Init menu on screen:
		self.setFocusPolicy(Qt.ClickFocus)
#End of Init of Left....................................


	def set_RotateStyle(self, hor_thick, hor_color, deg_thick, deg_color):
		self.hor_thickness = hor_thick
		self.deg_thickness = deg_thick

		self.hor_color = hor_color
		print(f"hor_color = {hor_color}")
		self.deg_color = deg_color

	def set_CropStyle(self, crop_thick, crop_color):
		self.crop_thickness = crop_thick
		self.crop_color = crop_color

#re-set image:
	def set_reset(self):
	#get copy of origin image:
		#self.tmp_image = self.origin_image.copy()
		self.tmp_image = self.loaded_image.copy()
		self.update() ############################
		return self.w ############################
	#.............................................


#Load image with cv2:
	def imgLoad(self):
		#try:
		##read with cv2:
		#	self.loaded_image = self.model.arx_data.get_image(self.model.colormap)
		##animation:
		#	try:
		#		self.startAnimation()
		#	except Exception as err: 
		#		self.timer.stop() ###############
		#		self.set_reset() ################
		#		self.setImage(self.loaded_image)#
		##set img at the end:
		#	self.set_reset() ####################
		#	self.setImage(self.loaded_image) ####
		#except Exception as err: 
		#	print(f"Unexpected {err=}, {type(err)=}")	
		try:
		    self.loaded_image = self.model.arx_data.get_image(self.model.colormap)
		    self.set_reset()  # self.tmp_image = self.loaded_image.copy(), self.update()
		    self.setImage(self.loaded_image)
		except Exception as err: 
		    print(f"Unexpected {err=}, {type(err)=}")	
	#................................................

#img Load animation:
	#def startAnimation(self): 
	#	self.current_row = 0 ################################
	#	self.partial_image = np.zeros_like(self.loaded_image)  
	#	self.timer.start(1) #################################
	#........................................................

#update animated img:
	#def updateAnimation(self):
	#	if self.current_row < self.partial_image.shape[0]:
	#		self.partial_image[self.current_row, :] = self.loaded_image[self.current_row, :]
	#		self.setImage(self.partial_image) ##############################################
	#		self.current_row += 1 ##########################################################
	#	else:
	#		self.timer.stop() ##############################################################
	#.......................................................................................

#set image into main QLabel:
	def setImage(self, img):
		img = imutils.resize(img,width=self.width(),height=self.height()) ########################
		frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #############################################
		qimg = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0], QImage.Format_RGB888)
		self.setPixmap(QtGui.QPixmap.fromImage(qimg)) ############################################
	#.............................................................................................

#update image:
	def update(self):
		if self.tmp_image is not None:
			self.setImage(self.tmp_image)
	#.................................

#if window is resized:
	def resizeEvent(self, e: QResizeEvent) -> None:
		super().resizeEvent(e) ####################
		if self.tmp_image is not None: #########
			self.setImage(self.tmp_image) ######
	#..............................................

#Manipulations:

#Rotate horizontal line:
	def rotateHorizontLine(self, x1, y1, x2, y2, angle):

		line = LineString([(x1,y1), (x2,y2)])

		cx = (x1+x2)/2
		cy = (y1+y2)/2

		rotated = affinity.rotate(line, angle, origin=(cx,cy))
		x1_rotated, x2_rotated = rotated.coords.xy[0][0], rotated.coords.xy[0][1] 
		y1_rotated, y2_rotated = rotated.coords.xy[1][0], rotated.coords.xy[1][1] 

		#len_initial = ((x1-x2)**2+(y1-y2)**2)**0.5
		#len_rotated = ((x1_rotated-x2_rotated)**2+(y1_rotated-y2_rotated)**2)**0.5
		
		return (int(x1_rotated),int(y1_rotated)),(int(x2_rotated),int(y2_rotated))
#End of rotate horizontal line....................................................

#Painting Horizontal line:
	def paintLineRotateHorizont(self,horvalue, deg):

		w = self.loaded_image.shape[1]
		h = self.loaded_image.shape[0]
		
		point1 = (0, int(h*horvalue/100))
		point2 = (w, int(h*horvalue/100))
				
		#rotate	
		point3, point4 =self.rotateHorizontLine(point1[0],point1[1],point2[0],point2[1],deg)
		point0_3, point0_4 =self.rotateHorizontLine(point1[0],point1[1],point2[0],point2[1],0)

		self.tmp_image = self.loaded_image.copy()
		self.tmp_image = cv2.line(self.tmp_image, point3, point4, self.deg_color, self.deg_thickness)
		self.tmp_image = cv2.line(self.tmp_image, point0_3, point0_4, self.hor_color, self.hor_thickness)

		try:
			self.update()

		except Exception as err: 
			print("Unexpected here")
#End of painting Horizontal line............................................................


#Painting Crop lines:
	def PaintCropLines(self, right, left, top, down):
		#Here
		color = self.crop_color#norm
		
		color_rl = (0,0,255)
		color_td = (0,0,255)
		
		#Line thickness of 9 px
		thickness = self.crop_thickness
		
		w = self.loaded_image.shape[1]
		h = self.loaded_image.shape[0]
		
		if( int(w*left/100) < (w - int(w*right/100)) ):
			color_rl = color
		if( int(h*top/100) < (h - int(h*down/100))):
			color_td = color
		
		lft_point1 = (int(w*left/100), 0)
		lft_point2 = (int(w*left/100), h)
		
		rht_point1 = (w - int(w*right/100), 0)
		rht_point2 = (w - int(w*right/100), h)
		
		top_point1 = (0, int(h*top/100))
		top_point2 = (w, int(h*top/100))
		
		dwn_point1 = (0, h - int(h*down/100))
		dwn_point2 = (w, h - int(h*down/100))
		
		#Cv draw algorithm:
		self.tmp_image = self.loaded_image.copy()
		
		self.tmp_image = cv2.line(self.tmp_image, lft_point1, lft_point2, color_rl, thickness)
		self.tmp_image = cv2.line(self.tmp_image, rht_point1, rht_point2, color_rl, thickness)
		self.tmp_image = cv2.line(self.tmp_image, top_point1, top_point2, color_td, thickness)
		self.tmp_image = cv2.line(self.tmp_image, dwn_point1, dwn_point2, color_td, thickness)
		
		self.update()
#End of painting Crop lines............................................................

#End of Left CLass


#CLass Right of model cropper:
class Right(QVBoxLayout):

#Initializer:
	def __init__(self, parent=None):

	#Init Main Layout:
		super().__init__() ##################
		self.setAlignment(Qt.AlignTop) ######
		self.setContentsMargins(0, 0, 0, 0) #
		#####################################

	#model as parent:
		self.model = parent

		self.w = self.model.win_w
		self.h = self.model.win_h
	#Browsing__________________________________________________
	#This is label to set Layout with widgets (text & btn):
		browse_label = QLabel() ###############################
		#browse_label.setAlignment(Qt.AlignTop)
		browse_label.setAlignment(Qt.AlignCenter) #############
		#перенос текста #######################################
		browse_label.setWordWrap(True) ########################
		browse_label.setObjectName("cropper_right_browse_label") 
	#Size of dark label #######################################
		browse_label.setFixedSize(int(self.w/5), int(self.h/13)) ###################
		#######################################################

	#text above the button:
		browse_text = """
			<p style="text-align: justify;"><b>
			Upload fits format file</b></p>
			"""
		browse_text_label = QLabel(browse_text, browse_label)
		#######################################################
		
		#browse_text_label.setAlignment(Qt.AlignTop)
	#Layout to hold text and button:
		layout = QVBoxLayout() #############################
		layout.setSpacing(7) ##############################
		layout.setContentsMargins(0, 0, 0, 5) ##############
		layout.setAlignment(Qt.AlignTop) ###################
		
	#label for the text:
		browse_text_label.setWordWrap(True) ################
		browse_text_label.setAlignment(Qt.AlignCenter) #####
		browse_text_label.setObjectName("cropper_browse_text")
		browse_text_label.setFixedSize(int(self.w/6), int(self.h/44)) ############
		####################################################

	#Button to browse:
		self.browse_button = QPushButton("Browse") ###########
		self.browse_button.setObjectName("cropper_browse_btn")
		self.browse_button.setEnabled(True) ##################
		self.browse_button.setFixedSize(int(self.w/12), int(self.h/30)) #############
		######################################################

	#Set btn & text(lbl) into layout then set layout into browse label widget:
		browse_label.setLayout(layout) #######################################
		layout.addWidget(browse_text_label, 0, Qt.AlignHCenter) #############
		layout.addWidget(self.browse_button, 0, Qt.AlignHCenter) #############

	#Set browse label into main QVBoxLayout:
		self.addWidget(browse_label, alignment=Qt.AlignCenter)
		######################################################################

	#Init connections:
		self.browse_button.clicked.connect(self.model.ImgLoad)
	#End of Browsing__________________________________________________________

	#For tabs controller______________________________________________________
		#This is label to hold tab_layout:
		main_tabs_label = QLabel() ################################
		main_tabs_label.setAlignment(Qt.AlignCenter) ##############
		main_tabs_label.setWordWrap(True) #########################
		main_tabs_label.setObjectName("cropper_right_browse_label") 
		#main_tabs_label.setFixedSize(720, 1000) ###################
		main_tabs_label.setMinimumSize(int(self.model.win_w/5), int(self.model.win_h/2))
		main_tabs_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		###########################################################

		#Text of tabs controller
		tab_text = """
		<p style="text-align:justify;"><b>FITS Data Processing</b></p>
			"""

		#label for text of tabs controller
		tab_text_label = QLabel(tab_text)##################
		tab_text_label.setWordWrap(True) ##################
		tab_text_label.setAlignment(Qt.AlignCenter) #######
		tab_text_label.setObjectName("cropper_browse_text")
		#Size of label for the text:
		#tab_text_label.setFixedSize(600, 70) ##############
		tab_text_label.setMinimumSize(int(self.model.win_w/6), int(self.model.win_h/44))
		tab_text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
		###################################################

		#Layout to hold text and tabs:
		tab_layout = QVBoxLayout() #############################
		#tab_layout.setSpacing(10) ##############################
		tab_layout.setContentsMargins(0, 0, 0, 0) ##############
		tab_layout.setAlignment(Qt.AlignTop) ###################

		#Main Tab:
		tabs = QTabWidget() ################
		#tabs.setObjectName("cropper_tab_widget")
		#tabs.setMinimumSize(int(self.model.win_w/7), int(self.model.win_h/4))
		tabs.setMinimumSize(int(self.model.win_w/6), int(self.model.win_h/4))
		tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

		#Number-1 tab page:
		tab_1 = QWidget() ##################
		tab_1.setObjectName("cropper_tab_widget") #??????????????
		tab_layout_rotate = QVBoxLayout() ##
		tab_layout_rotate.setAlignment(Qt.AlignTop)
		tab_1.setLayout(tab_layout_rotate) #
		tab_layout_rotate.setSpacing(1)  # Убирает вертикальные отступы между виджетами
		tab_layout_rotate.setContentsMargins(0, 0, 0, 5)  # Убирает внешние отступы

		#Number-2 tab page:
		tab_2 = QWidget()  #################
		#tab_2.setObjectName("cropper_tab_widget")
		tab_layout_crop = QVBoxLayout() ####
		tab_2.setLayout(tab_layout_crop) ###
		tab_layout_crop.setSpacing(1)
		tab_layout_crop.setContentsMargins(0, 0, 0, 5)  # Убирает внешние отступы

		tabs.addTab(tab_1, "Rotate") ###################################
		tabs.addTab(tab_2, "Crop") #####################################
		tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed) 
		################################################################
	
		#Install text & tabs into layout then set layout into main label:
		tab_layout.addWidget(tab_text_label, 1, Qt.AlignHCenter) #######
		tab_layout.addWidget(tabs, alignment=Qt.AlignCenter)#Left) ##############
		#################################################################
		main_tabs_label.setLayout(tab_layout) ###########################
		#Set main_tabs_label into Right:
		self.addWidget(main_tabs_label, alignment=Qt.AlignCenter)

	#End of Main Tab###########################################################

	#Rotation control:
		# 1. Create  2 specified sliders:
		self.rotate_hor_labslider = QLabeledSlider("""<p style="text-align:left;"><b>Vertical position:</b></p>""")
		self.rotate_deg_labslider = QLabeledSlider("""<p style="text-align:left;"><b>Angle:</b></p>""") ###########

		# 2. Set ranges for 2 sliders:
		self.rotate_hor_labslider.set_slider_range(0,100) ########
		self.rotate_deg_labslider.set_slider_range(0,180) ########

		tab_layout_rotate.addWidget(self.rotate_hor_labslider) ###
		tab_layout_rotate.addWidget(self.rotate_deg_labslider) ###
		##########################################################

		#Connections:
		self.rotate_hor_labslider.valueChanged.connect(lambda val: self.model.set_horval(val))##
		self.rotate_deg_labslider.valueChanged.connect(lambda val: self.model.set_degree(val))##
		########################################################################################


	#Ok-Cancel buttons:
		# 1. Create layout for 2 buttons:
		rotate_buttons_layout = QHBoxLayout() #################
		rotate_buttons_layout.setContentsMargins(2, 0, 3, 0)
		rotate_buttons_layout.setSpacing(5) ##################

		# 2. Create 2 buttons  ###########################################################
		self.btn_rotate_cancel = QPushButton("Cancel") ###################################
		self.btn_rotate_cancel.setObjectName("cropper_browse_btn") #######################
		self.btn_rotate_cancel.setMinimumHeight(int(self.model.win_h/30)) ######################################
		self.btn_rotate_cancel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		self.btn_rotate_ok = QPushButton("Rotate") ###################################
		self.btn_rotate_ok.setObjectName("cropper_browse_btn") #######################
		self.btn_rotate_ok.setMinimumHeight(int(self.model.win_h/30)) ######################################
		self.btn_rotate_ok.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		#DO CONNECTION OK/CANCEL BUTTONS HERE......................
		self.btn_rotate_ok.clicked.connect(self.model.Rotate_fits)
		self.btn_rotate_cancel.clicked.connect(self.model.rotate_Cancel)
		##############################################################

		# 3. Set 2 buttons into rotate buttons layout:
		rotate_buttons_layout.addWidget(self.btn_rotate_cancel) ###
		rotate_buttons_layout.addWidget(self.btn_rotate_ok) #######

		# 4. Set rotate buttons layout tab rotate layout:
		tab_layout_rotate.addLayout(rotate_buttons_layout) ########
		###########################################################

	#Cropping control:
		# 1. Create  4 specified sliders:
		self.crop_lft_labslider = QLabeledSlider("""<p style="text-align:left;"><b>Left position:</b></p>""") ##
		self.crop_rgt_labslider = QLabeledSlider("""<p style="text-align:left;"><b>Right position:</b></p>""") #
		self.crop_top_labslider = QLabeledSlider("""<p style="text-align:left;"><b>Upper position:</b></p>""") #
		self.crop_dwn_labslider = QLabeledSlider("""<p style="text-align:left;"><b>Lower position:</b></p>""") #

		# 2. Set ranges for 4 sliders:
		self.crop_lft_labslider.set_slider_range(0,100) ########
		self.crop_rgt_labslider.set_slider_range(0,100) ########
		self.crop_top_labslider.set_slider_range(0,100) ########
		self.crop_dwn_labslider.set_slider_range(0,100) ########

		#DO CONNECTION sliders WITH MODEL HERE..................

		tab_layout_crop.addWidget(self.crop_lft_labslider) ###
		tab_layout_crop.addWidget(self.crop_rgt_labslider) ###
		tab_layout_crop.addWidget(self.crop_top_labslider) ###
		tab_layout_crop.addWidget(self.crop_dwn_labslider) ###
		######################################################

		#Connections:
		self.crop_lft_labslider.valueChanged.connect(lambda val: self.model.set_leftcrop(val))###
		self.crop_rgt_labslider.valueChanged.connect(lambda val: self.model.set_rightcrop(val))##
		self.crop_top_labslider.valueChanged.connect(lambda val: self.model.set_topcrop(val))####
		self.crop_dwn_labslider.valueChanged.connect(lambda val: self.model.set_downcrop(val))###
		#########################################################################################


	#Ok-Cancel buttons:
		# 1. Create layout for 2 buttons:
		crop_buttons_layout = QHBoxLayout() #################
		crop_buttons_layout.setContentsMargins(2, 1, 3, 1)
		crop_buttons_layout.setSpacing(25) ##################

		# 2. Create 2 buttons  #########################################################
		self.btn_crop_cancel = QPushButton("Cancel") ###################################
		self.btn_crop_cancel.setObjectName("cropper_browse_btn") #######################
		self.btn_crop_cancel.setMinimumHeight(int(self.model.win_h/30)) ######################################
		self.btn_crop_cancel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)


		self.btn_crop_ok = QPushButton("Crop") ###################################
		self.btn_crop_ok.setObjectName("cropper_browse_btn") #######################
		self.btn_crop_ok.setMinimumHeight(int(self.model.win_h/30)) ######################################
		self.btn_crop_ok.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		#DO CONNECTION OK/CANCEL BUTTONS HERE......................
		self.btn_crop_ok.clicked.connect(self.model.Crop_fits)
		self.btn_crop_cancel.clicked.connect(self.model.crop_Cancel)

		# 3. Set 2 buttons into rotate buttons layout:
		crop_buttons_layout.addWidget(self.btn_crop_cancel) ###
		crop_buttons_layout.addWidget(self.btn_crop_ok) #######

		# 4. Set rotate buttons layout tab rotate layout:
		tab_layout_crop.addLayout(crop_buttons_layout) ########
		###########################################################
	#End of Horizontal line control___________________________________________

	#Edit header __________________________________________________
	#This is label to set Layout with widgets (text & btn):
		edit_label = QLabel() ###############################
		edit_label.setAlignment(Qt.AlignCenter) #############
		#перенос текста #######################################
		edit_label.setWordWrap(True) ########################
		edit_label.setObjectName("cropper_right_browse_label") 
	#Size of dark label #######################################
		edit_label.setFixedSize(int(self.model.win_w/5), int(self.model.win_h/13)) ###################
		#######################################################

	#text above the button:
		edit_text = """
			<p style="text-align: justify;"><b>Edit Header of fits file</b></p>
			"""
		edit_text_label = QLabel(edit_text, edit_label)
		#######################################################

	#Layout to hold text and button:
		layout = QVBoxLayout() #############################
		layout.setSpacing(7) ##############################
		layout.setContentsMargins(0, 0, 0, 5) ##############
		layout.setAlignment(Qt.AlignTop) ###################
		
	#label for the text:
		edit_text_label.setWordWrap(True) ################
		edit_text_label.setAlignment(Qt.AlignCenter) #####
		edit_text_label.setObjectName("cropper_browse_text")
	#Size of label for the text:
		edit_text_label.setFixedSize(int(self.model.win_w/6), int(self.model.win_h/44)) ############
		####################################################

	#Button to browse:
		self.edit_button = QPushButton("Edit") ###########
		self.edit_button.setObjectName("cropper_browse_btn")
		self.edit_button.setEnabled(True) ##################
		self.edit_button.setFixedSize(int(self.model.win_w/12), int(self.model.win_h/30)) #############
		######################################################

	#Set btn & text(lbl) into layout then set layout into browse label widget:
		edit_label.setLayout(layout) #######################################
		layout.addWidget(edit_text_label, 1, Qt.AlignHCenter) #############
		layout.addWidget(self.edit_button, 0, Qt.AlignHCenter) #############

	#Set browse label into main QVBoxLayout:
		self.addWidget(edit_label, alignment=Qt.AlignCenter)
		######################################################################

	#Init connections:
		self.edit_button.clicked.connect(self.model.EditHeader)
	#End of Editing__________________________________________________________


	#Saving __________________________________________________
	#This is label to set Layout with widgets (text & btn):
		save_label = QLabel() ###############################
		save_label.setAlignment(Qt.AlignCenter) #############
		#перенос текста #######################################
		save_label.setWordWrap(True) ########################
		save_label.setObjectName("cropper_right_browse_label") 
	#Size of dark label #######################################
		save_label.setFixedSize(int(self.model.win_w/5), int(self.model.win_h/13)) ###################
		#######################################################

	#text above the button:
		save_text = """
			<p style="text-align: justify;"><b>Save fits file</b></p>
			"""
		save_text_label = QLabel(save_text, save_label)
		#######################################################

	#Layout to hold text and button:
		layout = QVBoxLayout() #############################
		layout.setSpacing(7) ##############################
		layout.setContentsMargins(0, 0, 0, 5) ##############
		layout.setAlignment(Qt.AlignTop) ###################
		
	#label for the text:
		save_text_label.setWordWrap(True) ################
		save_text_label.setAlignment(Qt.AlignCenter) #####
		save_text_label.setObjectName("cropper_browse_text")
	#Size of label for the text:
		save_text_label.setFixedSize(int(self.model.win_w/6), int(self.model.win_h/44)) ############
		####################################################

	#Button to browse:
		self.save_button = QPushButton("Save") #############
		self.save_button.setObjectName("cropper_browse_btn")
		self.save_button.setEnabled(True) ##################
		self.save_button.setFixedSize(int(self.model.win_w/12), int(self.model.win_h/30)) #############
		####################################################

	#Set btn & text(lbl) into layout then set layout into browse label widget:
		save_label.setLayout(layout) #######################################
		layout.addWidget(save_text_label, 1, Qt.AlignHCenter) #############
		layout.addWidget(self.save_button, 0, Qt.AlignHCenter) #############

	#Set browse label into main QVBoxLayout:
		self.addWidget(save_label, alignment=Qt.AlignCenter)
		######################################################################
	#Init connections:
		#self.edit_button.clicked.connect(self.model.EditHeader)
	#End of Editing__________________________________________________________
	#Settings: ________________________________________________
	#This is label to set Layout with widgets (text & btn):
		Setting_label = QLabel() ################################
		Setting_label.setAlignment(Qt.AlignCenter) ##############
		#перенос текста #########################################
		Setting_label.setWordWrap(True) #########################
		Setting_label.setObjectName("cropper_right_browse_label") 
	#Size of dark label #########################################
		Setting_label.setFixedSize(int(self.model.win_w/5), int(self.model.win_h/13)) ####################
		#########################################################

	#text above the button:
		Setting_text = """
	<p style="text-align: justify;"><b>Tool View Settings:</b></p>
		"""
		Setting_text_label = QLabel(Setting_text, Setting_label)
		Setting_text_label.setWordWrap(True)
		Setting_text_label.setAlignment(Qt.AlignCenter)
		Setting_text_label.setObjectName("cropper_browse_text")
		
		# Убираем фиксированную ширину, можно оставить лишь высоту (при желании):
		# Setting_text_label.setFixedSize(600, 38)  # УДАЛЕНО
		Setting_text_label.setFixedHeight(int(self.model.win_h/30))		# Если хотите ограничить высоту
		
		# ---------------------------------------------
		# 3) Кнопка (с иконкой и прозрачным фоном)
		# ---------------------------------------------
		self.Setting_button = QPushButton("")
		self.Setting_button.setObjectName("trans")
		self.Setting_button.setEnabled(True)
		
		# Также убираем фиксированную ширину, оставляем только высоту:
		# self.Setting_button.setFixedSize(400, 70)  # УДАЛЕНО
		self.Setting_button.setFixedHeight(int(self.model.win_h/30))
		self.Setting_button.setFixedWidth(int(self.model.win_w/30))

		# Устанавливаем иконку
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "color_setting.png")
		self.Setting_button.setIcon(QIcon(icon_path))
		self.Setting_button.setIconSize(QSize(int(self.model.win_w/30), int(self.model.win_h/30)))

		self.Setting_button.clicked.connect(self.model.run_Settings)

		Empty_button = QPushButton("")
		Empty_button.setObjectName("trans")
		Empty_button.setEnabled(False)
		
		# Также убираем фиксированную ширину, оставляем только высоту:
		# self.Setting_button.setFixedSize(400, 70)  # УДАЛЕНО
		Empty_button.setFixedHeight(int(self.model.win_h/30))
		# ---------------------------------------------
		# 4) Создаём горизонтальный layout,
		#	кладём в него текст и кнопку
		# ---------------------------------------------
		h_layout = QHBoxLayout()
		h_layout.setSpacing(1)
		h_layout.setContentsMargins(0, 0, 0, 0)
		h_layout.setAlignment(Qt.AlignCenter)
		
		# Добавляем текст и кнопку в горизонтальный layout
		h_layout.addWidget(Setting_text_label, 10)   # Можно добавить stretch=10
		h_layout.addWidget(self.Setting_button, 10)   # stretch=0
		h_layout.addWidget(Empty_button, 5) 

		# Присваиваем layout контейнеру (Setting_label)
		Setting_label.setLayout(h_layout)
		
		# ---------------------------------------------
		# 5) Добавляем Setting_label в главный лейаут
		# ---------------------------------------------
		self.addWidget(Setting_label, alignment=Qt.AlignCenter)
		######################################################################

	#Init connections:
		self.save_button.clicked.connect(self.model.fitsSave)

	#End of Saving__________________________________________________________


#End of Init of Right ########################################################

	def reset_sliders(self):
		self.rotate_hor_labslider.reset()
		self.rotate_deg_labslider.reset()
		self.crop_lft_labslider.reset()
		self.crop_rgt_labslider.reset()
		self.crop_top_labslider.reset()
		self.crop_dwn_labslider.reset()

class ColorChangingSlider(QWidget):

	def __init__(self):
		super().__init__()
		self.layout = QVBoxLayout() 
		self.layout.setAlignment(Qt.AlignLeft)
		self.slider = QSlider(Qt.Horizontal)
		self.slider.setMinimum(0)
		self.slider.setMaximum(100)
		self.slider.setValue(0)
		self.slider.setTickInterval(1)
		self.slider.setTickPosition(QSlider.TicksBelow)
		self.slider.setObjectName("slider")  # Фиксированное имя

		self.slider.valueChanged.connect(self.update_slider_class)

		self.layout.addWidget(self.slider)
		self.setLayout(self.layout)

		self.load_styles()
		self.update_slider_class()

	def load_styles(self):
		if hasattr(sys, '_MEIPASS'):
			css_file = os.path.join(sys._MEIPASS, "crop_styles.css")
			print("Style file loaded: ",css_file)
		else:
			css_file = os.path.abspath("crop_styles.css")
			print("Style file loaded: ",css_file)
		if not os.path.isfile(css_file):
			print(f"File not found: {css_file}")
		else:
			with open(css_file, "r", encoding='windows-1251') as f:
				style = f.read()
				self.setStyleSheet(style)
				
	def update_slider_class(self):
		current_value = self.slider.value()
		max_value = self.slider.maximum()

		# Защищаемся от деления на ноль.
		# Если max_value == 0, сделаем ratio = 0.
		ratio = current_value / max_value if max_value != 0 else 0

		# Меняем "пороги" на основе ratio, а не фиксированных значений.
		if ratio < 0.3:	  # соответствует < 30% от максимума	
			color_class = "green"
		elif ratio < 0.6:	# 30%..60%
			color_class = "yellow"
		elif ratio < 0.8:	# 60%..80%
			color_class = "orange"
		else: 
			color_class = "red"

		self.slider.setProperty("styleClass", color_class)
		self.slider.style().unpolish(self.slider)
		self.slider.style().polish(self.slider)

	def reset(self):
		self.slider.setValue(0)


class QLabeledSlider(QWidget):

#Signal value changing:
	valueChanged = pyqtSignal(int)

#Initializer:
	def __init__(self, label_text, parent=None):
		super().__init__(parent)

		# Главный вертикальный layout для (строка с меткой+LCD) + слайдер
		main_layout = QVBoxLayout()
		main_layout.setContentsMargins(0, 0, 0, 0)
		main_layout.setSpacing(1)			  # 1 px между "верхним блоком" и слайдером
		main_layout.setAlignment(Qt.AlignTop)  # Прижимаем все виджеты к верхней границе

		# --- Горизонтальный layout: Label (слева) и LCD (справа)
		top_layout = QHBoxLayout()
		top_layout.setContentsMargins(0, 0, 1, 0)
		top_layout.setSpacing(1)			  # 1 px между Label и LCD

		# --- Метка
		self.my_label = QLabel(label_text)
		self.my_label.setObjectName("cropper_browse_text") 
		self.my_label.setAlignment(Qt.AlignCenter)  
		# Расширяется по горизонтали, не растягивается излишне по вертикали
		self.my_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		# --- LCD
		self.my_lcd = QLCDNumber()
		# Установим кол-во разрядов, если нужно (напр. до 3, если макс. значение слайдера 100)
		self.my_lcd.setDigitCount(3)
		#self.my_lcd.setSegmentStyle(QLCDNumber.Filled)

		base_height = self.my_label.sizeHint().height()
		new_height = base_height * 2
		# Применяем:
		self.my_label.setFixedHeight(new_height)
		self.my_lcd.setFixedHeight(new_height)
		self.my_lcd.setMinimumWidth(110)

		# Добавляем Label, «пружинку» для сдвига, и LCD
		top_layout.addWidget(self.my_label)
		top_layout.addStretch(1)  # растяжение, чтобы LCD прилипал к правому краю
		top_layout.addWidget(self.my_lcd)

		# --- Слайдер (ColorChangingSlider)
		self.my_slider = ColorChangingSlider()
		self.my_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		# Подключаем отображение значения слайдера в LCD
		#self.my_slider.slider.valueChanged.connect(self.my_lcd.display)
		self.my_slider.slider.valueChanged.connect(self.handleValueChange)
		# --- Собираем всё в главный layout
		main_layout.addLayout(top_layout)	 # строка (метка + LCD)
		main_layout.addWidget(self.my_slider) # сам слайдер
		self.setLayout(main_layout)

		# Размерная политика самого QLabeledSlider:
		# По горизонтали можем растягиваться, а по вертикали — «фиксирована» или "Preferred"
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
	#_______________________________________________________________________________________


#Init slider range:
	def set_slider_range(self, min_val, max_val):
		self.my_slider.slider.setMinimum(min_val)
		self.my_slider.slider.setMaximum(max_val)
#________________________________________________

#Reset value:
	def reset(self):
		self.handleValueChange(0)
		self.my_slider.reset()
#________________________________

#If Slider value changed:
	def handleValueChange(self, val):
		self.my_lcd.display(val)
		self.valueChanged.emit(val)
#___________________________________

#Get current value of slider:
	def getValue(self):
		return self.my_slider.slider.value()
#___________________________________________


# Dialogs Window:
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog
class SettingDialog(QMainWindow):

	def __init__(self, mainparent=None, model=None):
		super().__init__(mainparent)

	#To set params in model:
		self.Model = model

		self.setWindowTitle("View settings")
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "color_setting.png")
		self.setWindowIcon(QIcon(icon_path))

	#Obtain Screen Sizes:
		screen = QApplication.primaryScreen()#############
		screen_geometry = screen.availableGeometry()######
		screen_width = screen_geometry.width()############
		screen_height = screen_geometry.height()##########
		##################################################
		#Setting the main window size as a% of the screen:
		# 80% of screen width_____________________________
		self.window_width = int((screen_width) * 0.2)#####
		# 80% of screen height____________________________
		self.window_height = int((screen_height) * 0.4)### 
		self.resize(self.window_width,self.window_height)#
	######################################################


	#Central widget and it's Layout:
		central_widget = QWidget() ######################
		central_layout = QVBoxLayout() ##################
		central_layout.setContentsMargins(10, 0, 10, 10)

		central_widget.setLayout(central_layout) ########
		self.setCentralWidget(central_widget) ###########
		#################################################


	#Spec Display mode GroupBox:
		mode_gbox = QGroupBox("Setting spectral lines display mode:")
		mode_gbox.setObjectName("settingframe")
		mode_gbox_layout = QVBoxLayout()
		mode_gbox_layout.setContentsMargins(10, 20, 20, 20)
		#mode_gbox_layout.setSpacing(20)

	#set mode_gbox_layout into mode_gbox groupBox then set mode_gbox into central_layout
		mode_gbox.setLayout(mode_gbox_layout) 
		central_layout.addWidget(mode_gbox)

	##add spacer into layout:
		mode_gbox_layout.addItem(QSpacerItem(1, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

	#create list of modes in combobox:
		self.color_combo = QComboBox()
		self.colormaps_list = [
			("AUTUMN", cv2.COLORMAP_AUTUMN),
			("BONE", cv2.COLORMAP_BONE),
			("JET", cv2.COLORMAP_JET),
			("WINTER", cv2.COLORMAP_WINTER),
			("RAINBOW", cv2.COLORMAP_RAINBOW),
			("OCEAN", cv2.COLORMAP_OCEAN),
			("SUMMER", cv2.COLORMAP_SUMMER),
			("SPRING", cv2.COLORMAP_SPRING),
			("COOL", cv2.COLORMAP_COOL),
			("HSV", cv2.COLORMAP_HSV),
			("PINK", cv2.COLORMAP_PINK),
			("HOT", cv2.COLORMAP_HOT),
			("PARULA", cv2.COLORMAP_PARULA),
			("MAGMA", cv2.COLORMAP_MAGMA),
			("INFERNO", cv2.COLORMAP_INFERNO),
			("PLASMA", cv2.COLORMAP_PLASMA),
			("VIRIDIS", cv2.COLORMAP_VIRIDIS),
			("CIVIDIS", cv2.COLORMAP_CIVIDIS),
			("CIVIDIS", cv2.COLORMAP_CIVIDIS),
		]
		for name, cmap_id in self.colormaps_list:
			self.color_combo.addItem(name, userData=cmap_id)

	#add comboBox into groupBox:
		mode_gbox_layout.addWidget(self.color_combo, alignment=Qt.AlignTop)

	#End of display mode GroupBox##########################################


	#Rotation horizont GroupBox:
		rot_hor_gbox = QGroupBox("Rotation Horizont Line display:")
		rot_hor_gbox.setObjectName("settingframe")
		rot_hor_gbox_layout = QHBoxLayout()
		rot_hor_gbox_layout.setContentsMargins(10, 40, 20, 20)
		rot_hor_gbox_layout.setSpacing(20)

	#Thickness:
		rot_hor_thick_label = QLabel("Thickness:")
		rot_hor_thick_label.setObjectName("settingText")
		rot_hor_thick_label.setAlignment(Qt.AlignLeft)
		rot_hor_gbox_layout.addWidget(rot_hor_thick_label)

	#Thickness spin box (integer number)
		self.rot_hor_thick_spinbox = QSpinBox()
		self.rot_hor_thick_spinbox.setMinimum(1)	   
		self.rot_hor_thick_spinbox.setMaximum(99)	 
		self.rot_hor_thick_spinbox.setValue(8)		
		self.rot_hor_thick_spinbox.setSingleStep(1)  
		
		rot_hor_gbox_layout.addWidget(self.rot_hor_thick_spinbox, alignment=Qt.AlignTop)

	#Color:
		rot_hor_color_label = QLabel("Color:")
		rot_hor_color_label.setObjectName("settingText")
		rot_hor_color_label.setAlignment(Qt.AlignLeft)
		rot_hor_gbox_layout.addWidget(rot_hor_color_label)	

	#Color Button:
		self.rot_hor_color_button = QPushButton("")
		self.rot_hor_color_button.setObjectName("trans")
		self.rot_hor_color_button.setEnabled(True)
		self.rot_hor_color_button.setFixedHeight(int(self.window_width/10))
		self.rot_hor_color_button.setFixedWidth(int(self.window_width/10))

		#set icon into color btn:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "color_btn.png")
		self.rot_hor_color_button.setIcon(QIcon(icon_path))
		self.rot_hor_color_button.setIconSize(QSize(int(self.window_width/10), int(self.window_width/10)))

		#DO CONNECTION HERE...................................
		#self.self.rot_hor_color_button.clicked.connect(.....)	

		rot_hor_gbox_layout.addWidget(self.rot_hor_color_button, alignment=Qt.AlignTop)

		rot_hor_gbox.setLayout(rot_hor_gbox_layout) 
		central_layout.addWidget(rot_hor_gbox)
	#End of Rotation horizont GroupBox#################################################

		
	#Rotation Degree GroupBox:
		rot_deg_gbox = QGroupBox("Rotation Line display:")
		rot_deg_gbox.setObjectName("settingframe")
		rot_deg_gbox_layout = QHBoxLayout()
		rot_deg_gbox_layout.setContentsMargins(10, 40, 20, 20)
		rot_deg_gbox_layout.setSpacing(20)

	#Thickness:
		rot_deg_thick_label = QLabel("Thickness:")
		rot_deg_thick_label.setObjectName("settingText")
		rot_deg_thick_label.setAlignment(Qt.AlignLeft)
		rot_deg_gbox_layout.addWidget(rot_deg_thick_label)

	#Thickness spin box (integer number)
		self.rot_deg_thick_spinbox = QSpinBox()
		self.rot_deg_thick_spinbox.setMinimum(1)	   
		self.rot_deg_thick_spinbox.setMaximum(99)	 
		self.rot_deg_thick_spinbox.setValue(8)		
		self.rot_deg_thick_spinbox.setSingleStep(1)  
		
		rot_deg_gbox_layout.addWidget(self.rot_deg_thick_spinbox, alignment=Qt.AlignTop)

	#Color:
		rot_deg_color_label = QLabel("Color:")
		rot_deg_color_label.setObjectName("settingText")
		rot_deg_color_label.setAlignment(Qt.AlignLeft)
		rot_deg_gbox_layout.addWidget(rot_deg_color_label)	

	#Color Button:
		self.rot_deg_color_button = QPushButton("")
		self.rot_deg_color_button.setObjectName("trans")
		self.rot_deg_color_button.setEnabled(True)
		self.rot_deg_color_button.setFixedHeight(int(self.window_width/10))
		self.rot_deg_color_button.setFixedWidth(int(self.window_width/10))

		#set icon into color btn:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "color_btn_3.png")
		self.rot_deg_color_button.setIcon(QIcon(icon_path))
		self.rot_deg_color_button.setIconSize(QSize(int(self.window_width/10), int(self.window_width/10)))

		#DO CONNECTION HERE...................................
		#self.self.rot_deg_color_button.clicked.connect(.....)	

		rot_deg_gbox_layout.addWidget(self.rot_deg_color_button, alignment=Qt.AlignTop)

		rot_deg_gbox.setLayout(rot_deg_gbox_layout) 
		central_layout.addWidget(rot_deg_gbox)
	#End of Rotation Degree GroupBox#################################################


	#Crop GroupBox:
		crop_gbox = QGroupBox("Crop Lines display:")
		crop_gbox.setObjectName("settingframe")
		crop_gbox_layout = QHBoxLayout()
		crop_gbox_layout.setContentsMargins(10, 40, 20, 20)
		crop_gbox_layout.setSpacing(20)

	#Thickness:
		crop_thick_label = QLabel("Thickness:")
		crop_thick_label.setObjectName("settingText")
		crop_thick_label.setAlignment(Qt.AlignLeft)
		crop_gbox_layout.addWidget(crop_thick_label)

	#Thickness spin box (integer number)
		self.crop_thick_spinbox = QSpinBox()
		self.crop_thick_spinbox.setMinimum(1)	   
		self.crop_thick_spinbox.setMaximum(99)	 
		self.crop_thick_spinbox.setValue(8)		
		self.crop_thick_spinbox.setSingleStep(1)  
		
		crop_gbox_layout.addWidget(self.crop_thick_spinbox, alignment=Qt.AlignTop)

	#Color:
		crop_color_label = QLabel("Color:")
		crop_color_label.setObjectName("settingText")
		crop_color_label.setAlignment(Qt.AlignLeft)
		crop_gbox_layout.addWidget(crop_color_label)	

	#Color Button:
		self.crop_color_button = QPushButton("")
		self.crop_color_button.setObjectName("trans")
		self.crop_color_button.setEnabled(True)
		self.crop_color_button.setFixedHeight(int(self.window_width/10))
		self.crop_color_button.setFixedWidth(int(self.window_width/10))

		#set icon into color btn:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "color_btn_4.png")
		self.crop_color_button.setIcon(QIcon(icon_path))
		self.crop_color_button.setIconSize(QSize(int(self.window_width/10), int(self.window_width/10)))

		#DO CONNECTION HERE...................................
		#self.self.crop_color_button.clicked.connect(.....)	

		crop_gbox_layout.addWidget(self.crop_color_button, alignment=Qt.AlignTop)

		crop_gbox.setLayout(crop_gbox_layout) 
		central_layout.addWidget(crop_gbox)
	#End of Rotation Degree GroupBox#################################################

		#CONNECTION:
		self.rot_hor_color_button.clicked.connect(lambda: self.choose_color(self.rot_hor_color_button))
		self.rot_deg_color_button.clicked.connect(lambda: self.choose_color(self.rot_deg_color_button))
		self.crop_color_button.clicked.connect(lambda: self.choose_color(self.crop_color_button))

		self.rot_hor_color_button.selected_color = QColor("#00008B")
		self.rot_deg_color_button.selected_color = QColor("#00FF00")
		self.crop_color_button.selected_color = QColor("#0000FF")

		self.btn_crop_ok = QPushButton("Crop") ###################################
		self.btn_crop_ok.setObjectName("cropper_browse_btn") #######################
		self.btn_crop_ok.setMinimumHeight(int(screen_height/25)) ######################################
		self.btn_crop_ok.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		self.ok_button = QPushButton("Apply")
		self.ok_button.setObjectName("cropper_browse_btn") 
		self.ok_button.setFixedSize(int(self.window_width/4),int(screen_height/30))

		self.ok_button.clicked.connect(self.on_ok_clicked)

		central_layout.addWidget(self.ok_button,alignment=Qt.AlignRight)

		self.adjustSize()
		self.setMinimumSize(self.size())


	def choose_color(self, button):
		color = QColorDialog.getColor()
		if color.isValid():
			button.setStyleSheet(f"background-color: {color.name()};")
			button.selected_color = color  # Можно сохранить цвет прямо в кнопку

	def on_ok_clicked(self):
		# Получаем индекс и значение (ID colormap) из ComboBox:
		selected_index = self.color_combo.currentIndex()
		cmap_id = self.color_combo.itemData(selected_index)

		hor_color = (self.rot_hor_color_button.selected_color.red(), 
			   self.rot_hor_color_button.selected_color.green(),
			  self.rot_hor_color_button.selected_color.blue())

		deg_color = (self.rot_deg_color_button.selected_color.red(), 
			   self.rot_deg_color_button.selected_color.green(),
			  self.rot_deg_color_button.selected_color.blue())

		crop_color = (self.crop_color_button.selected_color.red(), 
			   self.crop_color_button.selected_color.green(),
			  self.crop_color_button.selected_color.blue())

		self.Model.set_HorThickness(self.rot_hor_thick_spinbox.value())
		self.Model.set_HorColor(hor_color)
		self.Model.set_DegThickness(self.rot_deg_thick_spinbox.value())
		self.Model.set_DegColor(deg_color)
		self.Model.set_CropThickness(self.crop_thick_spinbox.value())
		self.Model.set_CropColor(crop_color)


		self.Model.set_ColorMap(cmap_id)
		self.Model.ApplyLineStyle()

		self.close()
		

#___________________________________________
