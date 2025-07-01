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

#ADDING COMPONENTS OUT OF ROOT_________________________________________________
import sys
import os
# Add parent folder path sys.path:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mp_viewer_app import*

# Add parent folder path sys.path:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ArxSR import* 

# Add parent folder path sys.path:
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from GUIcomp import* 
###############################################################################


class model_align(object):

#Initializer:
	def __init__(self,parent=None):
		super().__init__()
        
	#QMainWindow as parent:
		self.main_parent = parent

	#The sizes of MainWindow:
		self.win_w = parent.window_width
		self.win_h = parent.window_height

	#data holder:
		self.arx_data = None#####
		self.data = None ########
		self.data_header = None##
		self.data_editor = None##
		self.isImgLoaded = None##
		self.prev_spectrum = None

	#ui_components:
		self.__left = Left(self)
		self.__right= Right(self)

	#show mode of picture:
		self.colorMap = cv2.COLORMAP_VIRIDIS

	#Line designs params:
		self.lineThickness = 6
		self.lineColor = (255,79,0)

	#Line control value:
		self.upValue = 0
		self.downValue = 0

	#message box:
		self.msg = QMessageBox()
		self.msg.setWindowTitle("Error:")
		self.msg.setStandardButtons(QMessageBox.Ok)

		self.prev = None
#End of Initializer................................


#SETTERS:
#set MainWindow as parent for dialog window:
	def set_dialog_parent(self, parent):
		self.main_parent = parent#######	
#...........................................

#color of picture:
	def setColorMap(self, colorMap):
		self.colorMap = colorMap####
		self.ImgReLoad()# update####
#...................................

#line thickness:
	def setLineThickness(self, thickness):
		self.lineThickness = thickness####
#.........................................

#color of line:
	def setLineColor(self, color):
		self.lineColor = color####
#.................................

#set y-value of upper line from left:
	def setUpLineValue(self, value):
		if self.isImgLoaded is True:
			self.upValue = value####
			self.__left.PaintLines(self.upValue, self.downValue)
#...............................................................

#set y-value of down line from left:
	def setDownLineValue(self,value):
		if self.isImgLoaded  is True:
			self.downValue = value##
			self.__left.PaintLines(self.upValue, self.downValue)
#...............................................................


#GETTERS:
#get instances Left widget:
	def getLeftWidget(self):
		return self.__left
#...........................

#get instances right widget:
	def getRightLayout(self):
		return self.__right
#............................

#get mainwindow instance:
	def getMainParent(self):
		return self.main_parent
#..............................


#METHODS:
	def ImgLoad(self):
		
		try:
		#get file path:		
			self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_parent, 'Single File', QtCore.QDir.rootPath() , '(*.fits *.fit)')
			
		#If user canceled dialog:
			if not self.fileName:
				print("Err: fits file has not been choosen")
			try:
				self.arx_data = ArxData(self.fileName)
			#save data from file as data:
				self.data = self.arx_data.get_data() ########
				self.data_header = self.arx_data.get_header()
			#Checking:
				self.isImgLoaded = True
				#############################################
				print(f"data header: {self.data_header}")
				print(f"fits data: {self.data}")#########
			except Exception as err:
				self.isImgLoaded = False
				print(f"err in reading {self.fileName}")
				return
			
			#show png in the label not data:
			self.__left.imgLoad() ######
			self.__right.reset_sliders()
			
		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
			return

#To Align spectrum
	def alignData(self):

		#TOP VALUE IS - self.upValue; DOWN VALUES IS - self.downValue:

		#Check data for existing:
		if self.arx_data is None:
			print("EMPTY DATA TO ALIGN")
			return

		#Create handler to processing:
		order = self.__right.orderSpinBox.value()#######
		self.data_editor = ArxSpectEditor(self.arx_data)
		################################################

		#Save previous data to back:
		self.prev_spectrum = self.arx_data

		#up/down params to distorsion:
		h = self.__left.getHeight()#######
		up = int(h*self.upValue/100)######
		down = h-int(h*self.downValue/100)
		##################################

		#Obtain new data after distorsion:
		self.arx_data = self.data_editor.SDistorsionCorr(up, down, order)
		self.data = self.arx_data.get_data()#############################

		#Show new data:
		self.ImgReLoad()
		self.__right.reset_sliders()

#End of Align spectrum...................................................

#Back to spectrum:
	def backToSpectrum(self):
		if self.prev_spectrum is None:
			return

		if self.isImgLoaded is True:
			self.arx_data = self.prev_spectrum ##########
			self.data = self.arx_data.get_data() ########
			self.data_header = self.arx_data.get_header()

			#Checking:
			self.isImgLoaded = True ####
			self.__left.imgLoad()#######
			self.__right.reset_sliders()
#........................................................


#Re Load Iamge not Data:
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


#Polynom loader:
	def polynomLoad(self):
		try:
		#get file path:		
			self.path, _ = QtWidgets.QFileDialog.getOpenFileName(self.main_parent, 'Single File', QtCore.QDir.rootPath() , '(*.csv)')
			print(self.path)

			# Check if user selected a file
			if not self.path:
				print("No file selected.")
				return

			polynoms = CsvPolynom.upLoadCsv(self.path)

			dialog = PolynomDialog(self.main_parent, self)
			dialog.setPolynoms(polynoms)
			
			target_date_str = self.data_header["DATE-OBS"]
			print(target_date_str)

			theBestPoly = CsvPolynom.findNearestDate(polynoms, target_date_str)
			dialog.setTheBestDate(theBestPoly)
			dialog.show()
			for p in polynoms:
				print(p)

		#run dialog to select the csvPolynom:
			#dialog = SettingDialog(self.main_parent, self)
			#dialog.show()

			
			

		#set green message of date:
			self.__right.polyProgress.setValue(1)
			self.__right.setPolyTextMode(1, "25.05.2015")
		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
			return
#........................................................

#run setting dialog:
	def runSetting(self):
		dialog = SettingDialog(self.main_parent, self)
		dialog.show()
#.....................................................

	def setAppliedPolynom(self,polynom):
		print("______________________________________")
		print(polynom)
		print(1)
		print(self.arx_data.get_data())
		editor = ArxSpectEditor(self.arx_data)
		print(editor)
		new_data = self.arx_data
		new_data = editor.OptDen2Int(polynom)
		#print(f"new_data = {new_data}")
		self.arx_data.set_data(new_data.get_data())
		print(3)
		#print(self.arx_data)
		self.__left.imgLoad()
		print(4)


	def save(self):
		print("going to save fits file")
		msg = QMessageBox()
		print(f"request to save fits file")


		try:
			print("001")
			data = self.arx_data.get_data()
			print("002")
			header = self.arx_data.get_header()
			print("003")

			if data is None or header is None:
				print("004")
				msg.setWindowTitle("Error:")
				msg.setText("data or header is None!!!")
				msg.setStandardButtons(QMessageBox.Ok)
				msg.exec_()
				return
			print("005")
			# Get file name without path and extensions:
			base_name = os.path.splitext(os.path.basename(self.fileName))[0]
			print("006")
			# Add data time:
			today_str = datetime.today().strftime('%Y-%m-%d')
			print("007")
			suggested_name = f"{base_name}_{today_str}.fits"
			print("008")
			# Open Dialog "Save as"
			save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
				self.main_parent,
				"Save FITS File As",
				QtCore.QDir.homePath() + "/" + suggested_name,
				"FITS files (*.fits *.fit)"
			)
			print("009")
			if save_path:
				# Extension check:
				print("010")
				if not save_path.lower().endswith(('.fits', '.fit')):
					save_path += ".fits"
				print("011")
				#self.arx_data.save_as_fits(save_path)

				hdu = fits.PrimaryHDU(data, header=header)
				print("012")
				hdulist = fits.HDUList([hdu])
				print("013")
				hdulist.writeto(save_path, overwrite=True)
				print("014")
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
		pass
#End of model ###############################################



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
		
		self.model = parent
		
		#w,h - the size of main QLabel:
		self.w = self.width() #####
		self.h = self.height() ####

		self.__img_w = 0
		self.__img_h = 0

		self.loaded_image = None
		self.tmp_image = None

		self.lineThickness = 6
		self.lineColor = (255,79,0)
		

		#curr row for the animation:
		self.current_row = 0 ###########################
	#timer for rendering:
		self.timer = QTimer() ##########################
		self.timer.timeout.connect(self.updateAnimation)  
	#to hold partial img:
		self.partial_image = None ######################


	#End of Initializer..............................................................

#GETTERS:
#get height of loaded image:
	def getHeight(self):
		return self.__img_h
#..........................

	def getWidth(self):
		return self.__img_w
#..........................

#SETTERS:
#set color & thickness of Line:
	def setLineStyle(self, lineThick, lineColor):
		self.lineThickness = lineThick #########
		self.lineColor = lineColor #############
	#...........................................

#re-set image:
	def set_reset(self):
	#get copy of origin image:
		self.tmp_image = self.loaded_image.copy()#
		self.update() ############################
		return self.w ############################
	#.............................................


#Load image with cv2:
	def imgLoad(self):
		try:
		#read with cv2:
			self.loaded_image = self.model.arx_data.get_image(self.model.colorMap)
		#animation:
			try:
				self.startAnimation()
			except Exception as err: 
				self.timer.stop() ###############
				self.set_reset() ################
				self.setImage(self.loaded_image)#
		#set img at the end:
			self.set_reset() ####################
			self.setImage(self.loaded_image) ####
		except Exception as err: 
			print(f"Unexpected {err=}, {type(err)=}")	
	#................................................

#img Load animation:
	def startAnimation(self): 
		self.current_row = 0 ################################
		self.partial_image = np.zeros_like(self.loaded_image)  
		self.timer.start(1) #################################
	#........................................................

#update animated img:
	def updateAnimation(self):
		if self.current_row < self.partial_image.shape[0]:
			self.partial_image[self.current_row, :] = self.loaded_image[self.current_row, :]
			self.setImage(self.partial_image) ##############################################
			self.current_row += 1 ##########################################################
		else:
			self.timer.stop() ##############################################################
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

#Paint Lines on the Left:
	def PaintLines(self, up, down):
		#Here
		color = self.lineColor
		
		color_td = (0,0,255)
		
		#Line thickness of 9 px
		thickness = self.lineThickness

		self.__img_h = self.loaded_image.shape[0]
		self.__img_w = self.loaded_image.shape[1]

		h = self.__img_h
		w = self.__img_w 
		
		if( int(h*up/100) < (h - int(h*down/100))):
			color_td = color
		
		top_point1 = (0, int(h*up/100))
		top_point2 = (w, int(h*up/100))
		
		dwn_point1 = (0, h - int(h*down/100))
		dwn_point2 = (w, h - int(h*down/100))
		
		#Cv draw algorithm:
		self.tmp_image = self.loaded_image.copy()
		
		self.tmp_image = cv2.line(self.tmp_image, top_point1, top_point2, color_td, thickness)
		self.tmp_image = cv2.line(self.tmp_image, dwn_point1, dwn_point2, color_td, thickness)
		
		self.update()
#.............................................................................................

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
		browseQLabel = QLabel() ################################
		browseQLabel.setAlignment(Qt.AlignCenter) ##############
		#перенос текста ########################################
		browseQLabel.setWordWrap(True) #########################
		browseQLabel.setObjectName("cropper_right_browse_label") 
	#Size of dark label ########################################
		#browseQLabel.setFixedSize(self.w/5.3, self.h/16) #######
		browseQLabel.setFixedSize(int(self.w/5.3), int(self.h/16))
		########################################################

	#Mian Big Text on the Label:
		browseText = """
            <p style="text-align: justify;"><b>
			Load the spectrum to start aligning</b></p>
            """
		browseTxtQlabel = QLabel(browseText, browseQLabel)
		#####################################################

	#Layout to hold Main Big text and button:
		layout = QVBoxLayout() ##############
		layout.setSpacing(7) ################
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop) ####
		
	#label for the text:
		browseTxtQlabel.setWordWrap(True) ##################
		browseTxtQlabel.setAlignment(Qt.AlignCenter) #######
		browseTxtQlabel.setObjectName("cropper_browse_text")
		#MAC CHANGE					
		#browseTxtQlabel.setFixedSize(self.w/6, self.h/44)
		browseTxtQlabel.setFixedSize(int(self.w/6), int(self.h/44))###
		######################################################

	#Button to browse:
		self.browseQbtn = QPushButton("Browse") ###########
		self.browseQbtn.setObjectName("cropper_browse_btn")
		self.browseQbtn.setEnabled(True) ##################
		#MAC CHANGE	
		#self.browseQbtn.setFixedSize(self.w/12, self.h/38)
		self.browseQbtn.setFixedSize(int(self.w/12), int(self.h/38))#

	#Set btn&text(lbl) into layout then set it into browse QLbl:
		browseQLabel.setLayout(layout) #########################
		layout.addWidget(browseTxtQlabel, 10, Qt.AlignHCenter) 
		layout.addWidget(self.browseQbtn, 0, Qt.AlignHCenter) 

	#Set browse label into Main QVBoxLayout:
		self.addWidget(browseQLabel, alignment=Qt.AlignCenter)
	#   %%%%%%%%%%%%%%   END OF BROWSING   %%%%%%%%%%%%%%%%%%%%%
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	#	%%%%%%%%%%%%%%
	#	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

		limitsQLabel = QLabel() ################################
		limitsQLabel.setAlignment(Qt.AlignCenter) ##############
		#перенос текста ########################################
		limitsQLabel.setWordWrap(True) #########################
		limitsQLabel.setObjectName("cropper_right_browse_label") 
	#Size of dark label ########################################
		#MAC CHANGE
		#limitsQLabel.setFixedSize(self.w/5.3, self.h/4)	
		limitsQLabel.setFixedSize(int(self.w/5.3), int(self.h/4)) #######
		########################################################

	#Mian Big Text on the Label:
		limitsText = """
            <p style="text-align: justify;"><b>
			Up and Down limits</b></p>
            """
		limitsTxtQlabel = QLabel(limitsText, limitsQLabel)
		#####################################################

	#Layout to hold Main Big text and button:
		layout = QVBoxLayout() ##############
		layout.setSpacing(7) ################
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop) ####
		
	#label for the text:
		limitsTxtQlabel.setWordWrap(True) ##################
		limitsTxtQlabel.setAlignment(Qt.AlignCenter) #######
		limitsTxtQlabel.setObjectName("cropper_browse_text")
		#MAC CHANGE
		#limitsTxtQlabel.setFixedSize(self.w/6, self.h/44)
		limitsTxtQlabel.setFixedSize(int(self.w/6), int(self.h/44))###
		######################################################

	#Up & Down QLabeled-Sliders:
		self.upQSlider = QLabeledSlider("""<p style="text-align:left;">
		<b>Up Line position:</b></p>""") ################################
		self.downQSlider = QLabeledSlider("""<p style="text-align:left;">
		<b>Down Line Position:</b></p>""") ##############################

	# 2. Set ranges for 2 sliders:
		self.upQSlider.set_slider_range(0,100)###########
		self.downQSlider.set_slider_range(0,100)#########
		#################################################

	#Set txt(lbl),2-sliders into layout
	#then set it into limits Qlabel:
		layout.addWidget(limitsTxtQlabel, 10, Qt.AlignHCenter)## 
		layout.addWidget(self.upQSlider) #######################
		layout.addWidget(self.downQSlider) #####################
		########################################################

	#Order Horizontal Layout%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#Horizontal Layout of limits order:
		hlayout = QHBoxLayout()############
		hlayout.setContentsMargins(0,0,0,0) 
		hlayout.setSpacing(15)#############
		###################################

	#Label for ward "Order":
		orderTxtQLabel = QLabel("Alignment Polynom Order:") #######
		orderTxtQLabel.setAlignment(Qt.AlignCenter) ###############
		orderTxtQLabel.setObjectName("cropper_right_browse_label")# 
		hlayout.addWidget(orderTxtQLabel)##########################
		###########################################################

	#Polynom Order spin box:
		self.orderSpinBox = QSpinBox()#######
		self.orderSpinBox.setMinimum(2)######     
		self.orderSpinBox.setMaximum(21)#####   
		self.orderSpinBox.setValue(2)########      
		self.orderSpinBox.setSingleStep(1)### 
		hlayout.addWidget(self.orderSpinBox)#
		#####################################

	#Set horizontal layout into container:
		container = QWidget() ###############
		container.setLayout(hlayout)#########
		#####################################

	#Set container into vertical layout:
		layout.addWidget(container, alignment=Qt.AlignLeft)###
	#Set vertical label into limits qlabel:
		limitsQLabel.setLayout(layout) #######################
	#Set limits main-label into Main QVBoxLayout:
		self.addWidget(limitsQLabel, alignment=Qt.AlignCenter)

	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#    %%%%%%%%%  ALIGHN BUTTONS AREA  %%%%%%%%%%%%%%%%%%%%%%:	
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#This is label to set Layout with widgets (text & btn):
		alignQLabel = QLabel() ################################
		alignQLabel.setAlignment(Qt.AlignCenter) ##############
		#перенос текста ########################################
		alignQLabel.setWordWrap(True) #########################
		alignQLabel.setObjectName("cropper_right_browse_label") 
	#Size of dark label ########################################
		#MAC CHANGE
		#alignQLabel.setFixedSize(self.w/5.3, self.h/24)#
		alignQLabel.setFixedSize(int(self.w/5.3), int(self.h/24)) #######
		########################################################


	#Layout to hold Main Big text and button:
		layout = QHBoxLayout() ##############
		layout.setSpacing(15) ################
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop) ####
		
	#Button to Align:
		self.backQbtn = QPushButton("Back") ###########
		self.backQbtn.setObjectName("cropper_browse_btn")
		self.backQbtn.setEnabled(True) ##################
		#MAC CHANGE
		#self.backQbtn.setFixedSize(self.w/14, self.h/38)
		self.backQbtn.setFixedSize(int(self.w/14), int(self.h/38))#

	#Button to Align:
		self.alignQbtn = QPushButton("Align") ###########
		self.alignQbtn.setObjectName("cropper_browse_btn")
		self.alignQbtn.setEnabled(True) ##################
		#MAC CHANGE
		#self.alignQbtn.setFixedSize(self.w/14, self.h/38)#
		self.alignQbtn.setFixedSize(int(self.w/14), int(self.h/38))#


	#Set btn&text(lbl) into layout then set it into browse QLbl:
		alignQLabel.setLayout(layout) #########################

		layout.addWidget(self.backQbtn, 0, Qt.AlignHCenter) 
		layout.addWidget(self.alignQbtn, 0, Qt.AlignHCenter)
		
	#Set browse label into Main QVBoxLayout:
		self.addWidget(alignQLabel, alignment=Qt.AlignCenter)
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	#    %%%%%%%%%  POLYNOM BROWSING(CSV)  %%%%%%%%%%%%%%%%%%%%%:	
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#This is label to set Layout with widgets (text & btn):
		polyBrowseQLabel = QLabel() ################################
		polyBrowseQLabel.setAlignment(Qt.AlignCenter) ##############
		#перенос текста ############################################
		polyBrowseQLabel.setWordWrap(True) #########################
		polyBrowseQLabel.setObjectName("cropper_right_browse_label") 
	#Size of dark label ############################################
		#MAC CHANGE
		#	polyBrowseQLabel.setFixedSize(self.w/5.3, self.h/8) 	
		polyBrowseQLabel.setFixedSize(int(self.w/5.3), int(self.h/8)) ########
		############################################################

	#Mian Big Text on the Label:
		polyBrowseText = """
            <p style="text-align: justify;"><b>
			Load the spectrum to start aligning</b></p>
            """
		polyBrowseTxtQlabel = QLabel(polyBrowseText, polyBrowseQLabel)
		##############################################################

	#Layout to hold Main Big text and button:
		layout = QVBoxLayout() ##############
		layout.setSpacing(15) ################
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop) ####
		
	#label for the text:
		polyBrowseTxtQlabel.setWordWrap(True) ##################
		polyBrowseTxtQlabel.setAlignment(Qt.AlignCenter) #######
		polyBrowseTxtQlabel.setObjectName("cropper_browse_text")
		#MAC CHANGE
		#polyBrowseTxtQlabel.setFixedSize(self.w/6, self.h/44)
		polyBrowseTxtQlabel.setFixedSize(int(self.w/6), int(self.h/44))###
		########################################################

	#Button to browse:
		self.polyBrowseQbtn = QPushButton("Upload polynom") ###
		self.polyBrowseQbtn.setObjectName("cropper_browse_btn")
		self.polyBrowseQbtn.setEnabled(True) ##################
		#MAC CHANGE
		#self.polyBrowseQbtn.setFixedSize(self.w/12, self.h/38)
		self.polyBrowseQbtn.setFixedSize(int(self.w/12), int(self.h/38))#

	#ProgressBar to show upload of polynom:
		self.polyProgress = QProgressBar() ####################
		self.polyProgress.setAlignment(Qt.AlignCenter) ########
		self.polyProgress.setTextVisible(False)################
		self.polyProgress.setEnabled(True)#####################
		self.polyProgress.setMinimum(0) #######################
		self.polyProgress.setMaximum(1) #######################
		self.polyProgress.setValue(0) #########################
		#MAC CHANGE
		#self.polyProgress.setFixedSize(self.w/5.5, self.h/250)
		self.polyProgress.setFixedSize(int(self.w/5.5), int(self.h/250))#
		#######################################################

	#Text of loading polynom:
		self.polyLoadText = QLabel()#############
		self.setPolyTextMode(0)
		#####################################################
		
	#Set btn&text(lbl) into layout then set it into browse QLbl:
		polyBrowseQLabel.setLayout(layout) ######################
		layout.addWidget(polyBrowseTxtQlabel,10, Qt.AlignHCenter) 
		layout.addWidget(self.polyBrowseQbtn, 0, Qt.AlignHCenter) 
		layout.addWidget(self.polyProgress, 0 ,  Qt.AlignHCenter) 
		layout.addWidget(self.polyLoadText, 0 ,  Qt.AlignHCenter)
	#Set browse label into Main QVBoxLayout:
	
		self.addWidget(polyBrowseQLabel, alignment=Qt.AlignCenter)
	#   %%%%%%%%%%%%%%   END OF BROWSING   %%%%%%%%%%%%%%%%%%%%%%%
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	#    %%%%%%%%%  SAVE BUTTONS AREA  %%%%%%%%%%%%%%%%%%%%%%:	
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#This is label to set Layout with widgets (text & btn):
		saveQLabel = QLabel() ################################
		saveQLabel.setAlignment(Qt.AlignCenter) ##############
		#перенос текста ########################################
		saveQLabel.setWordWrap(True) #########################
		saveQLabel.setObjectName("cropper_right_browse_label") 
	#Size of dark label ########################################
		#MAC CHANGE
		#saveQLabel.setFixedSize(self.w/5.3, self.h/24) #
		saveQLabel.setFixedSize(int(self.w/5.3), int(self.h/24)) #######
		########################################################


	#Layout to hold buttons:
		layout = QHBoxLayout() ##############
		layout.setSpacing(15) ################
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop) ####
		
	#Button to Cancel:
		self.cancelQbtn = QPushButton("Cancel") ###########
		self.cancelQbtn.setObjectName("cropper_browse_btn")
		self.cancelQbtn.setEnabled(True) ##################
		#MAC CHANGE
		#self.cancelQbtn.setFixedSize(self.w/14, self.h/38)
		self.cancelQbtn.setFixedSize(int(self.w/14), int(self.h/38))#

	#Button to save:
		self.saveQbtn = QPushButton("Save") #############
		self.saveQbtn.setObjectName("cropper_browse_btn")
		self.saveQbtn.setEnabled(True) ##################
		#MAC CHANGE
		#self.saveQbtn.setFixedSize(self.w/14, self.h/38)
		self.saveQbtn.setFixedSize(int(self.w/14), int(self.h/38))#

	#Set btn&text(lbl) into layout then set it into browse QLbl:
		saveQLabel.setLayout(layout) #########################

		layout.addWidget(self.cancelQbtn, 0, Qt.AlignHCenter) 
		layout.addWidget(self.saveQbtn, 0, Qt.AlignHCenter)
		
	#Set browse label into Main QVBoxLayout:
		self.addWidget(saveQLabel, alignment=Qt.AlignCenter)
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#    %%%%%%%%%  THICKNESS & COLOR AREA  %%%%%%%%%%%%%%%%%%%:	
	#    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	#This is label to set Layout with widgets (text & btn):
		SettingQLabel = QLabel() ################################
		SettingQLabel.setAlignment(Qt.AlignCenter) ##############
		SettingQLabel.setWordWrap(True) #########################
		SettingQLabel.setObjectName("cropper_right_browse_label") 
		#MAC CHANGE
		#SettingQLabel.setFixedSize(self.w/5.3, self.h/24)
		SettingQLabel.setFixedSize(int(self.w/5.3), int(self.h/24)) #######
		#########################################################

	#text above the button:
		SettingText = """
    <p style="text-align: justify;"><b>Tool View Settings:</b></p>
		"""
		SettingTextQLbl = QLabel(SettingText, SettingQLabel)
		SettingTextQLbl.setWordWrap(True) ##################
		SettingTextQLbl.setAlignment(Qt.AlignCenter) #######
		SettingTextQLbl.setObjectName("cropper_browse_text")
		#MAC CHANGE
		#SettingTextQLbl.setFixedHeight(self.h/44)
		SettingTextQLbl.setFixedHeight(int(self.h/44)) ##########
		####################################################

	#Setting button:
		self.SettingQbtn = QPushButton("")#####
		self.SettingQbtn.setObjectName("trans")
		self.SettingQbtn.setEnabled(True)#####
		#MAC CHANGE
		#self.SettingQbtn.setFixedHeight(self.h/30)####
		#self.SettingQbtn.setFixedWidth(self.w/5.3/6)####
		self.SettingQbtn.setFixedHeight(int(self.h/30))####
		self.SettingQbtn.setFixedWidth(int(self.w/5.3/6))####
		# -------------------------------------
		
	#Setting icons into button:
		script_dir = os.path.dirname(os.path.abspath(__file__))##
		icon_path = os.path.join(script_dir, "color_setting.png")
		self.SettingQbtn.setIcon(QIcon(icon_path))###############
		self.SettingQbtn.setIconSize(QSize(300, 65))#############
		#########################################################

	#Connection run:
		self.SettingQbtn.clicked.connect(self.model.runSetting)

	#Empty button:
		EmptyQbtn = QPushButton("")########
		EmptyQbtn.setObjectName("trans")###
		EmptyQbtn.setEnabled(False)########
		#MAC CHANGE
		#EmptyQbtn.setFixedHeight(self.h/30)
		EmptyQbtn.setFixedHeight(int(self.h/30))
		###################################
		
	#Holder layout:
		layout = QHBoxLayout() ###################
		layout.setSpacing(10) ####################
		layout.setContentsMargins(0,0,0,0) #######
		layout.setAlignment(Qt.AlignCenter)#######
		##########################################
		layout.addWidget(SettingTextQLbl, 10)  
		layout.addWidget(self.SettingQbtn, 10)####  
		layout.addWidget(EmptyQbtn, 5) ###########
		##########################################
		SettingQLabel.setLayout(layout)
		# ----------------------------------------

		self.addWidget(SettingQLabel, alignment=Qt.AlignCenter)
		#######################################################


#End of Ui components....................................................


	def InitConnects(self):
		#pass
		#CONNECTION OF - browse_button:====================
		self.browseQbtn.clicked.connect(self.model.ImgLoad)
		#==================================================

		#CONNECTION OF - Up & Down sliders:=================================================
		self.upQSlider.valueChanged.connect(lambda val: self.model.setUpLineValue(val))#####
		self.downQSlider.valueChanged.connect(lambda val: self.model.setDownLineValue(val))#
		#===================================================================================

		#CONNECTION OF - align button:======================
		self.alignQbtn.clicked.connect(self.model.alignData)
		#===================================================

		#CONNECTION OF - back button:===========================
		self.backQbtn.clicked.connect(self.model.backToSpectrum)
		#=======================================================

		#CONNECTION OF - polynom browse QButton:====================
		self.polyBrowseQbtn.clicked.connect(self.model.polynomLoad)
		#===========================================================

		#CONNECTION OF - polynom browse QButton:=========================
		#self.polyProgress.valueChanged.connect(self.update_poly_progress)
		#================================================================

		#CONNECTION OF - polynom browse QButton:=========================
		self.saveQbtn.clicked.connect(self.model.save)
		#================================================================
#........................................................................

#Clear sliders position:
	def reset_sliders(self):
		self.downQSlider.reset()
		self.upQSlider.reset()
#................................

	def setPolyTextMode(self, mode, txt = None):
		if mode == 0:
			self.polyLoadText.setObjectName("polyLblNotloaded")###
			self.polyLoadText.style().unpolish(self.polyLoadText)#
			self.polyLoadText.style().polish(self.polyLoadText)###
			self.polyLoadText.setText("No Polinomial!")###########
		elif mode == 1 and txt is not None:
			self.polyLoadText.setObjectName("polyLblLoaded")######
			self.polyLoadText.style().unpolish(self.polyLoadText)#
			self.polyLoadText.style().polish(self.polyLoadText)###
			self.polyLoadText.setText("Polynomial on " + str(txt))
		else: 
			self.polyLoadText.setObjectName("polyLblNotloaded")###
			self.polyLoadText.style().unpolish(self.polyLoadText)#
			self.polyLoadText.style().polish(self.polyLoadText)###
			self.polyLoadText.setText("Error status")#############
#.................................................................


#End of Right ########################################################




#+++++++++++++++++++++++++++++ADDITIONA MAIN PACKEDGE++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#+++++++++++++++++++++++++++++ADDITIONA MAIN CLASSES++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#THICKNESS & COLOR DIALOG WINDOW:
class SettingDialog(QMainWindow):

	def __init__(self, mainparent=None, model=None):
		super().__init__(mainparent)

	#To set params in model:
		self.Model = model

	#set icons:
		self.setWindowTitle("View settings")#####################
		script_dir = os.path.dirname(os.path.abspath(__file__))##
		icon_path = os.path.join(script_dir, "color_setting.png")
		self.setWindowIcon(QIcon(icon_path))#####################

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
		self.window_height = int((screen_height) * 0.3)### 
		self.resize(self.window_width,self.window_height)#
	######################################################


	#Central widget and it's Layout:
		central_widget = QWidget() ######################
		central_layout = QVBoxLayout() ##################
		central_layout.setContentsMargins(10, 10, 10, 10)

		central_widget.setLayout(central_layout) ########
		self.setCentralWidget(central_widget) ###########
		#################################################


	#Spec Display mode GroupBox:
		mode_gbox = QGroupBox("Setting spectral lines display mode:")
		mode_gbox.setObjectName("settingframe")
		mode_gbox_layout = QVBoxLayout()
		mode_gbox_layout.setContentsMargins(10, 20, 20, 20)

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
		]
		for name, cmap_id in self.colormaps_list:
			self.color_combo.addItem(name, userData=cmap_id)

	#add comboBox into groupBox:
		mode_gbox_layout.addWidget(self.color_combo, alignment=Qt.AlignTop)

	#End of display mode GroupBox##########################################
		

	#Align GroupBox:
		align_gbox = QGroupBox("Align Lines display:")
		align_gbox.setObjectName("settingframe")
		align_gbox_layout = QHBoxLayout()
		align_gbox_layout.setContentsMargins(10, 40, 20, 20)
		align_gbox_layout.setSpacing(20)

	#Thickness:
		align_thick_label = QLabel("Thickness:")
		align_thick_label.setObjectName("settingText")
		align_thick_label.setAlignment(Qt.AlignLeft)
		align_gbox_layout.addWidget(align_thick_label)

	#Thickness spin box (integer number)
		self.align_thick_spinbox = QSpinBox()
		self.align_thick_spinbox.setMinimum(1)       
		self.align_thick_spinbox.setMaximum(99)     
		self.align_thick_spinbox.setValue(8)        
		self.align_thick_spinbox.setSingleStep(1)  
		
		align_gbox_layout.addWidget(self.align_thick_spinbox, alignment=Qt.AlignTop)

	#Color:
		align_color_label = QLabel("Color:")
		align_color_label.setObjectName("settingText")
		align_color_label.setAlignment(Qt.AlignLeft)
		align_gbox_layout.addWidget(align_color_label)	

	#Color Button:
		self.align_color_button = QPushButton("")
		self.align_color_button.setObjectName("trans")
		self.align_color_button.setEnabled(True)
		self.align_color_button.setFixedHeight(60)
		self.align_color_button.setFixedWidth(80)

		#set icon into color btn:
		script_dir = os.path.dirname(os.path.abspath(__file__))
		icon_path = os.path.join(script_dir, "color_btn_4.png")
		self.align_color_button.setIcon(QIcon(icon_path))
		self.align_color_button.setIconSize(QSize(80, 55))

		align_gbox_layout.addWidget(self.align_color_button, alignment=Qt.AlignTop)

		align_gbox.setLayout(align_gbox_layout)  
		central_layout.addWidget(align_gbox)
	#End of Align GroupBox#################################################

		self.btn_align_ok = QPushButton("Align") ###################################
		self.btn_align_ok.setObjectName("cropper_browse_btn") #######################
		self.btn_align_ok.setMinimumHeight(70) ######################################
		self.btn_align_ok.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

		self.ok_button = QPushButton("Apply")
		self.ok_button.setObjectName("cropper_browse_btn") 
		self.ok_button.setFixedSize(180,65)
		
		central_layout.addWidget(self.ok_button,alignment=Qt.AlignRight)

		self.adjustSize()
		self.setMinimumSize(self.size())

	#CONNECTIONS:
		self.align_color_button.clicked.connect(lambda: self.choose_color(self.align_color_button))
		self.align_color_button.selected_color = QColor("#0000FF")
		self.ok_button.clicked.connect(self.on_ok_clicked)
#End of Init.......................................................................................
		


	def choose_color(self, button):
		color = QColorDialog.getColor()
		if color.isValid():
			button.setStyleSheet(f"background-color: {color.name()};")
			button.selected_color = color  # Можно сохранить цвет прямо в кнопку

	def on_ok_clicked(self):
	    # Получаем индекс и значение (ID colormap) из ComboBox:
		selected_index = self.color_combo.currentIndex()
		cmap_id = self.color_combo.itemData(selected_index)

		align_color = (self.align_color_button.selected_color.red(), 
			   self.align_color_button.selected_color.green(),
			  self.align_color_button.selected_color.blue())

		#IMPLEMENT THIS!!!!!!!!:
		#self.Model.set_AlignThickness(self.crop_thick_spinbox.value())
		#self.Model.set_AlignColor(align_color)
		#self.Model.set_ColorMap(cmap_id)
		#self.Model.ApplyLineStyle()

		self.close()
		
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#SELECT POLYNOMIAL DIALOG WINDOW:
class PolynomDialog(QMainWindow):

	def __init__(self, mainparent=None, model=None):
		super().__init__(mainparent)

	#To set params in model:
		self.Model = model

	#set icons:
		self.setWindowTitle("Polynomial selection")##############
		script_dir = os.path.dirname(os.path.abspath(__file__))##
		icon_path = os.path.join(script_dir, "color_setting.png")
		#Add Icon Path Here..........
		self.setWindowIcon(QIcon(icon_path))#####################

    #Obtain Screen Sizes:
		screen = QApplication.primaryScreen()#############
		screen_geometry = screen.availableGeometry()######
		screen_width = screen_geometry.width()############
		screen_height = screen_geometry.height()##########
        ##################################################
        #Setting the main window size as a% of the screen:
		self.w = int((screen_width) * 0.4)#####
		self.h = int((screen_height) * 0.9)### 
		self.resize(self.w,self.h)#
	######################################################

	#Central widget and it's Layout:
		central_widget = QWidget() ######################
		central_layout = QVBoxLayout() ##################
		central_layout.setContentsMargins(10, 10, 10, 10)

		central_widget.setLayout(central_layout) ########
		self.setCentralWidget(central_widget) ###########
		#################################################


		from matplotlib import rcParams
		# a figure instance to plot on
		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)
	#Add plt into Layout:
		central_layout.addWidget(self.canvas)

	#Best date message Label & QTableWidget containers:
		# Создаем QLabel и QTableWidget
		self.bestDateQlbl = QLabel("This is a QLabel")
		self.bestDateQlbl.setAlignment(Qt.AlignCenter) ##############
		self.bestDateQlbl.setObjectName("cropper_right_browse_label") 
		self.bestDateQlbl.setWordWrap(True) #########################

	#Mian Big Text on the Label:
		browseText = """
            <p style="text-align: justify;"><b>
			The Best Polynom on 25.05.2025</b></p>
            """
		self.bestDateQlbl.setText(browseText)
		
	#ADD SETTING OF RAW AND CALUMN:

		self.polyTable = QTableWidget()
		self.polyTable.setColumnCount(3)
		self.polyTable.setHorizontalHeaderLabels(["ID", "Date", "Note"])
		self.polyTable.setEditTriggers(QTableWidget.NoEditTriggers) 
		self.polyTable.setSelectionBehavior(QTableWidget.SelectRows)  
		self.polyTable.setSelectionMode(QTableWidget.SingleSelection) 
		self.polyTable.verticalHeader().setVisible(False)  
		self.polyTable.horizontalHeader().setStretchLastSection(True)
		###################################


	#Stack:
		self.stacked_layout = QStackedLayout()

		self.stacked_layout.addWidget(self.bestDateQlbl)  
		self.stacked_layout.addWidget(self.polyTable)   
		self.stack_widget = QWidget()
		self.stack_widget.setLayout(self.stacked_layout)
		
		#Add stack into mainLayout:
		central_layout.addWidget(self.stack_widget)

	#Size of dark label ########################################

		
		#This is label to set Layout with widgets (text & btn):
		buttonsQLbl = QLabel() ################################
		buttonsQLbl.setAlignment(Qt.AlignCenter) ##############
		#перенос текста ########################################
		buttonsQLbl.setWordWrap(True) #########################
		buttonsQLbl.setObjectName("cropper_right_browse_label") 
	#Size of dark label ########################################
		buttonsQLbl.setFixedSize(self.w/5.3, self.h/24) #######
		########################################################


	#Layout to hold buttons:
		layout = QHBoxLayout() ##############
		layout.setSpacing(15) ################
		layout.setContentsMargins(0, 0, 0, 0)
		layout.setAlignment(Qt.AlignTop) ####
		
	#Button to Cancel:
		self.anotherQbtn = QPushButton("Choose another date") ###########
		self.anotherQbtn.setObjectName("cropper_browse_btn")
		self.anotherQbtn.setEnabled(True) ##################
		self.anotherQbtn.setFixedSize(self.w/14, self.h/38)#

	#Button to save:
		self.acceptQbtn = QPushButton("Accept this date") #############
		self.acceptQbtn.setObjectName("cropper_browse_btn")
		self.acceptQbtn.setEnabled(True) ##################
		self.acceptQbtn.setFixedSize(self.w/14, self.h/38)#

	#Set btn&text(lbl) into layout then set it into browse QLbl:
		buttonsQLbl.setLayout(layout) #########################

		layout.addWidget(self.anotherQbtn, 0, Qt.AlignHCenter) 
		layout.addWidget(self.acceptQbtn, 0, Qt.AlignHCenter)
		
	#Set browse label into Main QVBoxLayout:
		central_layout.addWidget(buttonsQLbl, alignment=Qt.AlignCenter)
		
		self.POLYnom = None

		self.anotherQbtn.clicked.connect(self.showAnotherDate)
		self.acceptQbtn.clicked.connect(lambda: self.Model.setAppliedPolynom(self.POLYnom))
		self.adjustSize()
		self.setMinimumSize(self.size())
#End of Init.......................................................................................
		
	def setPolynoms(self, polynoms):
		self.polyTable.setRowCount(len(polynoms))  # Установить нужное количество строк

		for row, poly in enumerate(polynoms):
			# ID (может быть None, тогда показать пусто)
			id_item = QTableWidgetItem(str(poly.Id) if poly.Id is not None else "")
			id_item.setTextAlignment(Qt.AlignCenter)

			# Date
			date_item = QTableWidgetItem(str(poly.date))
			date_item.setTextAlignment(Qt.AlignCenter)

			# Note (может быть None)
			note_item = QTableWidgetItem(poly.note if poly.note is not None else "")
			note_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

			self.polyTable.setItem(row, 0, id_item)
			self.polyTable.setItem(row, 1, date_item)
			self.polyTable.setItem(row, 2, note_item)

	def showAnotherDate(self):
		self.stacked_layout.setCurrentIndex(1)

	def setTheBestDate(self, theBestPoly):
		self.bestDateQlbl.setText("Selected Date: \n" + str(theBestPoly.date))
		self.POLYnom = theBestPoly
	#def 