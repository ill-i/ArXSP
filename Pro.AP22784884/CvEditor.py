#Qt________________________________________
from PyQt5 import QtGui, QtCore, QtWidgets#
from PyQt5.QtGui import QImage#############
from PyQt5.QtWidgets import* ##############
from PyQt5.QtCore import* #################
from PyQt5.QtGui import* ##################
from PyQt5.QtCore import pyqtSignal########
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
from skimage import io, color, transform
#CONNECT TO DATABASE_______________________
from Db import Fai_db #####################
###########################################
  
from fsave import fSave
  

class HAND_PRO_STATE(Enum):
	STATE_NONE = 0
	STATE_CROP = 1
	STATE_ROTATE = 2

#Start of model part:
class Cv_model(QObject):
			
#init:
	def __init__(self):
		super().__init__()
		
		#get instance:
		self.db = Fai_db()
			
		#Left side (QWidget):
		self.__left = Left(self)
		#Right side (QVBoxLayout):
		self.__right= Right(self)
		#To call a dialog from outside:
		self.dialog_parent = None
	
		#original image:
		self.origin_image = None
		#data of fits file:
		self.data = None
		self.data_header = None
		
		#HAND PROCCESSING STATE:
		self.HND_STATE = HAND_PRO_STATE.STATE_NONE
				
#get instances:
	def getLeftWidget(self):
		return self.__left
	def getRightLayout(self):
		return self.__right	

			
#img path:
	def ImgLoad(self):
		try:
			fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self.dialog_parent , 'Single File', QtCore.QDir.rootPath() , '*.fits')
			
			#save data from fits file:
			file_fits = fits.open(fileName)
			data_fits = file_fits[0].data
			#set data into png
			plt.imsave("converted.png",data_fits[::-1])
			
			#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
			self.data = data_fits
			self.data_header = file_fits[0].header
			
			self.__left.ImgLoad()
			
		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
	#@@@@
	def ImgConvert(self):
		#self.PreobImage()
		
		fname = ""
		dialog = fSave(self)
		if dialog.exec_():
			fname = dialog.fname	
		else:
			fname = dialog.fname
			
		msg_box = QMessageBox()
		msg_box.addButton(QMessageBox.Ok)

		if fname =="":
			msg_box.setWindowTitle("Message Box")
			msg_box.setText("Error: Such file is already exist!")
			msg_box.exec_()
		else:
			msg_box.setWindowTitle("Success:")
			msg_box.setText("The File " + fname + " is created!")
			msg_box.exec_()

			try:
				
				user = self.db.getCurrent_User()
				
				#time = QDateTime.currentDateTime()
				
				#arra_log10 2D array for img
				
				
				array_log10 = np.log10(65535/self.data)*1000
				
				
				#mean is 1d array
				mean = array_log10.mean(axis=0)
				print("mean = ",mean)
				
				print("current user is " + user)
				#print('Local datetime: ', time.toString(Qt.ISODate))
				##print(mean)
				
				#You should put real typo 0-collib/ 1- spec:
				#LiNUx
				fname_no_path = fname.replace('\\','+').replace('/','+').split("+")[-1]
	#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
				
				type_value = 0
				if (self.__right.radiobutton2.isChecked()):
					type_value = 1
						
				self.db.Add_files_data(fname_no_path, type_value, mean)
				
				print("file: ", fname)
				self.Writefits(fname)
				#Add fname into list of preob files
				#..................................
				
				self.__right.Add_to_files_list(fname_no_path)
			except:
				print("transformation error!")
			
			print("Convert img into array and put into db")
		
#set parent before call dialog:
	def set_dialog_parent(self, parent):
		self.dialog_parent = parent	
	
#set hand state:
	def set_hand_state(self, numer):
		if(numer == 1):
			self.HND_STATE = HAND_PRO_STATE.STATE_CROP
		elif(numer == 2):
			self.HND_STATE = HAND_PRO_STATE.STATE_ROTATE
		else:
			self.HND_STATE = HAND_PRO_STATE.STATE_NONE
	
#for drawing:	
	def HorizontChanged(self, value, deg):
		self.__left.PaintLineHorizontRotate(value, deg)
		
	def DegChanged(self, value, deg):
		self.__left.PaintLineHorizontRotate(value, deg)
		
	def CropLinesChanged(self, right, left, top, down):
		self.__left.PaintCropLines(right, left, top, down)
		
	
#To rotate:	
	def RotateImg(self, deg):
		self.RotateImage(self.data, -deg)
		print("rotation has been done...")
		try:
			#save data from fits file:
			data_fits = fits.open('rotatedimg.fits')[0].data
			#set data into png
			plt.imsave("converted.png",data_fits[::-1])
			
			self.data = data_fits
			self.__left.ImgLoad()
			
		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
		
	def RotateImage(self, data, angle):
		"""
		Rotates image around center with angle theta (in deg)
		then crops the image according to width and height.
		"""
		shape = (data.shape[1],data.shape[0])
		center = (int(shape[0]/2),int(shape[1]/2))
		matrix = cv2.getRotationMatrix2D( center=center, angle=angle, scale=1 ) 	
		
		data_rotated = cv2.warpAffine(src=data, M=matrix, dsize=shape) 
		
		#save as fits
		hdu = fits.PrimaryHDU(data_rotated)
		hdulist = fits.HDUList([hdu])
		#number of pixels per axis
		NAXIS1 = hdulist[0].header["NAXIS1"]
		NAXIS2 = hdulist[0].header["NAXIS2"]
		#to save initial data about frame 
		hdulist[0].header = self.data_header
		#we have to write real number of pixels, not initial number
		hdulist[0].header["NAXIS1"] = NAXIS1
		hdulist[0].header["NAXIS2"] = NAXIS1
		#### rotatedimg.fits - write path instaed:
		hdulist.writeto('rotatedimg.fits',overwrite=True)

#To crop:
	def CropImg(self, right, left, top, down):
	
		w = self.__left.origin_image.shape[1]
		h = self.__left.origin_image.shape[0]
	
		Left = int(left*w/100)
		Right = int(right*w/100)

		Top = int(top*h/100)
		Down = int(down*h/100)

		d = self.data
		
		d = d[Down:(h-Top),Left:(w - Right)]
		
		#save as fits
		hdu = fits.PrimaryHDU(d)
		hdulist = fits.HDUList([hdu])
		#number of pixels per axis
		NAXIS1 = hdulist[0].header["NAXIS1"]
		NAXIS2 = hdulist[0].header["NAXIS2"]
		#to save initial data about frame
		hdulist[0].header = self.data_header
		#we want to locate BSCLE and BZERO card on certain and their standart position, so we use .set function
		#for unknown reasons the cards can change their positions, but only this two...
		hdulist[0].header.set("BSCALE",self.data_header["BSCALE"],self.data_header.comments["BSCALE"],after="NAXIS2")
		hdulist[0].header.set("BZERO",self.data_header["BZERO"],self.data_header.comments["BZERO"],after="BSCALE")
		#we have to write real number of pixels, not initial number
		hdulist[0].header["NAXIS1"] = NAXIS1
		hdulist[0].header["NAXIS2"] = NAXIS1
		hdulist.writeto('croppedImg.fits',overwrite=True)
		
		print("cropping has been done...")
		try:
			#save data from fits file:
			data_fits = fits.open('croppedImg.fits')[0].data
			#set data into png
			plt.imsave("converted.png",data_fits[::-1])
			
			self.data = data_fits
			self.__left.ImgLoad()
			
		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")
			
			
	def PreobImage(self):
		#arra_log10 2D array for img
		arra_log10 = np.log10(65535/self.data)*1000
		#mean is 1d array
		mean = array_log10.mean(axis=0)
		
		pass
		
	def CloseHandProcessing(self):
		self.__right.CloseHandProcessing()
		
	def Writefits(self, path):
		#hdu = fits.PrimaryHDU(data_rotated)
		hdu = fits.PrimaryHDU(self.data)
		hdulist = fits.HDUList([hdu])
		#number of pixels per axis
		NAXIS1 = hdulist[0].header["NAXIS1"]
		NAXIS2 = hdulist[0].header["NAXIS2"]
		#to save initial data about frame 
		hdulist[0].header = self.data_header
		#we have to write real number of pixels, not initial number
		hdulist[0].header["NAXIS1"] = NAXIS1
		hdulist[0].header["NAXIS2"] = NAXIS1
		#### rotatedimg.fits - write path instaed:
		hdulist.writeto(path,overwrite=True)
		
	
#Start of left part:
class Left(QLabel):
		
#init super:
	def __init__(self, parent=None):
		#Init main QLabel and it's property:
		QLabel.__init__(self, None)
		#self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
		self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.setAlignment(QtCore.Qt.AlignCenter)
		#self.setScaledContents(True)
		#self.showMaximized()
		self.setStyleSheet("color: rgb(255,79,0); font-size: 10pt")
		self.setText('TO START WORKING BROWSE & LOAD FITS FILE...')	
		
		#Init it's parent:
		self.parent = parent
		
		#w,h - the size of main QLabel
		self.w = self.width()
		self.h = self.height()
		
		#Init mouse moving:
		#self.setMouseTracking(True)
		
		#the image to draw:
		self.origin_image = None
		self.tmp_image = None
		
		#Init menu on screen:
		self.setFocusPolicy(Qt.ClickFocus)
			


	def set_reset(self):
		self.tmp_image = self.origin_image.copy()
		self.Update()
		print("RESET")
		return self.w



	def ImgLoad(self):
		try:
			#read image by cv2:
			self.origin_image = cv2.imread("converted.png")
			
			self.origin_w = self.origin_image.shape[0]
			self.origin_h = self.origin_image.shape[1]

			#reset states:
			self.set_reset()
			#call method SetImage:
			self.SetImage(self.origin_image)
		except Exception as err:
			print(f"Unexpected {err=}, {type(err)=}")		
			
	def SetImage(self, img):
		img = imutils.resize(img,width=self.width(),height=self.height())
		frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = QImage(frame, frame.shape[1],frame.shape[0],frame.strides[0], QImage.Format_RGB888)
		self.setPixmap(QtGui.QPixmap.fromImage(img))	
		
	def Update(self):
		self.SetImage(self.tmp_image)
			
	def resizeEvent(self, e: QResizeEvent) -> None:
		"""Window resize event."""
		super().resizeEvent(e)
		self.SetImage(self.tmp_image)
		print("size {" + str(self.width()) + ";" + str(self.height()) + "}")

	def PaintLineHorizontRotate(self,horvalue, deg):
		color = (0,79,255)
		#Line thickness of 9 px
		thickness = 6
		
		w = self.origin_image.shape[1]
		h = self.origin_image.shape[0]
		
		self.ch = h*horvalue/100
		
		point1 = (0, int(h*horvalue/100))
		point2 = (w, int(h*horvalue/100))
				
		#rotate	
		point3, point4 =self.RotateLine(point1[0],point1[1],point2[0],point2[1],deg)
		
		#Cv draw algorithm:
		self.tmp_image = self.origin_image.copy()
		self.tmp_image = cv2.line(self.tmp_image, point3, point4, color, thickness)
		self.Update()
		
	def PaintCropLines(self, right, left, top, down):
	
		color = (255,79,0)#norm
		
		color_rl = (0,0,255)
		color_td = (0,0,255)
		
		#Line thickness of 9 px
		thickness = 8
		
		w = self.origin_image.shape[1]
		h = self.origin_image.shape[0]
		
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
		self.tmp_image = self.origin_image.copy()
		
		self.tmp_image = cv2.line(self.tmp_image, lft_point1, lft_point2, color_rl, thickness)
		self.tmp_image = cv2.line(self.tmp_image, rht_point1, rht_point2, color_rl, thickness)
		self.tmp_image = cv2.line(self.tmp_image, top_point1, top_point2, color_td, thickness)
		self.tmp_image = cv2.line(self.tmp_image, dwn_point1, dwn_point2, color_td, thickness)
		
		self.Update()
	
		
	def RotateLine(self, x1, y1, x2, y2, angle):
		"""
		Rotates line with know start and end points coordinates
		in 2d space by an angle
		"""
		#plt.plot([x1,x2],[y1,y2])
		line = LineString([(x1,y1), (x2,y2)])
		x1_line, x2_line = line.coords.xy[0][0], line.coords.xy[0][1] 
		y1_line, y2_line = line.coords.xy[1][0], line.coords.xy[1][1] 
		cx = (x1+x2)/2
		cy = (y1+y2)/2
		rotated = affinity.rotate(line, angle, origin=(cx,cy))
		x1_rotated, x2_rotated = rotated.coords.xy[0][0], rotated.coords.xy[0][1] 
		y1_rotated, y2_rotated = rotated.coords.xy[1][0], rotated.coords.xy[1][1] 
		#plt.ylim(8,12)
		#plt.plot([x1_rotated,x2_rotated],[y1_rotated,y2_rotated],c="r")
		
		len_initial = ((x1-x2)**2+(y1-y2)**2)**0.5
		len_rotated = ((x1_rotated-x2_rotated)**2+(y1_rotated-y2_rotated)**2)**0.5
		#print(len_initial==len_rotated)
		
		return (int(x1_rotated),int(y1_rotated)),(int(x2_rotated),int(y2_rotated))
		
#Start of right part:		
class Right(QVBoxLayout):
	
#signals:	
	def __init__(self, parent):
		#setting:
		super().__init__()
		
		self.parent = parent
		
		self.setContentsMargins(0, 0, 0, 0)
		self.setAlignment(Qt.AlignTop)
		
		self.QGrbox1 = QGroupBox("spectrum type:")
		self.QGrbox1.setContentsMargins(0, 0, 0, 0)
		self.QGrbox1.setObjectName("grbox_search")
		self.addWidget(self.QGrbox1)
		
		box1 = QVBoxLayout()
		box1.setContentsMargins(4, 0, 0, 0)
		#@@@
		self.radiobutton1 = QRadioButton("calibration")
		box1.addWidget(self.radiobutton1)
		#@@@
		self.radiobutton2 = QRadioButton("scale")
		box1.addWidget(self.radiobutton2)
		self.QGrbox1.setLayout(box1)
		
		#Buttons:
		self.btn_ImgLoad = QPushButton("browse...")
		self.btn_convert = QPushButton("Convert")
			
		btn_container = QWidget()
		btn_container.setContentsMargins(0,0,0,0)
		btn_layout = QHBoxLayout()
		btn_layout.setContentsMargins(0,0,0,0)
		btn_container.setLayout(btn_layout)
			
		btn_layout.addWidget(self.btn_ImgLoad)
		btn_layout.addWidget(self.btn_convert)
			
		self.addWidget(btn_container)
		#Buttons end:

		self.QGrbox2 = QGroupBox("tool bar")
		self.QGrbox2.setContentsMargins(0, 0, 0, 0)
		self.QGrbox2.setObjectName("grbox_search")
		self.addWidget(self.QGrbox2)
		
		##############################################
		self.QGrbox2.setLayout(self.initToolbuttons())		  
		##############################################
		
		#Add Label in VLayout  
		#box3 = QVBoxLayout()		
		#box3.setContentsMargins(0, 0, 0, 0)
		LblImgTxt = QLabel("Image... ")
		LblImgTxt.setContentsMargins(2, 0, 0, 0)
		LblImgTxt.setObjectName("lbl1")
		#self.LblImgTxt.setText("Found objects: ")
		LblImgTxt.setAlignment(Qt.AlignLeft)
		LblImgTxt.setStyleSheet("QLabel#lbl1 { color : green; }")
		LblImgTxt.setAutoFillBackground(True)
		#self.box3.addWidget(self.lbl1)
		self.addWidget(LblImgTxt)
		
		"""
		#buttons layout:
		hbox = QHBoxLayout()
		#None button:
		btn = QPushButton("submit")
		btn.setEnabled(False)
		btn.setObjectName("hidden")
			

			
		#Button to load image:
		self.btn_convert = QPushButton("Convert")
		hbox.addWidget(btn)
		hbox.addWidget(self.btn_convert)
				
				
				
				
		#button group box:		
		self.QGrbox3 = QGroupBox("do:")
		self.QGrbox3.setContentsMargins(0, 0, 0, 0)
		self.QGrbox3.setLayout(hbox)
		
		
		
		#add button group box into widgets box:
		self.addWidget(self.QGrbox3)  
		"""
				
		self.data_table = QListView()
		sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		sizePolicy.setHorizontalStretch(0)
		sizePolicy.setVerticalStretch(0)
		sizePolicy.setHeightForWidth(self.data_table.sizePolicy().hasHeightForWidth())
		self.data_table.setSizePolicy(sizePolicy)
		self.data_table.setObjectName("list_table")
		self.addWidget(self.data_table)
		
		
		preob_items = self.Init_files_list()
		self.Update_files_list(preob_items)
		#set connections:
		self.initConnections()
		
	def initToolbuttons(self):
		#Butons of Handle, Mouse & Net processing...
		self.HndBtn = QPushButton("")
		self.AimBtn = QPushButton("")
		self.NetBtn = QPushButton("")
		
		self.HndBtn.setObjectName("HandBtn")
		self.AimBtn.setObjectName("AimBtn")
		self.NetBtn.setObjectName("RobotBtn")

		self.HndBtn.setFixedSize(35, 35)
		self.AimBtn.setFixedSize(35, 35)
		self.NetBtn.setFixedSize(35, 35)
		
		self.HndBtn.setIconSize(QSize(30,30));
		self.AimBtn.setIconSize(QSize(30,30));
		self.NetBtn.setIconSize(QSize(30,30));
		
		self.HndBtn.setIcon(QIcon(':/resources/ghand.png'))
		self.AimBtn.setIcon(QIcon(':/resources/gaim.png'))
		self.NetBtn.setIcon(QIcon(':/resources/grobot.png'))	  
				  
		#Box for Handle, Mouse Handle, Net buttons...
		BtnHbox = QHBoxLayout()
		BtnHbox.setContentsMargins(2, 2, 2, 2)
		BtnHbox.addWidget(self.HndBtn)	
		BtnHbox.addWidget(self.AimBtn)	
		BtnHbox.addWidget(self.NetBtn)
		
		#Messages tablet for tool Btns:
		self.MsgLbl = QLabel("")
		self.MsgLbl.setObjectName("lbl1")
		self.MsgLbl.setText("Not ready")
		self.MsgLbl.setAlignment(Qt.AlignCenter)
		self.MsgLbl.setStyleSheet("QLabel#lbl1 { color : rgb(255,79,0); }")
		
		#Container for Message tablet:
		MsgHbox = QHBoxLayout()
		MsgHbox.setContentsMargins(2, 2, 2, 2)
		MsgHbox.addWidget(self.MsgLbl)
		
		#Container of BtnHbox and Msgbox
		mainBox = QVBoxLayout()
		mainBox.addLayout(BtnHbox)
		mainBox.addLayout(MsgHbox)
		
		return mainBox
		
#connections:	
	def initConnections(self):
		self.btn_ImgLoad.clicked.connect(self.parent.ImgLoad)
		self.btn_convert.clicked.connect(self.parent.ImgConvert)
		#Messages for tablet
		self.HndBtn.enterEvent = lambda e: self.MsgLbl.setText("Hand processing..")
		self.AimBtn.enterEvent = lambda e: self.MsgLbl.setText("Mouse processing.")
		
		self.NetBtn.enterEvent = lambda e: self.MsgLbl.setText("Neural processing")
		
		self.HndBtn.clicked.connect(self.OpenHandProcessing)
	
	def OpenHandProcessing(self):
		print("Hand Processing is opened...")
		self.HndBtn.setEnabled(False)
		self.AimBtn.setEnabled(False)
		self.NetBtn.setEnabled(False)
		
		dialog = HndProcessingDialog(self.parent.dialog_parent, self.parent)
		dialog.show()
		
	def CloseHandProcessing(self):
		print("Hand Processing is closed...")
		self.HndBtn.setEnabled(True)
		self.AimBtn.setEnabled(True)
		self.NetBtn.setEnabled(True)
		
	def Update_files_list(self, flist):
		model = QStandardItemModel()
		for item in flist:
			model.appendRow(QStandardItem(item))

		# set the model for the list view
		self.data_table.setModel(model)
		
	def Init_files_list(self):

		root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		folder_name = 'resources_db'
		fpath = os.path.join(root_dir, folder_name)
		
		fits_files = [f for f in os.listdir(fpath) if f.endswith('.fits')]
		
		return fits_files
		
	def Add_to_files_list(self, fname):
		new_item = QStandardItem(fname)

		# get the current model from the QListView
		model = self.data_table.model()

		# add the new item to the model
		model.appendRow(new_item)
		
		print("file hase been created")
		
#Cut button pressed:
class HndProcessingDialog(QDialog):

#constructor:
	def __init__(self, mainparent=None, model = None):
		super().__init__(mainparent)

		#cv model
		self.Model = model
		
		#title & size:
		self.setWindowTitle("Hand proccessing image")
		#self.setGeometry(0,0,200,450)
		#self.setFixedWidth(240)
		#self.setFixedHeight(450)

		#set settings:
		self.setContentsMargins(1, 1, 1, 1)

		#main layout for dialog window:
		dialoglayout = QVBoxLayout()
		dialoglayout.setContentsMargins(1, 1, 1, 1)
		self.setLayout(dialoglayout)
		
		#set green border:
		greenborder = QGroupBox("",self)  
		greenborder.setObjectName('green-bordered')
		
		#label for title:
		lbl = QLabel("")
		lbl.setContentsMargins(1, 1, 1, 1)	
		lbl.setObjectName("LblTitle")
		lbl.setText("Hand proccessing...")
		lbl.setFixedHeight(20)
		lbl.setAlignment(Qt.AlignLeft)
		
		#Label & QGroupbox into dialoglayout:
		dialoglayout.addWidget(lbl)
		dialoglayout.addWidget(greenborder)
		
		#Layout for green Groupbox:
		self.mainLayout = QVBoxLayout()
		self.mainLayout.setContentsMargins(2, 2, 2, 2)
	
		greenborder.setLayout(self.mainLayout)
		
		#MainLayout to hold control widgets:
		self.mainLayout.addLayout(self.initUi())
		
		
		#Values:
		self.deg_Value = 0
		self.hor_Value = 0
		
		
		self.lft_Value = 0
		self.rht_Value = 0
		
		
		self.top_Value = 0
		self.dwn_Value = 0
		
		self.rotate_gbox.setEnabled(False)
		self.crop_gbox.setEnabled(False)
		#end __init__
		
#INITIALIZATION:
	def initUi(self):
		#Radio buttons:
		self.rbtn_rotate= QRadioButton("Rotate")
		self.rbtn_crop = QRadioButton("Crop")
		self.rbtn_rotate.toggled.connect(self.rotate_Select)
		self.rbtn_crop.toggled.connect(self.crop_Select)
		
		#Layout for Radio buttons:
		rbtn_box = QVBoxLayout()
		rbtn_box.setContentsMargins(4, 0, 0, 0)
		rbtn_box.addWidget(self.rbtn_rotate)
		rbtn_box.addWidget(self.rbtn_crop)
		
		#GroupBox for radiobutton Layout
		self.grbox = QGroupBox("modes ")
		self.grbox.setContentsMargins(0, 0, 0, 0)
		
		self.grbox.setObjectName("grbox_search")
		self.grbox.setLayout(rbtn_box)
		
		#Add GroupBox for radiobutton Layout into mainLayout
		mainLayout = QVBoxLayout()
		mainLayout.setContentsMargins(2, 2, 2, 2)
		mainLayout.addWidget(self.grbox)
		
		#Add UiRotation and UiCroping into mainLayout
		mainLayout.addLayout(self.initUiRotate())
		mainLayout.addLayout(self.initUiCrop())
		
		return mainLayout
		#end initUI
		
#initialize ui of rotation:
	def initUiRotate(self):

		self.CreateRotateElements()
		
		#For Desighn_____________________________________________
		#1 Layout
		rotateHLayout1 = QHBoxLayout()
		rotateHLayout1.addWidget(self.rotate_dial)
		rotateHLayout1.addWidget(self.horizont_slider)
	
		#2 Layout
		rotateHLayout2 = QHBoxLayout()
		rotateHLayout2.setContentsMargins(0,0,0,0)
		rotateHLayout2.addWidget(self.rotate_lcd)
		rotateHLayout2.addWidget(self.rotate_spbox)
		
		#3 group rotation tools:
		rotateVLayout3 = QVBoxLayout()
		rotateVLayout3.setContentsMargins(0,0,0,0)
		rotateVLayout3.addLayout(rotateHLayout1)
		
		DegLayout = QVBoxLayout()
		DegLayout.setContentsMargins(0,0,0,0)
		
		deg_lbl = QLabel("Degree:")
		deg_lbl.setStyleSheet("color: rgba(69, 172, 66,255);padding: 0px; margin: 0px;")
		
		DegLayout.addWidget(deg_lbl)
		DegLayout.addLayout(rotateHLayout2)
		
		HorisonLayout = QVBoxLayout()
		HorisonLayout.setContentsMargins(0,0,0,0)
		
		hor_lbl = QLabel("Horizont:")
		hor_lbl.setStyleSheet("color: rgba(69, 172, 66,255); padding: 0px; margin: 0px")
		
		#2 Layout
		rotateHLayout3 = QHBoxLayout()
		rotateHLayout3.addWidget(self.horizone_lcd)
		rotateHLayout3.addWidget(self.horizone_spbox)
		
		#rotateVLayout3.addLayout(rotateHLayout3)
		HorisonLayout.addWidget(hor_lbl)
		HorisonLayout.addLayout(rotateHLayout3)
		
		groupLayout = QHBoxLayout()
		groupLayout.setContentsMargins(0,0,0,0)
		groupLayout.addLayout(DegLayout)
		groupLayout.addLayout(HorisonLayout)
		
		rotateVLayout3.addLayout(groupLayout)
		
		#buttons layout:
		btn_layout = QHBoxLayout()
		#None button:
		hidden_btn = QPushButton("submit")
		hidden_btn.setEnabled(False)
		hidden_btn.setObjectName("hidden")
				
		btn_layout.addWidget(hidden_btn)
		btn_layout.addWidget(self.btn_rotate)
		
		rotateVLayout3.addLayout(btn_layout)
		
		#self.rotate_gbox
		self.rotate_gbox = QGroupBox("rotate tools:")
		self.rotate_gbox.setContentsMargins(0, 0, 0, 0)
		self.rotate_gbox.setObjectName("grbox_search")
		self.rotate_gbox.setLayout(rotateVLayout3)
		#rotate_gbox.setEnabled(False)

		rotateLayout = QVBoxLayout()
		rotateLayout.addWidget(self.rotate_gbox)
		return rotateLayout
		#end initUiRotate
		
#initialize ui of cropping:
	def initUiCrop(self):

		self.CreateCropElements()
		
		#For Desighn_____________________________________________
		V_crop_slider_Layout1 = QVBoxLayout()
		V_crop_slider_Layout1.addWidget(self.left_crop_slider)
		V_crop_slider_Layout1.addWidget(self.right_crop_slider)

		H_crop_slider_Layout1 = QHBoxLayout()
		H_crop_slider_Layout1.addWidget(self.top_crop_slider)
		H_crop_slider_Layout1.addWidget(self.dwn_crop_slider)
		
		crop_slider_Layout = QHBoxLayout()
		crop_slider_Layout.addLayout(V_crop_slider_Layout1)
		crop_slider_Layout.addLayout(H_crop_slider_Layout1)
		
		crop_Layout = QVBoxLayout()
		crop_Layout.setContentsMargins(0,0,0,0)
		crop_Layout.addLayout(crop_slider_Layout)
		
		#Top-Bottom:
		top_down_Layout = QHBoxLayout()
		
		top_lbl = QLabel("Top-Bottom:")
		top_lbl.setStyleSheet("color: rgba(69, 172, 66,255); padding: 0px; margin: 0px")
		crop_Layout.addWidget(top_lbl)
		
		topHLayout = QHBoxLayout()
		topHLayout.addWidget(self.top_lcd)
		topHLayout.addWidget(self.top_spbox)
		
		downHLayout = QHBoxLayout()
		downHLayout.addWidget(self.dwn_lcd)
		downHLayout.addWidget(self.dwn_spbox)
		
		top_down_Layout.addLayout(topHLayout)
		top_down_Layout.addLayout(downHLayout)
		
		crop_Layout.addLayout(top_down_Layout)

		#Left-Right:
		left_right_Layout = QHBoxLayout()
		
		left_right_lbl = QLabel("Left-Right:")
		left_right_lbl.setStyleSheet("color: rgba(69, 172, 66,255); padding: 0px; margin: 0px")
		crop_Layout.addWidget(left_right_lbl)
		
		topHLayout = QHBoxLayout()
		topHLayout.addWidget(self.left_lcd)
		topHLayout.addWidget(self.left_spbox)
		
		downHLayout = QHBoxLayout()
		downHLayout.addWidget(self.right_lcd)
		downHLayout.addWidget(self.right_spbox)
		
		left_right_Layout.addLayout(topHLayout)
		left_right_Layout.addLayout(downHLayout)
		
		crop_Layout.addLayout(left_right_Layout)
		
		#buttons layout:
		btn_layout = QHBoxLayout()
		#None button:
		hidden_btn = QPushButton("submit")
		hidden_btn.setEnabled(False)
		hidden_btn.setObjectName("hidden")
				
		#Button to crope:
		btn_layout.addWidget(hidden_btn)
		btn_layout.addWidget(self.btn_crop)
	
		crop_Layout.addLayout(btn_layout)
	
		self.crop_gbox = QGroupBox("crop tools:")
		self.crop_gbox.setContentsMargins(0, 0, 0, 0)
		self.crop_gbox.setObjectName("grbox_search")
		self.crop_gbox.setLayout(crop_Layout)

		d = QVBoxLayout()
		d.addWidget(self.crop_gbox)
		return d
		#end initUiCrop

	def closeEvent(self, evnt):
		self.Model.CloseHandProcessing()
		
#control elements:
	def CreateRotateElements(self):
	#ROTATE:
		#rotate dial:
		self.rotate_dial = QDial()
		self.rotate_dial.setStyleSheet("color: rgba(69, 172, 66,255)")
		self.rotate_dial.setMinimum(-179.99)
		self.rotate_dial.setMaximum(179.99)
		self.rotate_dial.setValue(0)
		
		self.rotate_dial.valueChanged.connect(lambda: self.DegChanged(self.rotate_dial.value()))
		
		#Rotate LCD:
		self.rotate_lcd = QLCDNumber()
		self.rotate_lcd.setFixedHeight(20)
		self.rotate_lcd.setSegmentStyle(QLCDNumber.Flat)
		self.rotate_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		#Rotate SpinBox:
		self.rotate_spbox = QDoubleSpinBox()
		self.rotate_spbox.setStyleSheet("padding: 0px;")
		self.rotate_spbox.setFixedHeight(20)
		self.rotate_spbox.setRange(-179.99, 179.99)
		
		self.rotate_spbox.valueChanged.connect(lambda: self.DegChanged(self.rotate_spbox.value()))
		
	#HORIZONT:
		#horizont rotate slider:
		self.horizont_slider = QSlider(Qt.Vertical)
		self.horizont_slider.setFixedWidth(60)
		self.horizont_slider.setMinimum(0)
		self.horizont_slider.setMaximum(100)
		self.horizont_slider.setValue(0)
		self.horizont_slider.setTickPosition(QSlider.TicksBelow)
		self.horizont_slider.setTickInterval(0.1)
		
		self.horizont_slider.valueChanged.connect(lambda: self.HorizontChanged(self.horizont_slider.value()))
		
		#Horizone LCD:
		self.horizone_lcd = QLCDNumber()
		self.horizone_lcd.setFixedHeight(20)
		
		self.horizone_lcd.setSegmentStyle(QLCDNumber.Flat)
		self.horizone_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		#Horizone SpinBox:
		self.horizone_spbox = QDoubleSpinBox()
		self.horizone_spbox.setFixedHeight(20)
		self.horizone_spbox.setRange(0, 100)
		
		self.horizone_spbox.valueChanged.connect(lambda: self.HorizontChanged(self.horizone_spbox.value()))
		
		#Button to load image:
		self.btn_rotate = QPushButton("Rotate")
		self.btn_rotate.clicked.connect(lambda: self.rotateImg())
		#need connections......
		
	def CreateCropElements(self):
				
		#left - slider:
		self.left_crop_slider = QSlider(Qt.Horizontal)
		self.left_crop_slider.setMinimum(0)
		self.left_crop_slider.setMaximum(100)
		self.left_crop_slider.setValue(0)
		self.left_crop_slider.setTickPosition(QSlider.TicksBelow)
		self.left_crop_slider.setTickInterval(1)
		
		self.left_crop_slider.valueChanged.connect(lambda: self.LeftLineChanged(self.left_crop_slider.value()))
		
		#Left - LCD:
		self.left_lcd = QLCDNumber()
		self.left_lcd.setFixedHeight(20)
		self.left_lcd.setSegmentStyle(QLCDNumber.Flat)
		self.left_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		#Left - SpinBox: 
		self.left_spbox = QDoubleSpinBox()
		self.left_spbox.setFixedHeight(20)
		
		self.left_spbox.valueChanged.connect(lambda: self.LeftLineChanged(self.left_spbox.value()))
		
		
		#right - slider:
		self.right_crop_slider = QSlider(Qt.Horizontal)
		self.right_crop_slider.setMinimum(0)
		self.right_crop_slider.setMaximum(100)
		self.right_crop_slider.setValue(0)
		self.right_crop_slider.setTickPosition(QSlider.TicksBelow)
		self.right_crop_slider.setTickInterval(1)
		
		self.right_crop_slider.valueChanged.connect(lambda: self.RightLineChanged(self.right_crop_slider.value()))
		
		#Right - LCD:
		self.right_lcd = QLCDNumber()
		self.right_lcd.setFixedHeight(20)
		self.right_lcd.setSegmentStyle(QLCDNumber.Flat)
		self.right_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		#Right - SpinBox:
		self.right_spbox = QDoubleSpinBox()
		self.right_spbox.setFixedHeight(20)
		
		self.right_spbox.valueChanged.connect(lambda: self.RightLineChanged(self.right_spbox.value()))

		
		#top - slider:
		self.top_crop_slider = QSlider(Qt.Vertical)
		self.top_crop_slider.setMinimum(0)
		self.top_crop_slider.setMaximum(100)
		self.top_crop_slider.setValue(0)
		self.top_crop_slider.setTickPosition(QSlider.TicksBelow)
		self.top_crop_slider.setTickInterval(1)
		
		self.top_crop_slider.valueChanged.connect(lambda: self.TopLineChanged(self.top_crop_slider.value()))

		#Top - LCD:
		self.top_lcd = QLCDNumber()
		self.top_lcd.setFixedHeight(20)
		self.top_lcd.setSegmentStyle(QLCDNumber.Flat)
		self.top_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		#Top - SpinBox:
		self.top_spbox = QDoubleSpinBox()
		self.top_spbox.setFixedHeight(20)
		
		self.top_spbox.valueChanged.connect(lambda: self.TopLineChanged(self.top_spbox.value()))

		
		#down - slider:
		self.dwn_crop_slider = QSlider(Qt.Vertical)
		self.dwn_crop_slider.setMinimum(0)
		self.dwn_crop_slider.setMaximum(100)
		self.dwn_crop_slider.setValue(0)
		self.dwn_crop_slider.setTickPosition(QSlider.TicksBelow)
		self.dwn_crop_slider.setTickInterval(1)
		
		self.dwn_crop_slider.valueChanged.connect(lambda: self.DownLineChanged(self.dwn_crop_slider.value()))
		
		#Down- LCD:
		self.dwn_lcd = QLCDNumber()
		self.dwn_lcd.setFixedHeight(20)
		self.dwn_lcd.setSegmentStyle(QLCDNumber.Flat)
		self.dwn_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		#Down SpinBox:
		self.dwn_spbox = QDoubleSpinBox()
		self.dwn_spbox.setFixedHeight(20)
		
		self.dwn_spbox.valueChanged.connect(lambda: self.DownLineChanged(self.dwn_spbox.value()))
		
		self.btn_crop = QPushButton("Crop")
		self.btn_crop.clicked.connect(lambda: self.cropImg())
		
		#need connections
		
		
#SLOTS SELECT:
	def rotate_Select(self):
		#command to model:
		self.Model.set_hand_state(2)
		#command to viewer:
		self.rotate_gbox.setEnabled(True)
		self.crop_gbox.setEnabled(False)
		
		self.rotate_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		self.horizone_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		self.left_lcd.setStyleSheet("padding: 0px; color: rgba(120, 120, 120, 120)")
		self.right_lcd.setStyleSheet("padding: 0px; color: rgba(120, 120, 120, 120)")
		self.top_lcd.setStyleSheet("padding: 0px; color: rgba(120, 120, 120, 120)")
		self.dwn_lcd.setStyleSheet("padding: 0px; color: rgba(120, 120, 120, 120)")
		
	def crop_Select(self):
		#command to model:
		self.Model.set_hand_state(1)
		#command to viewer:
		self.rotate_gbox.setEnabled(False)
		self.crop_gbox.setEnabled(True)
		
		self.rotate_lcd.setStyleSheet("padding: 0px; color: rgba(120, 120, 120, 120)")
		self.horizone_lcd.setStyleSheet("padding: 0px; color: rgba(120, 120, 120, 120)")
		
		self.left_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		self.right_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		self.top_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		self.dwn_lcd.setStyleSheet("padding: 0px; color: rgba(69, 172, 66, 255)")
		
		
#SLOTS ROTATE & HORIZONT:	
	def DegChanged(self, value):
		self.deg_Value = value
		
		self.rotate_lcd.display(int(value))
		self.rotate_spbox.setValue(value)
		self.rotate_dial.setValue(value)
		self.Model.HorizontChanged(self.hor_Value, self.deg_Value)
	
	def HorizontChanged(self, value):
		self.hor_Value = value
		
		self.horizont_slider.setValue(value)
		self.horizone_lcd.display(int(value))
		self.horizone_spbox.setValue(value)
		
		self.Model.HorizontChanged(self.hor_Value, self.deg_Value)
		
#SLOTS CROP LINES:
	def LeftLineChanged(self, value):
	
		self.lft_Value = value
	
		self.left_lcd.display(int(value))
		self.left_crop_slider.setValue(value)
		self.left_spbox.setValue(value)
		
		self.CropLinesChanged(self.rht_Value, self.lft_Value, self.top_Value, self.dwn_Value)

	def RightLineChanged(self, value):
		self.rht_Value = value
	
		self.right_lcd.display(int(value))
		self.right_crop_slider.setValue(value)
		self.right_spbox.setValue(value)
		
		self.CropLinesChanged(self.rht_Value, self.lft_Value, self.top_Value, self.dwn_Value)
		
	def TopLineChanged(self, value):
		self.top_Value = value
	
		self.top_lcd.display(int(value))
		self.top_crop_slider.setValue(value)
		self.top_spbox.setValue(value)
		
		self.CropLinesChanged(self.rht_Value, self.lft_Value, self.top_Value, self.dwn_Value)
		
	def DownLineChanged(self, value):
		self.dwn_Value = value
	
		self.dwn_lcd.display(int(value))
		self.dwn_crop_slider.setValue(value)
		self.dwn_spbox.setValue(value)
		
		self.CropLinesChanged(self.rht_Value, self.lft_Value, self.top_Value, self.dwn_Value)
	
	def CropLinesChanged(self, right, left, top, down):
		self.Model.CropLinesChanged(right, left, top, down)
	
	
	def rotateImg(self):
		self.Model.RotateImg(self.deg_Value)
	
	def cropImg(self):
		self.Model.CropImg(self.rht_Value, self.lft_Value, self.top_Value, self.dwn_Value)
	

	
#self.deg_Value = 0
#self.hor_Value = 0

#self.lft_Value = 0
#self.rht_Value = 0

#self.top_Value = 0
#self.dwn_Value = 0
		
#self.btn_crop = QPushButton("Crop")
#self.btn_rotate = QPushButton("Rotate")


"""
		#save as fits
		hdu = fits.PrimaryHDU(data_rotated)
		hdulist = fits.HDUList([hdu])
		#number of pixels per axis
		NAXIS1 = hdulist[0].header["NAXIS1"]
		NAXIS2 = hdulist[0].header["NAXIS2"]
		#to save initial data about frame 
		hdulist[0].header = self.data_header
		#we have to write real number of pixels, not initial number
		hdulist[0].header["NAXIS1"] = NAXIS1
		hdulist[0].header["NAXIS2"] = NAXIS1
		#### rotatedimg.fits - write path instaed:
		hdulist.writeto('rotatedimg.fits',overwrite=True)
"""

		
		
		
		
		
		
		
	

