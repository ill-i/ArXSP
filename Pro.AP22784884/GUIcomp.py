﻿#Qt5 files:
from PyQt5.QtGui import* ###########################
from PyQt5.QtCore import* ##########################
from PyQt5.QtWidgets import* #######################
from PyQt5.QtGui import QIcon ######################
from PyQt5.QtGui import QScreen ####################
from PyQt5.QtWidgets import QFrame #################
from PyQt5.QtCore import Qt, QPropertyAnimation #### 
from PyQt5.QtCore import QSize, pyqtSignal, QObject#
from PyQt5 import QtGui, QtCore, QtWidgets ######### 
####################################################
from PyQt5.QtWidgets import (
    QWidget, QDockWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QToolButton, QTextEdit, QLabel
)#########################################################

#system files:
import sys #####
import inspect #
import os 
################



#CLass for animated toolbar buttons:
class AnimatedToolButton(QToolButton):

#Constructor:
    def __init__(self, default_icon, active_icon, text, w, h, parent=None):

        #set icon on toolbar buttons:
        super().__init__(parent) ############################
        self.default_icon = default_icon ####################
        self.active_icon = active_icon ######################
        self.setIcon(self.default_icon) #####################
        self.setText(text) ##################################
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon) #
        #####################################################
        
        #setting the button size:
        #initial size of buttons ###############################
        self.original_size = QSize(int(w/18), int(w/18)) #######
        self.setFixedSize(self.original_size) ##################
        self.original_geometry = None ##########################
        self.animation = QPropertyAnimation(self, b"geometry") #
        self.is_active = False #################################
        ########################################################
    #End of Constructor#########################################
    
#Active or Not:
    def setActive(self, active):
        #Sets the state of the button (active or not) #################
        self.is_active = active #######################################
        self.setIcon(self.active_icon if active else self.default_icon)
    #End of activity ##################################################

#Hover animation:
    def enterEvent(self, event):

        if not self.original_geometry:
            #Сохраняем изначальную геометрию кнопки:
            self.original_geometry = self.geometry()  
        
        #Увеличить кнопку:
        new_geometry = self.original_geometry.adjusted(-10, -10, 10, 10) ##
        self.animation.setDuration(200) ###################################
        # Текущая геометрия
        self.animation.setStartValue(self.geometry()) ##################### 
        # Увеличенная геометрия
        self.animation.setEndValue(new_geometry) ########################## 
        self.animation.start() ############################################
        super().enterEvent(event) #########################################
        ###################################################################
    #End of enter Event ###################################################

#Leave animation:
    def leaveEvent(self, event):
        # Анимация уменьшения при покидании мыши
        if self.original_geometry: #############
            self.animation.setDuration(200)
            # Текущая геометрия ###############################
            self.animation.setStartValue(self.geometry()) ##### 
            # Возвращаем к изначальной геометрии ##############
            self.animation.setEndValue(self.original_geometry)# 
            self.animation.start() ############################
        super().leaveEvent(event) #############################
#End of CLass AnimatedToolButton ######



#Main DockWidget on the right side to hold Model's right: 
class QDashboard(QDockWidget):

#Constructor:
    def __init__(self,parent=None):

    #settings dockwidget:
        super().__init__('Dock', parent) #################################
        ##################################################################
        self.setWindowTitle('Tool bar...') ###############################
        self.setAllowedAreas(Qt.LeftDockWidgetArea|Qt.RightDockWidgetArea)
        ##################################################################

    #main container Widget is a QGroupbox:
        self.container = QGroupBox()##################
        self.container.setObjectName('dashborder') ##
        #############################################

    #layout for main container: #####################
        self.layout = QVBoxLayout() #################
    #Set layout into container: #####################
        self.container.setLayout(self.layout) #######
    #set container as main widget: ##################
        self.setWidget(self.container) ##############
        #############################################

    #Map of number & w-QWidget 
        self.NumWidgetMap = dict() 
        self.show()
    #End of Constructor##############################



#Close pressed:
    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.hide()
    #...............................

#Show:
    def Show(self):
        #self.setVisible(True)
        self.show()
    #..............

#Hide:
    def hide(self):
        pass
        for key in self.NumWidgetMap:
            self.NumWidgetMap[key].hide()
    #....................................
    
#Show Layout:
    def SetLayout(self, num):
        if num in self.NumWidgetMap:
            for key in self.NumWidgetMap:
                self.NumWidgetMap[key].hide()
        else: return
        self.NumWidgetMap[num].show()
    #........................................
    
#Register Layout:
    def AddLayout(self, num, layout):
        #Проверить layout на сущ в карте! 
        if not num in self.NumWidgetMap:
            self.w = QWidget() #QGroupBox()
            self.w.setLayout(layout)
            self.NumWidgetMap[num] = self.w
            self.layout.addWidget(self.w)
    #........................................

#Resize the Dockwidget:
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.parent() and hasattr(self.parent(), "sync_width"):
            self.parent().sync_width()   
#End of CLass DockWidget ##############



#CLass for holding signals:
class SignalEmitter(QObject):

#Signal to send text to GUI:
    text_written = pyqtSignal(str)
#End of CLass SignalEmitter#######



#CLass Terminal for output:
class GUITerminal(QWidget):

    #Singleton-templated
    
#OnLy object:
    _instance = None  

#Constructor:
    def __new__(cls, *args, **kwargs):
        if cls._instance is None: ###########################
            cls._instance = super(GUITerminal, cls).__new__(cls)
        return cls._instance ################################
    #########################################################

#Initializer:
    def __init__(self, parent=None):

    #Initialization is performed once:
        super().__init__(parent) ###########
        if hasattr(self, "_initialized"): ##
            return #########################
        ####################################
        
    #FLag of initialization:
        self._initialized = True ###########
        self.setObjectName("GUITerminal")######
        ####################################

    #Bunner of terminal:
        self.terminal_label = QLabel("Terminal") ############
        self.terminal_label.setObjectName("terminal_label")##
        #####################################################
        
    #Close terminal button:
        self.close_button = QPushButton() ############################
        self.close_button.setObjectName("close_terminal_button")######
        self.close_button.setIcon(QIcon("resources/OR-arrow.png")) 
        self.close_button.setIconSize(QSize(36, 36)) #################
        self.close_button.setFixedSize(48, 48) #######################
        self.close_button.setCursor(Qt.PointingHandCursor) ###########
        #connect this bottun with method:
        self.close_button.clicked.connect(self.hide) #################
        ##############################################################

    #Layout with label + button:
        top_bar = QHBoxLayout() ##################################
        top_bar.addWidget(self.terminal_label) ###################
        top_bar.addStretch()#Set the space between bunner&button##
        top_bar.addWidget(self.close_button) #####################
        top_bar.setContentsMargins(10, 0, 0, 0) ##################
        ##########################################################

    #field to show text:
        self.text_edit = QTextEdit()###########################
        self.text_edit.setReadOnly(True)#######################
        self.text_edit.setObjectName("terminal_text_editor")###
        #######################################################

    #group all components of terminal in layout:
        layout = QVBoxLayout() ##################
        layout.addLayout(top_bar) ###############
        layout.addWidget(self.text_edit) ########
        layout.setContentsMargins(5, 5, 5, 5) ###
        layout.setSpacing(0) ####################
        self.setLayout(layout) ##################
        #########################################

    #create and connect the signals:
        self.emitter = SignalEmitter() ####################
        self.emitter.text_written.connect(self.append_text)
        ###################################################

    #redirect sys.stdout&sys.stderr to the terminal:
        sys.stdout = self
        sys.stderr = self
    #End of initializing #######################################


#To appent the text into terminal:
    def append_text(self, text):
        self.text_edit.append(text)
    #End of text appender##########

#To show the text in the terminal:
    def write(self, text):
        #redirect sys.stdout&sys.stderr:
        if text.strip(): ##################################
            info = self.get_caller_info() #################
            formatted_text = f"<<< {text.strip()}" ########
            self.emitter.text_written.emit(formatted_text)#
    #End of write #########################################

#Needed for compatibility with sys.stdout&sys.stderr:
    def flush(self):
        pass ########################################
    #End of flush####################################

#To hide the terminal:
    def hide(self):
        self.setVisible(False)
    #End of hide terminal#####

#To show the terminal:
    def show(self):
        self.setVisible(True)
    #End of show terminal#####

#To determine from where the text came: 
    def get_caller_info(self):
        stack = inspect.stack() ###############################
        for frame in stack[1:]:  #Skip the write() method######
            module = inspect.getmodule(frame[0]) ##############
            if module and module.__file__:  #Checking that this is not a built-in module ###########
                file_name = module.__file__.split("/")[-1]  # Getting the file name ################
                class_name = frame[0].f_locals.get("self", None) # Checking called from the class###
                if class_name: #####################################################################
                    class_name = class_name.__class__.__name__ #####################################
                method_name = frame.function #######################################################

                if class_name:
                    return f"{file_name}::{class_name}.{method_name}()" ############################
                return f"{file_name}::{method_name}()" #############################################
        # if not defined ###########################################################################
        return "unknown"  

#End of CLass GUITerminal ########



#New Labeled QSlider:
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
		"""Загружает стили из файла crop_styles.css"""
		css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_styles.css")

		if not os.path.exists(css_file):
			print(f"Файл не найден: {css_file}")  # Проверка пути
			return

		with open(css_file, "r") as file:
			self.setStyleSheet(file.read())

	def update_slider_class(self):
		current_value = self.slider.value()
		max_value = self.slider.maximum()

		# Защищаемся от деления на ноль.
		# Если max_value == 0, сделаем ratio = 0.
		ratio = current_value / max_value if max_value != 0 else 0

		# Меняем "пороги" на основе ratio, а не фиксированных значений.
		if ratio < 0.3:      # соответствует < 30% от максимума    
			color_class = "green"
		elif ratio < 0.6:    # 30%..60%
			color_class = "yellow"
		elif ratio < 0.8:    # 60%..80%
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
		main_layout.setSpacing(1)              # 1 px между "верхним блоком" и слайдером
		main_layout.setAlignment(Qt.AlignTop)  # Прижимаем все виджеты к верхней границе

		# --- Горизонтальный layout: Label (слева) и LCD (справа)
		top_layout = QHBoxLayout()
		top_layout.setContentsMargins(0, 0, 30, 0)
		top_layout.setSpacing(1)              # 1 px между Label и LCD

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
		self.my_lcd.setMinimumWidth(100)

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
		main_layout.addLayout(top_layout)     # строка (метка + LCD)
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

