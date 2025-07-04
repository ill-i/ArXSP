#Qt5 files:
###############################################
from PyQt5.QtGui import*####################### 
from PyQt5.QtCore import*###################### 
from PyQt5.QtWidgets import*###################
from PyQt5.QtGui import QScreen################
from PyQt5.QtWidgets import QFrame#############
from PyQt5 import QtGui, QtCore, QtWidgets#####
###############################################

#system files:
import sys#############
import os #############
import importlib.util #
#######################

#support files:
from pickle import TRUE #############
from PIL import Image ###############
from pathlib import Path ############
#####################################

#internal files:
from GUIcomp import *
##########################



# __CLass Main________________
class MainWindow(QMainWindow):

#Constructor:
    def __init__(self):
        super().__init__()
        self.init_window()
    #End of constructor###



#Initializer of main window:
    def init_window(self):

    #Calling dark theme:
        self.apply_theme("dark.css")

    #Setting Title & Icon of Main Window:
        self.setWindowTitle("Archive Spectra Processing")######################
        self.setWindowIcon(QIcon("resources/main_icon.png"))

    #Obtain Screen Sizes:
        screen = QApplication.primaryScreen()#########
        screen_geometry = screen.availableGeometry()##
        screen_width = screen_geometry.width()########
        screen_height = screen_geometry.height()######
        ##############################################
        #Setting the main window size as a% of the screen:
        # 80% of screen width_____________________________
        self.window_width = int(screen_width)# * 0.8)#######
        # 80% of screen height____________________________
        self.window_height = int(screen_height)# * 0.8)#####  
        self.resize(self.window_width,self.window_height)#
        ##################################################

    #HoLder of MainWidget & Terminal set as CentralWidget:
        self.center_widget = QGroupBox("___________________") 
        self.center_widget.setObjectName('The_center_widget')
        self.center_widget.setContentsMargins(0, 0, 0, 0)
        #####################################################
        self.center_layout = QVBoxLayout()###################
        self.center_layout.setContentsMargins(1, 4, 4, 0)
        self.center_widget.setLayout(self.center_layout)#####
        #####################################################
        self.setCentralWidget(self.center_widget)############
        #####################################################

    #Main Widget which is the screen:
        self.MainWidget = QGroupBox("T")###########  
        self.MainWidget.setObjectName('MainWidget')
        self.MainLayout = QVBoxLayout()############
        self.MainWidget.setLayout(self.MainLayout)#
        ###########################################

    #Map for all left components widget:
        self.MainWidgetsMap = dict()########
        ####################################

    #Initialize DockWidget on right side:
        self.dashboard = None 
        self.init_dashboard()

    #GUITerminal(singleton) to show output:
        self.terminal = GUITerminal(self) #####
        self.terminal.setMinimumHeight(100) ###
        self.terminal.hide() ##################
        # Redirect print() to the terminal:
        sys.stdout = self.terminal ############
        sys.stderr = self.terminal ############
        #######################################

    #Splitter creates as a widget which will be added into layout:
        self.mainhor_splitter = QSplitter(Qt.Vertical) ###############
        self.mainhor_splitter.setHandleWidth(4) ######################
        self.mainhor_splitter.setObjectName('mainhorsplitter') #######
        ##############################################################

        #To splite MainWidget & Terminal add to splitter widget:
        self.mainhor_splitter.addWidget(self.MainWidget) ######
        self.mainhor_splitter.addWidget(self.terminal)   ######
        #######################################################

        #Splitter installs into central layout which is in central widget:
        self.center_layout.addWidget(self.mainhor_splitter) #############
        ################################################################# 

    #Initialize menu-bar:
        self.init_menu()
        self.add_menu_line()

    #Initialize tool-bar:
        self.init_toolbar()

    #Hide all pages
        self.HidePages() 

    #Calling dark theme:
        self.apply_theme("dark.css")

    #Check output:
        print("The software is ready to use.")
    #End of main window initializer#####################



#Initialize dash-board:
    def init_dashboard(self):
    # If self.dashboard is not exist:
        if self.dashboard is None:
            self.dashboard = QDashboard(self)
            self.dashboard.setAllowedAreas(Qt.RightDockWidgetArea|Qt.LeftDockWidgetArea)
            self.addDockWidget(Qt.RightDockWidgetArea,  self.dashboard)

        #Temp size definition and Priority:
            self.dashboard.setFloating(False) ##########################################
            self.dashboard.setMinimumWidth(800) ########################################
            self.dashboard.setMaximumWidth(1200) #######################################
            self.dashboard.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) #
            ############################################################################
        else: self.dashboard.Show()
        self.dashboard.hide()
    #End of dashboard initialization##################################################



#Initialize MenuBar:
    def init_menu(self):
    #Create menu bar with several menu: 
        menu_bar = self.menuBar()
        #########################

    #Create menu "View" in menu bar:
        view_menu = menu_bar.addMenu("View") 
        ####################################
        #Create actions of "View" menu: 
        view_light_action = QAction("Light", self) ##
        view_dark_action = QAction("Dark", self) ####
        #############################################
        #Connect "View" actions with method:
        view_light_action.triggered.connect(lambda: self.apply_theme("light.css")) ##
        view_dark_action.triggered.connect(lambda: self.apply_theme("dark.css")) ####
        #############################################################################
        #Add "View" actions into "View" menu:
        view_menu.addAction(view_light_action) #####
        view_menu.addAction(view_dark_action) ######
        ############################################

    #Create menu "Terminal" in menu bar:
        terminal_menu = menu_bar.addMenu("Terminal") 
        ############################################
        #Create actions of "Terminal" menu: 
        terminal_output_action = QAction("output", self) ##
        terminal_apart_action = QAction("apart run", self)#
        ###################################################
         
        #Connect "View" actions with method:
        terminal_output_action.triggered.connect(lambda: self.show_output()) ##
        terminal_apart_action.triggered.connect(lambda: self.show_apart()) ####
        #######################################################################
        #Add "terminal" actions into "Terminal" menu:
        terminal_menu.addAction(terminal_output_action) ##
        terminal_menu.addAction(terminal_apart_action) ##
        ##################################################

    #Create menu "Extensions" in menu bar:
        extensions_menu = menu_bar.addMenu("Extensions")
        ################################################
        #Create actions of "Extensions" menu: 
        extension_rootfolder_action = QAction("In root folder", self) ##
        ################################################################
        #Connect "Extensions" actions with method:
        extension_rootfolder_action.triggered.connect(lambda: self.loadExtensionfromRoot()) ##
        ######################################################################################
        #Add "Extensions" actions into "Extensions" menu:
        extensions_menu.addAction(extension_rootfolder_action) ##
    #End of menu initializer ################################################################
       
    

#Draw menu line:
    def add_menu_line(self):
        # Create a horizontal line:
        self.menu_line = QFrame(self) #########################################
        self.menu_line.setObjectName("menuline") ##############################
        self.menu_line.setFrameShape(QFrame.HLine) ############################
        self.menu_line.setFrameShadow(QFrame.Sunken) ##########################
        self.menu_line.setGeometry(0, self.menuBar().height(), self.width(), 2)
        # Update line size when window changes:
        self.resizeEvent = self.update_menu_line##
    #End of drawing menu line ####################

#Update menu line:
    def update_menu_line(self, event):
        #Updating line size when window resizes:
        self.menu_line.setGeometry(0, self.menuBar().height(), self.width(), 2)
        super().resizeEvent(event)#############################################
    #End of updating menu line size############################################

#Menu Actions:
#Menu->Terminal->output action's method:
    def show_output(self):
        try:
            self.terminal.show()
        except AttributeError:
            pass
    #End of Terminal output action's method### 

    def show_apart(self):
        pass


#Not Implemented yet..........................
#Menu->Extensions->rootfolder action's method:
    def loadExtensionfromRoot(self):
        pass
    #End of Menu & it's components############


#TOOL-BAR & LOAD MODELS:
#Initialize tool-bar:
    def init_toolbar(self): 
    #If the toolbar already exists, delete it:
        if hasattr(self, "Tb") and self.Tb: #
            self.removeToolBar(self.Tb) #####
        #####################################
    
    #Create a new toolbar and set on the left:
        self.Tb = QToolBar("TOOLS") #####################################
        self.addToolBar(Qt.LeftToolBarArea, self.Tb) ####################
        self.Tb.setAllowedAreas(Qt.LeftToolBarArea | Qt.RightToolBarArea)
        #Set not to move the toolbar:
        self.Tb.setMovable(False) ########################################
        #################################################################

        found_models = self.Scan_models()  # ваш метод из вопроса

        # Очищаем структуры, чтобы по-новому заполнить
        self.models_info = []         # список словарей (каждый словарь — одна модель)
        self.MainWidgetsMap = {}      # индекс -> виджет (left)
        #self.dashboard.ClearAll()     # допустим, в Dashboard есть метод очистки всего

        # Это нам понадобится для переключения:
        self.buttons = []
        self.TbBtnMap = dict()   # button -> функция или индекс
    
        # В цикле добавляем все найденные модели
        for idx, model_data in enumerate(found_models):
            model_name = model_data["name"]                   # например "Cut", "View" и т.д.
            model_instance = model_data["model_instance"]     # основной класс (model_Xxx)
    
            # 1) Создаем кнопки для тулбара
            icon_gr = model_data["icon_gr"]
            icon_or = model_data["icon_or"]
            
            # Пример: AnimatedToolButton(...) — как у вас было
            button = AnimatedToolButton(
                icon_gr,
                icon_or,
                "____",  # или какой-то "title"
                self
            )
            self.Tb.addWidget(button)
            self.buttons.append(button)
    
            # 2) Сохраняем в TbBtnMap "что делать по нажатию"
            #    Можно сразу записать lambda, которая вызывает SetPage(idx).
            self.TbBtnMap[button] = lambda _checked, i=idx: self.on_toolbar_button_clicked(i)
    
            # 3) Добавляем left/right в ваш Dashboard + Mапы
            left_widget = model_instance.getLeftWidget()
            right_layout = model_instance.getRightLayout()
    
            # Допустим, Dashboard умеет добавлять layout-ы по индексу
            self.dashboard.AddLayout(idx, right_layout)
            self.AddMainWidget(idx, left_widget)
            #left_widget.hide()  # изначально скрываем
    
            print(f"Model {model_name} has been installed.")

            # Чтобы потом удобно переключаться в SetPage(idx),
            # мы сохраним meta-информацию в self.models_info
            self.models_info.append({
                "index": idx,
                "name": model_name,
                "model_instance": model_instance,
                "button": button
            })
    
        # Подписываем все кнопки на on_toolbar_button_clicked (общий обработчик),
        # но в этот раз у нас уже есть index для каждой:
        for button in self.buttons:
            button.clicked.connect(
                lambda checked, b=button: self.TbBtnMap[b](checked)
            )

    #End of toolbar & it's components Initialization#####################

#Action on toolbar button clicked:
    def on_toolbar_button_clicked(self, index):
    # 1) Меняем активность (оранжевый/серый) у кнопок:
        clicked_button = None
        for info in self.models_info:
            if info["index"] == index:
                clicked_button = info["button"]
                info["button"].setActive(True)
            else:
                info["button"].setActive(False)
    
        # 2) Переключаемся на нужную модель в dashboard
        self.SetPage(index)

#Search & Install Models:
    def Scan_models(self,base_path=None):

        #Сканиурет заданный каталог (по умолчанию текущий),
        #ищет подпапки вида model.Xxx и внутри них файлы:
        #  - model_Xxx.py
        #  - FIATb-GR-Xxx.png
        #  - FIATb-OR-Xxx.png
        #Проверяет, есть ли в модуле классы (model_Xxx, Left, Right),
        #и создаёт из них объект model_Xxx() + QIcon'ы.
        #Возвращает список найденных моделей.
    
    
        if base_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
    
        print(base_path)
    
        # Дальше — как в предыдущем коде:
        found_models = []
        
        for entry in os.listdir(base_path):
            # Ищем директории, начинающиеся на "model."
            if entry.startswith("model."):
                folder_path = os.path.join(base_path, entry)
                if not os.path.isdir(folder_path):
                    continue
                
                # Получим имя модели (то, что после "model.")
                # Например, "Cut", "View" и т.д.
                model_name = entry[6:]  # убираем "model."
        
                # Предполагаем, что внутри папки лежит файл: model_<Name>.py
                # Пример: model_Cut.py
                py_filename = f"model_{model_name}.py"
                py_path = os.path.join(folder_path, py_filename)
        
                # Иконки: FIATb-GR-Name.png и FIATb-OR-Name.png
                icon_gr_name = f"FIATb-GR-{model_name}.png"
                icon_or_name = f"FIATb-OR-{model_name}.png"
                icon_gr_path = os.path.join(folder_path, icon_gr_name)
                icon_or_path = os.path.join(folder_path, icon_or_name)
        
                # Проверяем, что всё существует
                if not (os.path.isfile(py_path) and
                        os.path.isfile(icon_gr_path) and
                        os.path.isfile(icon_or_path)):
                    continue
        
                # Импортируем модуль динамически
                # (чтобы классы были доступны и после выхода из функции,
                #  мы зарегистрируем модуль в sys.modules)
                module_name_for_import = f"model_{model_name}"  # уникальное имя
                spec = importlib.util.spec_from_file_location(module_name_for_import, py_path)
                if spec is None:
                    continue
        
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name_for_import] = module
                spec.loader.exec_module(module)
        
                # Проверяем, что в модуле есть 3 класса: 
                #   1) model_<Name>, 2) Left, 3) Right
                class_model_name = f"model_{model_name}"
                if all(hasattr(module, c) for c in (class_model_name, "Left", "Right")):
                    model_cls = getattr(module, class_model_name)
                    left_cls  = getattr(module, "Left")
                    right_cls = getattr(module, "Right")
        
        
                    if issubclass(right_cls, QVBoxLayout) and issubclass(left_cls, QWidget):
                        
                        # Смотрим сигнатуру конструктора
                        signature_init = inspect.signature(model_cls.__init__)
        
                        model_instance = None

                        # Если нужен именно 'parent' в сигнатуре, можно проверить так:
                        if "parent" in signature_init.parameters:
                            # Создаём экземпляр с родителем (если parent=None, передадим None)
                            model_instance = model_cls(self)
                        else:
                        # Создаём экземпляр без аргументов
                            model_instance = model_cls()

                        if hasattr(model_instance, "set_dialog_parent"):
                            print(f"Model {model_name} has atribute set_dialog_parent." )
                            model_instance.set_dialog_parent(self)

                        # Создаём QIcon
                        icon_gr = QIcon(icon_gr_path)
                        icon_or = QIcon(icon_or_path)
        

                        print(f"Model {model_name} has been found." )
                        
                        found_models.append({
                            "name": model_name,
                            "module": module,
                            "model_instance": model_instance,
                            "icon_gr": icon_gr,
                            "icon_gr_name": str(icon_gr_name) + "",  # <-- сохраняем имя файла
                            "icon_or": icon_or,
                            "icon_or_name": str(icon_or_name)+ ""   # <-- сохраняем имя файла
                        })
        
        return found_models



#CONTROL METHODS:
#Set Main widget & Dashboard Layout:
    def SetPage(self, index):       
    # Show needed widget in MainWidget:
        self.SetMainWidget(index)
    # Show needed layout in  dashboard:
        self.dashboard.SetLayout(index)
#........................................

#Add Widgets into Widgets map:  
    def AddMainWidget(self, num, w):
    #Check if layout exists in map & add if not:
        if not num in self.MainWidgetsMap:            
            self.MainWidgetsMap[num] = w
            self.MainLayout.addWidget(w)
#........................................
        
#Show & hide widgets:
    def SetMainWidget(self, num):
        #hide all widgets on MainWidget
        if num in self.MainWidgetsMap:
            for key in self.MainWidgetsMap:
                self.MainWidgetsMap[key].hide()
        else: return
        #show only needed widget:
        self.MainWidgetsMap[num].show()
#................................................

#Hide all pages:
    def HidePages(self):
        #hide dashboard widget not dashboard
        self.dashboard.hide() ##############
        #hide all widgets on MainWidget: 
        for key in self.MainWidgetsMap:
            self.MainWidgetsMap[key].hide()
#................................................

#Calling dark/light theme:
    def apply_theme(self, theme_file):
        try:
            with open(theme_file, "r") as file:
                app.setStyleSheet(file.read())
                #self.setStyleSheet(file.read())
                print(f"Stylesheet: " + str(theme_file))

                apply_matplotlib_theme(theme_file)


                #self.left.figure.set_facecolor(rcParams['figure.facecolor'])
                #if self.left.figure.axes:
                    #self.left.figure.axes[0].set_facecolor(rcParams['axes.facecolor'])
                #self.left.canvas.draw()


        except FileNotFoundError:
            print(f"Fail {theme_file} not found.")
#.................................................

    def Qprint(self, message):
        print("FROM modules...")
        print(message)
#...............................


def apply_matplotlib_theme(theme_name: str):
    from matplotlib import rcParams

    if "dark" in theme_name.lower():
        # Тёмная тема
        rcParams.update({
            'axes.facecolor': '#1e1e1e',
            'figure.facecolor': '#1e1e1e',
            'savefig.facecolor': '#1e1e1e',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white',
            'grid.color': '#444444',
            'axes.labelsize': 30,
            'xtick.labelsize': 28,
            'ytick.labelsize': 28,
            'font.size': 30
        })
    else:
        # Светлая тема
        rcParams.update({
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'axes.edgecolor':  'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black',
            'grid.color': 'black',
            'axes.labelsize': 30,
            'xtick.labelsize': 28,
            'ytick.labelsize': 28,
            'font.size': 30
        })
#..........................................
#'axes.edgecolor':  'black',
#'grid.color': '#cccccc',
          #  'axes.labelcolor': 'black',
           # 'xtick.color': 'black',
            #'ytick.color': 'black',
            #'text.color': 'black',
             #'grid.color': '#444444',
        
# __main____________________________
####################################
if __name__ == "__main__":

    #####CONFIG#####################
    app = QApplication(sys.argv)####

    #####RUN MAIN WINDOW############
    win = MainWindow()##############
    win.show()######################
    sys.exit(app.exec_())###########

