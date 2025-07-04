##### ###############################################
from PyQt5.QtGui import*####################### 
from PyQt5.QtCore import*###################### 
from PyQt5.QtWidgets import*###################
from PyQt5.QtGui import QScreen################
from PyQt5.QtWidgets import QFrame#############
from PyQt5 import QtGui, QtCore, QtWidgets#####
###############################################
#Qt__________________________________________
from PyQt5 import QtGui, QtCore, QtWidgets###
from PyQt5.QtWidgets import* ################
from PyQt5.QtCore import* ###################
from PyQt5.QtGui import* ####################
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

# __Class Main__
class MainWindow(QMainWindow):
    """
    Main application window that configures the GUI:
    sets theme, title, icon, central layout, dashboard, terminal,
    splitter, menu bar, and toolbar.
    """

    # Constructor
    def __init__(self):
        """
        Initialize MainWindow instance.
        Calls init_window() to set up all widgets and layout,
        then calls show_output() to display initial message.
        Inputs: none
        Returns: None
        """
        super().__init__()
        self.init_window()
        self.show_output()
    # End of constructor ###

    # Initializer of main window
    def init_window(self):
        """
        Set up the main window:
        - Apply visual theme
        - Configure window title and icon
        - Determine screen size and resize window
        - Create central widget with layout
        - Initialize dashboard dock and terminal output
        - Set up splitter to separate main view and terminal
        - Initialize menu bar and tool bar
        - Hide all pages initially
        - Print readiness message
        Inputs: none
        Returns: None
        """
        # Apply dark theme stylesheet
        self.apply_theme("dark.css")

        # Set window title and icon
        self.setWindowTitle("Archive Spectra Processing")
        self.setWindowIcon(QIcon("resources/main_icon.png"))

        # Obtain screen dimensions
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Resize window to desired fraction of screen (currently 100%)
        # Intended: int(screen_width * 0.8), int(screen_height * 0.8)
        self.window_width = int(screen_width)
        self.window_height = int(screen_height)
        self.resize(self.window_width, self.window_height)

        # Create central container widget
        self.center_widget = QGroupBox("___________________")
        self.center_widget.setObjectName('The_center_widget')
        self.center_widget.setContentsMargins(0, 0, 0, 0)
        self.center_layout = QVBoxLayout()
        self.center_layout.setContentsMargins(1, 4, 4, 0)
        self.center_widget.setLayout(self.center_layout)
        self.setCentralWidget(self.center_widget)

        # Main content widget inside center
        self.MainWidget = QGroupBox("T")
        self.MainWidget.setObjectName('MainWidget')
        self.MainLayout = QVBoxLayout()
        self.MainWidget.setLayout(self.MainLayout)

        # Map for left-side components
        self.MainWidgetsMap = dict()

        # Initialize the dashboard dock (right side)
        self.dashboard = None 
        self.init_dashboard()

        # Create and configure GUI terminal for logging output
        self.terminal = GUITerminal(self)
        self.terminal.setMinimumHeight(100)
        self.terminal.hide()
        # Redirect standard output and error to the terminal widget
        sys.stdout = self.terminal
        sys.stderr = self.terminal

        # Create vertical splitter to hold main view and terminal
        self.mainhor_splitter = QSplitter(Qt.Vertical)
        self.mainhor_splitter.setHandleWidth(2)
        self.mainhor_splitter.setObjectName('mainhorsplitter')

        # Add main widget and terminal to splitter
        self.mainhor_splitter.addWidget(self.MainWidget)
        self.mainhor_splitter.addWidget(self.terminal)

        # Insert splitter into the central layout
        self.center_layout.addWidget(self.mainhor_splitter)

        # Initialize menu bar and add initial menu line
        self.init_menu()
        self.add_menu_line()

        # Initialize tool bar
        self.init_toolbar()

        # Hide all pages until needed
        self.HidePages()

        # Re-apply dark theme (duplicate call – можно убрать)
        self.apply_theme("dark.css")

        # Notify that software is ready
        print("The software is ready to use.")
    # End of main window initializer #####################

    # Initialize dashboard dock widget
    def init_dashboard(self):
        """
        Create or show the dashboard dock widget on the right side:
        - If not created, instantiate QDashboard and add to main window
        - Configure allowed docking areas and size constraints
        - Initially hide the dashboard until needed
        Inputs: none
        Returns: None
        """
        if self.dashboard is None:
            # Create dashboard and allow docking on left or right
            self.dashboard = QDashboard(self)
            self.dashboard.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
            self.addDockWidget(Qt.RightDockWidgetArea, self.dashboard)

            # Prevent floating and set maximum width
            self.dashboard.setFloating(False)
            # Intended minimum width: int(self.window_width / 3)
            self.dashboard.setMaximumWidth(int(self.window_width / 2))
        else:
            # If dashboard exists, bring it to front
            self.dashboard.Show()

        # Hide dashboard by default
        self.dashboard.hide()
    # End of dashboard initialization ##################################################


    # Initialize MenuBar:
    def init_menu(self):
        """
        Create the main menu bar and populate it with 'View', 'Terminal', and 'Extensions' menus.
        Inputs: none
        Returns: None
        """
        menu_bar = self.menuBar()

        # View menu: switch between light and dark themes
        view_menu = menu_bar.addMenu("View")
        view_light_action = QAction("Light", self)
        view_dark_action = QAction("Dark", self)
        # Connect theme actions
        view_light_action.triggered.connect(lambda: self.apply_theme("light.css"))
        view_dark_action.triggered.connect(lambda: self.apply_theme("dark.css"))
        view_menu.addAction(view_light_action)
        view_menu.addAction(view_dark_action)

        # Terminal menu: show or hide the terminal output
        terminal_menu = menu_bar.addMenu("Terminal")
        terminal_output_action = QAction("Output", self)
        terminal_apart_action = QAction("Apart Run", self)
        # Connect terminal actions
        terminal_output_action.triggered.connect(lambda: self.show_output())
        terminal_apart_action.triggered.connect(lambda: self.show_apart())
        terminal_menu.addAction(terminal_output_action)
        terminal_menu.addAction(terminal_apart_action)

        # Extensions menu: load external plugins or scripts
        extensions_menu = menu_bar.addMenu("Extensions")
        extension_rootfolder_action = QAction("In Root Folder", self)
        # Connect extension loading action
        extension_rootfolder_action.triggered.connect(lambda: self.loadExtensionfromRoot())
        extensions_menu.addAction(extension_rootfolder_action)
    # End of menu initializer ################################################################


    # Draw menu line under the menu bar
    def add_menu_line(self):
        """
        Add a horizontal line below the menu bar to separate it from the content area.
        Inputs: none
        Returns: None
        """
        self.menu_line = QFrame(self)
        self.menu_line.setObjectName("menuline")
        self.menu_line.setFrameShape(QFrame.HLine)
        self.menu_line.setFrameShadow(QFrame.Sunken)
        # Position line at top of window content
        self.menu_line.setGeometry(0, self.menuBar().height(), self.width(), 2)
        # Update line position and size on window resize
        self.resizeEvent = self.update_menu_line
    # End of drawing menu line ####################


    # Update menu line geometry on resize
    def update_menu_line(self, event):
        """
        Handle window resize events to reposition and resize the menu separator line.
        Inputs:
            event (QResizeEvent): the resize event containing new dimensions
        Returns:
            None
        """
        # Recalculate line width and position
        self.menu_line.setGeometry(0, self.menuBar().height(), self.width(), 2)
        super().resizeEvent(event)
    # End of updating menu line size############################################


    # Menu -> Terminal -> Output action
    def show_output(self):
        """
        Show the terminal widget for capturing stdout and stderr.
        Inputs: none
        Returns: None
        """
        try:
            self.terminal.show()
        except AttributeError:
            # Terminal not yet initialized
            pass
    # End of Terminal output action's method###


    def show_apart(self):
        """
        Placeholder for 'apart run' functionality.
        Inputs: none
        Returns: None
        """
        # Not implemented yet
        pass


    # Menu -> Extensions -> In Root Folder action
    def loadExtensionfromRoot(self):
        """
        Load and initialize extensions located in the application's root folder.
        Inputs: none
        Returns: None
        """
        # Not implemented yet
        pass
    # End of Menu & its components############
    # Initialize toolbar and load model buttons
    def init_toolbar(self):
        """
        Initialize or recreate the toolbar on the left side,
        then populate it with buttons for each discovered model.
        Inputs: none
        Returns: None
        """
        # Remove existing toolbar if present
        if hasattr(self, "Tb") and self.Tb:
            self.removeToolBar(self.Tb)

        # Create new toolbar
        self.Tb = QToolBar("TOOLS")
        self.addToolBar(Qt.LeftToolBarArea, self.Tb)
        self.Tb.setAllowedAreas(Qt.LeftToolBarArea | Qt.RightToolBarArea)
        # Fix toolbar width to 1/18 of window width
        self.Tb.setMaximumWidth(int(self.window_width / 18))
        self.Tb.setMinimumWidth(int(self.window_width / 18))
        self.Tb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.Tb.setMovable(False)

        # Discover available models in filesystem
        found_models = self.Scan_models()

        # Reset structures for fresh population
        self.models_info = []       # List of dicts for each model
        self.MainWidgetsMap = {}    # Map index -> left widget
        self.buttons = []           # List of toolbar buttons
        self.TbBtnMap = {}          # Map button -> click handler
        # Ensure the button itself is square and matches the toolbar width
        btn_size = int(self.window_width / 18)

        # Add each model as a button in the toolbar
        for idx, model_data in enumerate(found_models):
            model_name = model_data["name"]
            model_instance = model_data["model_instance"]

            # Create animated tool button with gray and orange icons
            button = AnimatedToolButton(
                model_data["icon_gr"],
                model_data["icon_or"],
                "",
                self.window_width,
                self.window_height,
                self
            )
            # Force button to be a square of exact dimensions
            button.setFixedSize(btn_size, btn_size)
            # Show only the icon, no extra space for text
            button.setToolButtonStyle(Qt.ToolButtonIconOnly)
            # Remove default padding/margins so icon can fill whole area
            button.setStyleSheet("QToolButton { padding: 0px; margin: 0px; }")

            # Scale source pixmaps to exactly fill the button
            pixmap_gr = model_data["icon_gr"].pixmap(btn_size, btn_size)
            pixmap_or = model_data["icon_or"].pixmap(btn_size, btn_size)
            icon_gr_scaled = QIcon(pixmap_gr)
            icon_or_scaled = QIcon(pixmap_or)

            # Assign the scaled icons
            button.setIcon(icon_gr_scaled)
            button.setIconSize(QSize(btn_size, btn_size))

            # If AnimatedToolButton supports hover/active states, set the alternate icon
            if hasattr(button, "setHoverIcon"):
                button.setHoverIcon(icon_or_scaled)
            # Finally set the icon size
            button.setIconSize(QSize(btn_size, btn_size))


            self.Tb.addWidget(button)
            self.buttons.append(button)

            # Map button to handler with bound index
            
            self.TbBtnMap[button] = lambda _checked, i=idx: self.on_toolbar_button_clicked(i)

            # Add model's left widget and right layout to dashboard and main map
            left_widget = model_instance.getLeftWidget()
            right_layout = model_instance.getRightLayout()
            self.dashboard.AddLayout(idx, right_layout)
            self.AddMainWidget(idx, left_widget)

            print(f"Model {model_name} has been installed.")

            # Store metadata for runtime switching
            self.models_info.append({
                "index": idx,
                "name": model_name,
                "model_instance": model_instance,
                "button": button
            })

        # Connect button signals to their handlers
        for button in self.buttons:
            button.clicked.connect(
                lambda checked, b=button: self.TbBtnMap[b](checked)
            )
    # End of toolbar initialization


    def on_toolbar_button_clicked(self, index):
        """
        Handle toolbar button click:
        - Highlight the clicked button and dim others
        - Switch the dashboard page to the selected model index
        Inputs:
            index (int): index of the selected model
        Returns: None
        """
        for info in self.models_info:
            if info["index"] == index:
                info["button"].setActive(True)
            else:
                info["button"].setActive(False)
        self.SetPage(index)


    def Scan_models(self, base_path=None):
        """
        Scan the filesystem for model.<Name> directories, load each module,
        verify required classes, and instantiate model objects with their icons.
        Inputs:
            base_path (str, optional): directory to scan (defaults to application dir)
        Returns:
            List[Dict]: each dict contains keys:
                'name': model name (str)
                'model_instance': instance of the model class
                'icon_gr': QIcon for gray icon
                'icon_or': QIcon for orange icon
        """
        try:
            if hasattr(sys, '_MEIPASS'):
                base_path = sys._MEIPASS
            else:
                base_path = os.getcwd()
        except NameError:
            base_path = os.path.dirname(os.path.abspath(sys.argv[0]))
            print("Exception determining base path:", base_path)

        print("Scanning models in:", base_path)

        found_models = []
        for entry in os.listdir(base_path):
            # Only consider directories named "model.<Name>"
            if not entry.startswith("model"):
                continue
            folder_path = os.path.join(base_path, entry)
            if not os.path.isdir(folder_path):
                continue

            model_name = entry[5:]  # убираем "model."
            py_path = os.path.join(folder_path, f"model_{model_name}.py")
            icon_gr_path = os.path.join(folder_path, f"FIATb-GR-{model_name}.png")
            icon_or_path = os.path.join(folder_path, f"FIATb-OR-{model_name}.png")

            # Skip if any required file is missing
            if not (os.path.isfile(py_path) and os.path.isfile(icon_gr_path) and os.path.isfile(icon_or_path)):
                continue

            # Dynamically import the module
            module_name = f"model_{model_name}"
            spec = importlib.util.spec_from_file_location(module_name, py_path)
            if spec is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Verify presence of model, Left, and Right classes
            class_model = getattr(module, module_name, None)
            left_cls = getattr(module, "Left", None)
            right_cls = getattr(module, "Right", None)
            if not (class_model and left_cls and right_cls):
                continue
            if not (issubclass(left_cls, QWidget) and issubclass(right_cls, QVBoxLayout)):
                continue

            # Instantiate model, passing self if constructor accepts parent
            sig = inspect.signature(class_model.__init__)
            if "parent" in sig.parameters:
                model_instance = class_model(self)
            else:
                model_instance = class_model()
            if hasattr(model_instance, "set_dialog_parent"):
                model_instance.set_dialog_parent(self)

            icon_gr = QIcon(icon_gr_path)
            icon_or = QIcon(icon_or_path)
            print(f"Model {model_name} has been found.")

            found_models.append({
                "name": model_name,
                "model_instance": model_instance,
                "icon_gr": icon_gr,
                "icon_or": icon_or
            })

        return found_models


    # CONTROL METHODS:

    def SetPage(self, index):
        """
        Switch the displayed main widget and dashboard layout to the given index.
        Inputs:
            index (int): identifier of the page/model to display
        Returns: None
        """
        # Update main content area
        self.SetMainWidget(index)
        # Update dashboard to corresponding layout
        self.dashboard.SetLayout(index)


    def AddMainWidget(self, num, w):
        """
        Add a widget to the main layout map if not already present.
        Inputs:
            num (int): key/index under which to store the widget
            w (QWidget): the widget to add
        Returns: None
        """
        if num not in self.MainWidgetsMap:
            self.MainWidgetsMap[num] = w
            self.MainLayout.addWidget(w)


    def SetMainWidget(self, num):
        """
        Hide all main widgets, then show only the widget at the specified index.
        Inputs:
            num (int): index of widget to display
        Returns: None
        """
        if num in self.MainWidgetsMap:
            # Hide every widget
            for key, widget in self.MainWidgetsMap.items():
                widget.hide()
            # Show only the selected one
            self.MainWidgetsMap[num].show()


    def HidePages(self):
        """
        Hide the dashboard dock and all main content widgets.
        Inputs: none
        Returns: None
        """
        # Hide the dashboard dock widget
        self.dashboard.hide()
        # Hide every widget in the main content area
        for widget in self.MainWidgetsMap.values():
            widget.hide()


    def apply_theme(self, theme_file):
        """
        Load and apply a Qt stylesheet from file, and adjust matplotlib theme.
        Inputs:
            theme_file (str): filename of the CSS stylesheet to load
        Returns: None
        """
        def resource_path(filename):
            # Handle PyInstaller _MEIPASS resource path if present
            if hasattr(sys, '_MEIPASS'):
                return os.path.join(sys._MEIPASS, filename)
            return os.path.abspath(filename)

        try:
            real_theme_path = resource_path(theme_file)
            with open(real_theme_path, "r", encoding='windows-1252') as file:
                app.setStyleSheet(file.read())
                print(f"Stylesheet: {real_theme_path}")
                apply_matplotlib_theme(theme_file)
        except FileNotFoundError:
            print(f"Fail: {theme_file} not found.")


    def Qprint(self, message):
        """
        Simple debug print to console and terminal widget.
        Inputs:
            message (Any): the content to print
        Returns: None
        """
        print("FROM modules...")
        print(message)


def apply_matplotlib_theme(theme_name: str):
    """
    Configure matplotlib rcParams for light or dark themes based on filename.
    Inputs:
        theme_name (str): name of the theme file containing 'dark' or not
    Returns: None
    """
    from matplotlib import rcParams

    if "dark" in theme_name.lower():
        # Dark theme settings
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
            'axes.labelsize': 28,
            'xtick.labelsize': 26,
            'ytick.labelsize': 26,
            'figure.dpi': 150,
            'font.family': 'serif',
            'font.size': 28})
    else:
        # Light theme settings
        rcParams.update({
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'savefig.facecolor': 'white',
            'axes.edgecolor': 'black',
            'axes.labelcolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black',
            'grid.color': 'gray',
            'axes.labelsize': 28,
            'xtick.labelsize': 26,
            'ytick.labelsize': 26,
            'figure.dpi': 150,
            'font.family': 'serif',
            'font.size': 28
        })

    # __main__ entry point
if __name__ == "__main__":
    """
    Create the QApplication, instantiate and show MainWindow,
    then start the Qt event loop.
    """
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
