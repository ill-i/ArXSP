#Qt________________________________________
from PyQt5 import QtGui, QtCore, QtWidgets#
from PyQt5.QtWidgets import* ##############
from PyQt5.QtCore import* #################
from PyQt5.QtCore import pyqtSignal########
from PyQt5.QtGui import* ##################
from PyQt5.QtGui import QResizeEvent ######
from PyQt5.QtGui import QIcon, QPixmap ####
from PyQt5.QtCore import QSize ############
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QColorDialog
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


class model_intensity(object):
    """
    Encapsulates loading, displaying, and aligning a FITS spectrum.
    Holds references to the main window, data, display settings, and UI components.
    """

    # ────────────────────────────────────────────────────────────────────────────
    # INITIALIZER
    # ────────────────────────────────────────────────────────────────────────────
    def __init__(self, parent=None):
        """
        Initialize the alignment model.

        Args:
            parent (QMainWindow): reference to the main application window,
                                  used as parent for file dialogs.
        """
        super().__init__()

        # Parent QMainWindow (needed for dialogs)
        self.main_parent = parent

        # Dimensions of the main window for laying out controls
        self.win_w = parent.window_width
        self.win_h = parent.window_height

        # Data holders
        self.arx_data = None         # ArxData instance wrapping the FITS file
        self.data = None             # numpy array of pixel values
        self.data_header = None      # FITS header metadata
        self.data_editor = None      # ArxSpectEditor instance for alignment ops
        self.isImgLoaded = False     # Has an image been successfully loaded?
        self.prev_spectrum = None    # Store previous spectrum for "undo"

        # UI components, created here but inserted into the main window elsewhere
        self.__left = Left(self)     # Image display panel
        self.__right = Right(self)   # Control panel (sliders, buttons, etc.)

        # Display settings
        self.colorMap = cv2.COLORMAP_VIRIDIS  # default OpenCV colormap
        self.lineThickness = 6                # default guide-line thickness (px)
        self.lineColor = (255, 79, 0)         # default guide-line color (BGR tuple)

        # Line positions, as percentages (0–100)
        self.upValue = 0
        self.downValue = 0

        # Simple message box for errors
        self.msg = QMessageBox()
        self.msg.setWindowTitle("Error:")
        self.msg.setStandardButtons(QMessageBox.Ok)

        # Placeholder for any other previous state
        self.prev = None


    # ────────────────────────────────────────────────────────────────────────────
    # SETTERS
    # ────────────────────────────────────────────────────────────────────────────
    def set_dialog_parent(self, parent):
        """
        Set the QMainWindow parent for any subsequent dialogs.

        Args:
            parent (QMainWindow): main application window.
        """
        self.main_parent = parent

    def setColorMap(self, colorMap):
        """
        Change the displayed colormap and refresh the image.

        Args:
            colorMap (int): OpenCV colormap constant.
        """
        self.colorMap = colorMap
        self.ImgReLoad()  # redraw with new colormap

    def setLineThickness(self, thickness):
        """
        Update the thickness used when drawing the alignment lines.

        Args:
            thickness (int): line thickness in pixels.
        """
        self.lineThickness = thickness

    def setLineColor(self, color):
        """
        Update the BGR color used when drawing the alignment lines.

        Args:
            color (tuple): (B, G, R) tuple.
        """
        self.lineColor = color

    def setUpLineValue(self, value):
        """
        Update the 'upper' line position (as percentage) and redraw.

        Args:
            value (int): percentage from top where the upper line is drawn.
        """
        if self.isImgLoaded:
            self.upValue = value
            self.__left.PaintLines(self.upValue, self.downValue)

    def setDownLineValue(self, value):
        """
        Update the 'lower' line position (as percentage) and redraw.

        Args:
            value (int): percentage from bottom where the lower line is drawn.
        """
        if self.isImgLoaded:
            self.downValue = value
            self.__left.PaintLines(self.upValue, self.downValue)


    # ────────────────────────────────────────────────────────────────────────────
    # GETTERS
    # ────────────────────────────────────────────────────────────────────────────
    def getLeftWidget(self):
        """
        Returns:
            Left: the image‐display panel.
        """
        return self.__left

    def getRightLayout(self):
        """
        Returns:
            Right: the control panel layout.
        """
        return self.__right

    def getMainParent(self):
        """
        Returns:
            QMainWindow: the main application window.
        """
        return self.main_parent


    # ────────────────────────────────────────────────────────────────────────────
    # CORE METHODS
    # ────────────────────────────────────────────────────────────────────────────
    def ImgLoad(self):
        """
        Prompt the user to select a FITS file, load its data and header,
        then display it and reset the control sliders.
        """
        try:
            # File dialog for FITS selection
            self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.main_parent,
                'Single File',
                QtCore.QDir.rootPath(),
                '(*.fits *.fit)'
            )

            # If user cancels, exit early
            if not self.fileName:
                print("Err: fits file has not been chosen")
                return

            try:
                # Load data via ArxData helper
                self.arx_data = ArxData(self.fileName)
                self.data = self.arx_data.get_data()
                self.data_header = self.arx_data.get_header()
                self.isImgLoaded = True
                print(f"data header: {self.data_header}")
                print(f"fits data: {self.data}")
            except Exception as err:
                # Handle read errors
                self.isImgLoaded = False
                print(f"Error reading {self.fileName}: {err}")
                return

            # Refresh display and controls
            self.__left.imgLoad()
            self.__right.reset_sliders()

        except Exception as err:
            # Catch any other unexpected errors
            print(f"Unexpected {err=}, {type(err)=}")
            return

    def alignData(self):
        """
        Apply distortion correction to the currently loaded spectrum,
        based on the slider positions and polynomial order.
        """
        if self.arx_data is None:
            print("EMPTY DATA TO ALIGN")
            return

        # Read polynomial order from UI
        order = self.__right.orderSpinBox.value()
        self.data_editor = ArxSpectEditor(self.arx_data)

        # Save current spectrum for undo
        self.prev_spectrum = self.arx_data

        # Convert percentages to pixel offsets
        h = self.__left.getHeight()
        up = int(h * self.upValue / 100)
        down = h - int(h * self.downValue / 100)

        # Perform correction and reload data
        self.arx_data = self.data_editor.SDistorsionCorr(up, down, order)
        self.data = self.arx_data.get_data()

        # Refresh display and reset sliders
        self.ImgReLoad()
        self.__right.reset_sliders()

    def backToSpectrum(self):
        """
        Revert to the previous (pre‐aligned) spectrum, if available.
        """
        if self.prev_spectrum is None:
            return

        if self.isImgLoaded:
            # Restore from backup
            self.arx_data = self.prev_spectrum
            self.data = self.arx_data.get_data()
            self.data_header = self.arx_data.get_header()
            self.isImgLoaded = True

            # Refresh UI
            self.__left.imgLoad()
            self.__right.reset_sliders()

    def ImgReLoad(self):
        """
        Redraw the currently loaded image without re‐reading the FITS file.
        """
        if not getattr(self, 'fileName', None):
            return
        if not getattr(self, 'isImgLoaded', False):
            return

        self.__left.imgLoad()
        self.__right.reset_sliders()

    def polynomLoad(self):
        """
        Prompt for a CSV of polynomials, show them in a dialog,
        and mark the best match based on the FITS header date.
        """
        try:
            # CSV file dialog
            self.path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.main_parent,
                'Single File',
                QtCore.QDir.rootPath(),
                '(*.csv)'
            )
            print(self.path)

            if not self.path:
                print("No file selected.")
                return

            # Load polynomial entries
            polynoms = CsvPolynom.upLoadCsv(self.path)

            # Show selection dialog
            dialog = PolynomDialog(self.main_parent, self)
            dialog.setPolynoms(polynoms)

            # Find and show the best‐match polynomial
            target_date_str = self.data_header["DATE-OBS"]
            print(target_date_str)
            theBestPoly = CsvPolynom.findNearestDate(polynoms, target_date_str)
            dialog.setTheBestDate(theBestPoly)
            dialog.show()

            # Update right‐panel status
            self.__right.polyProgress.setValue(1)
            self.__right.setPolyTextMode(1, f"{target_date_str}")

        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            return

    def runSetting(self):
        """
        Open the view-settings dialog for line style and colormap.
        """
        dialog = SettingDialog(self.main_parent, self)
        dialog.show()

    def setAppliedPolynom(self, polynom):
        """
        Apply the user-selected polynomial to the spectrum data,
        converting optical density to intensity.
        """
        print("______________________________________")
        print(polynom)
        print(self.arx_data.get_data())

        editor = ArxSpectEditor(self.arx_data)
        new_data = editor.OptDen2Int(polynom)
        self.arx_data.set_data(new_data.get_data())

        print("Polynomial applied.")
        self.__left.imgLoad()

    def save(self):
        """
        Prompt for a save location and write the current FITS data to disk.
        """
        print("going to save fits file")
        msg = QMessageBox()

        try:
            data = self.arx_data.get_data()
            header = self.arx_data.get_header()

            if data is None or header is None:
                msg.setWindowTitle("Error:")
                msg.setText("data or header is None!!!")
                msg.exec_()
                return

            # Suggest filename based on original and date
            base_name = os.path.splitext(os.path.basename(self.fileName))[0]
            today_str = datetime.today().strftime('%Y-%m-%d')
            suggested_name = f"{base_name}_{today_str}.fits"

            save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self.main_parent,
                "Save FITS File As",
                QtCore.QDir.homePath() + "/" + suggested_name,
                "FITS files (*.fits *.fit)"
            )

            if save_path:
                if not save_path.lower().endswith(('.fits', '.fit')):
                    save_path += ".fits"
                hdu = fits.PrimaryHDU(data, header=header)
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(save_path, overwrite=True)
                print(f"File saved as: {save_path}")

                msg.setWindowTitle("Success")
                msg.setText("File has been saved!!!")
                msg.exec_()
            else:
                msg.setWindowTitle("Cancelled")
                msg.setText("Operation was cancelled by user.")
                msg.exec_()

        except Exception as e:
            msg.setWindowTitle("Error...")
            msg.setText("Error saving FITS file!")
            msg.exec_()




#CLass Left of model cropper:
class Left(QLabel):
    """
    A QLabel subclass that displays a FITS image, handles resizing,
    and paints alignment lines on top of the image.
    """

    def __init__(self, parent=None):
        """
        Constructor:

        - Initializes the QLabel with expanding size policy and centered text.
        - Sets a placeholder prompt for the user.
        - Stores references to the model and prepares image buffers.
        """
        super().__init__(None)
        # Make the label expand to fill available space
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # Center-align the placeholder text
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText('BROWSE & LOAD FITS FILE TO SEE THE IMAGE')
        self.setObjectName("cropper_left_mainlabel")

        # Reference back to the parent model
        self.model = parent

        # Internal tracking of image dimensions
        self.__img_w = 0
        self.__img_h = 0

        # Buffers for the original and temporary (annotated) images
        self.loaded_image = None
        self.tmp_image = None

        # Default style for the painted lines
        self.lineThickness = 6
        self.lineColor = (255, 79, 0)


    # ─────────────────────────────────────────────────────────────────────
    # GETTERS
    # ─────────────────────────────────────────────────────────────────────

    def getHeight(self):
        """
        Return:
            The current image height in pixels.
        """
        return self.__img_h

    def getWidth(self):
        """
        Return:
            The current image width in pixels.
        """
        return self.__img_w


    # ─────────────────────────────────────────────────────────────────────
    # SETTERS
    # ─────────────────────────────────────────────────────────────────────

    def setLineStyle(self, thickness, color):
        """
        Update the thickness and color used when painting lines.

        Parameters:
            thickness (int): Line thickness in pixels.
            color (tuple): BGR color tuple, e.g. (blue, green, red).
        """
        self.lineThickness = thickness
        self.lineColor = color


    # ─────────────────────────────────────────────────────────────────────
    # IMAGE BUFFER MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────

    def set_reset(self):
        """
        Copy the loaded image into the temporary buffer and trigger a redraw.

        Returns:
            The stored widget width (unused by caller).
        """
        # Make a fresh copy for drawing overlays
        self.tmp_image = self.loaded_image.copy()
        self.update()
        return self.width()


    # ─────────────────────────────────────────────────────────────────────
    # IMAGE LOADING & DISPLAY
    # ─────────────────────────────────────────────────────────────────────

    def imgLoad(self):
        """
        Load the image from the model's ArxData instance using the current colormap,
        then reset and display it.
        """
        try:
            # Retrieve an OpenCV-style BGR image
            self.loaded_image = self.model.arx_data.get_image(self.model.colorMap)
            # Prepare the temp buffer and display
            self.set_reset()
            self.setImage(self.loaded_image)
        except Exception as err:
            print(f"Unexpected error in imgLoad: {err}, {type(err)}")


    def setImage(self, img):
        """
        Resize the image to fit the label, convert to QImage, and display as a pixmap.

        Parameters:
            img (ndarray): BGR image array from OpenCV.
        """
        # Resize while preserving aspect ratio
        img_resized = imutils.resize(img, width=self.width(), height=self.height())
        # Convert BGR → RGB for Qt
        frame = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        qimg = QImage(frame.data, w, h, frame.strides[0], QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))


    def update(self):
        """
        If a temporary image buffer exists, re‐display it (used after drawing lines).
        """
        if self.tmp_image is not None:
            self.setImage(self.tmp_image)


    def resizeEvent(self, e: QResizeEvent) -> None:
        """
        Re‐render the current image buffer when the widget is resized.
        """
        super().resizeEvent(e)
        if self.tmp_image is not None:
            self.setImage(self.tmp_image)


    # ─────────────────────────────────────────────────────────────────────
    # OVERLAY PAINTING
    # ─────────────────────────────────────────────────────────────────────

    def PaintLines(self, up_pct: int, down_pct: int):
        """
        Draw two horizontal lines on the image to indicate 'up' and 'down' limits.

        Parameters:
            up_pct (int): Percentage from top to draw the upper line.
            down_pct (int): Percentage from bottom to draw the lower line.
        """
        # Ensure we have a loaded image to annotate
        if self.loaded_image is None:
            return

        # Current image dimensions
        h, w = self.loaded_image.shape[:2]
        self.__img_h, self.__img_w = h, w

        # Convert percentages to pixel coordinates
        y_up = int(h * up_pct / 100)
        y_down = h - int(h * down_pct / 100)

        # Choose the line color: default blue unless valid ratio
        color = self.lineColor if y_up < y_down else (0, 0, 255)
        thickness = self.lineThickness

        # Copy the base image for overlay
        self.tmp_image = self.loaded_image.copy()
        # Draw the lines
        self.tmp_image = cv2.line(self.tmp_image, (0, y_up), (w, y_up), color, thickness)
        self.tmp_image = cv2.line(self.tmp_image, (0, y_down), (w, y_down), color, thickness)

        # Update the display
        self.update()

# End of Left class


#End of Left ########################################################


#CLass Right of model cropper:
class Right(QVBoxLayout):
    """
    The right‐hand control panel for spectrum alignment:
    includes file browsing, limit sliders, order selector,
    align/back buttons, polynomial loader, save, and view settings.
    """

    def __init__(self, parent=None):
        super().__init__()  
        # Align everything at the top, no margins
        self.setAlignment(Qt.AlignTop)
        self.setContentsMargins(0, 0, 0, 0)

        # Reference back to the model
        self.model = parent

        # Cache window size for layout
        self.w = self.model.win_w
        self.h = self.model.win_h

        # Build UI and wire up signals
        self.InitUi()
        self.InitConnects()


    def InitUi(self):
        """
        Create and arrange all widgets in the right panel.
        """

        # ─── Browsing Section ───────────────────────────────
        browseQLabel = QLabel()
        browseQLabel.setObjectName("cropper_right_browse_label")
        browseQLabel.setAlignment(Qt.AlignCenter)
        browseQLabel.setWordWrap(True)
        browseQLabel.setFixedSize(int(self.w/5), int(self.h/13))

        browseText = """
            <p style="text-align: justify;"><b>
            Load the spectrum to start aligning</b></p>
        """
        browseTxtQlabel = QLabel(browseText, browseQLabel)
        browseTxtQlabel.setObjectName("cropper_browse_text")
        browseTxtQlabel.setWordWrap(True)
        browseTxtQlabel.setAlignment(Qt.AlignCenter)
        browseTxtQlabel.setFixedSize(int(self.w/6), int(self.h/44))

        self.browseQbtn = QPushButton("Browse")
        self.browseQbtn.setObjectName("cropper_browse_btn")
        self.browseQbtn.setFixedSize(int(self.w/12), int(self.h/30))

        browseLayout = QVBoxLayout()
        browseLayout.setSpacing(7)
        browseLayout.setContentsMargins(0, 0, 0, 5)
        browseLayout.setAlignment(Qt.AlignTop)
        browseLayout.addWidget(browseTxtQlabel, alignment=Qt.AlignHCenter)
        browseLayout.addWidget(self.browseQbtn,    alignment=Qt.AlignHCenter)
        browseQLabel.setLayout(browseLayout)

        self.addWidget(browseQLabel, alignment=Qt.AlignCenter)
        # ────────────────────────────────────────────────────


        # ─── Limits Section ──────────────────────────────────
        limitsQLabel = QLabel()
        limitsQLabel.setObjectName("cropper_right_browse_label")
        limitsQLabel.setAlignment(Qt.AlignCenter)
        limitsQLabel.setWordWrap(True)
        limitsQLabel.setFixedSize(int(self.w/5), int(self.h/4))

        limitsText = """
            <p style="text-align: justify;"><b>
            Up and Down limits</b></p>
        """
        limitsTxtQlabel = QLabel(limitsText, limitsQLabel)
        limitsTxtQlabel.setObjectName("cropper_browse_text")
        limitsTxtQlabel.setWordWrap(True)
        limitsTxtQlabel.setAlignment(Qt.AlignCenter)
        limitsTxtQlabel.setFixedSize(int(self.w/6), int(self.h/44))

        # Two labeled sliders
        self.upQSlider   = QLabeledSlider("<b>Up Line position:</b>")
        self.downQSlider = QLabeledSlider("<b>Down Line position:</b>")
        self.upQSlider.set_slider_range(0, 100)
        self.downQSlider.set_slider_range(0, 100)

        limitsLayout = QVBoxLayout()
        limitsLayout.setSpacing(7)
        limitsLayout.setContentsMargins(5, 0, 0, 5)
        limitsLayout.setAlignment(Qt.AlignTop)
        limitsLayout.addWidget(limitsTxtQlabel, alignment=Qt.AlignHCenter)
        limitsLayout.addWidget(self.upQSlider)
        limitsLayout.addWidget(self.downQSlider)

        # Order selector
        orderLayout = QHBoxLayout()
        orderLayout.setContentsMargins(0, 0, 0, 5)
        orderLayout.setSpacing(15)
        orderLabel = QLabel("Alignment Polynom Order:")
        orderLabel.setObjectName("cropper_right_browse_label")
        orderLabel.setAlignment(Qt.AlignCenter)
        self.orderSpinBox = QSpinBox()
        self.orderSpinBox.setMinimum(2)
        self.orderSpinBox.setMaximum(21)
        self.orderSpinBox.setValue(2)
        self.orderSpinBox.setSingleStep(1)
        orderLayout.addWidget(orderLabel)
        orderLayout.addWidget(self.orderSpinBox, alignment=Qt.AlignLeft)

        container = QWidget()
        container.setLayout(orderLayout)
        limitsLayout.addWidget(container, alignment=Qt.AlignLeft)

        limitsQLabel.setLayout(limitsLayout)
        self.addWidget(limitsQLabel, alignment=Qt.AlignCenter)
        # ────────────────────────────────────────────────────


        # ─── Align/Back Buttons ─────────────────────────────
        alignQLabel = QLabel()
        alignQLabel.setObjectName("cropper_right_browse_label")
        alignQLabel.setAlignment(Qt.AlignCenter)
        alignQLabel.setWordWrap(True)
        alignQLabel.setFixedSize(int(self.w/5), int(self.h/20))

        buttonLayout = QHBoxLayout()
        buttonLayout.setSpacing(15)
        buttonLayout.setContentsMargins(0, 5, 0, 5)
        buttonLayout.setAlignment(Qt.AlignTop)

        self.backQbtn  = QPushButton("Back")
        self.backQbtn.setObjectName("cropper_browse_btn")
        self.backQbtn.setFixedSize(int(self.w/12), int(self.h/30))

        self.alignQbtn = QPushButton("Align")
        self.alignQbtn.setObjectName("cropper_browse_btn")
        self.alignQbtn.setFixedSize(int(self.w/12), int(self.h/30))

        buttonLayout.addWidget(self.backQbtn,  alignment=Qt.AlignHCenter)
        buttonLayout.addWidget(self.alignQbtn, alignment=Qt.AlignHCenter)
        alignQLabel.setLayout(buttonLayout)

        self.addWidget(alignQLabel, alignment=Qt.AlignCenter)
        # ────────────────────────────────────────────────────


        # ─── Polynomial Loader ──────────────────────────────
        polyBrowseQLabel = QLabel()
        polyBrowseQLabel.setObjectName("cropper_right_browse_label")
        polyBrowseQLabel.setAlignment(Qt.AlignCenter)
        polyBrowseQLabel.setWordWrap(True)
        polyBrowseQLabel.setFixedSize(int(self.w/5), int(self.h/8))

        polyBrowseText = """
            <p style="text-align: justify;"><b>
            Load the spectrum to start aligning</b></p>
        """
        polyBrowseTxtQlabel = QLabel(polyBrowseText, polyBrowseQLabel)
        polyBrowseTxtQlabel.setObjectName("cropper_browse_text")
        polyBrowseTxtQlabel.setWordWrap(True)
        polyBrowseTxtQlabel.setAlignment(Qt.AlignCenter)
        polyBrowseTxtQlabel.setFixedSize(int(self.w/6), int(self.h/44))

        self.polyBrowseQbtn = QPushButton("Upload polynom")
        self.polyBrowseQbtn.setObjectName("cropper_browse_btn")
        self.polyBrowseQbtn.setFixedSize(int(self.w/12), int(self.h/30))

        self.polyProgress = QProgressBar()
        self.polyProgress.setAlignment(Qt.AlignCenter)
        self.polyProgress.setTextVisible(False)
        self.polyProgress.setMinimum(0)
        self.polyProgress.setMaximum(1)
        self.polyProgress.setValue(0)
        self.polyProgress.setFixedSize(int(self.w/5.5), int(self.h/250))

        self.polyLoadText = QLabel()
        self.setPolyTextMode(0)

        polyLayout = QVBoxLayout()
        polyLayout.setSpacing(7)
        polyLayout.setContentsMargins(0, 0, 0, 5)
        polyLayout.setAlignment(Qt.AlignTop)
        polyLayout.addWidget(polyBrowseTxtQlabel, alignment=Qt.AlignHCenter)
        polyLayout.addWidget(self.polyBrowseQbtn,   alignment=Qt.AlignHCenter)
        polyLayout.addWidget(self.polyProgress,      alignment=Qt.AlignHCenter)
        polyLayout.addWidget(self.polyLoadText,      alignment=Qt.AlignHCenter)
        polyBrowseQLabel.setLayout(polyLayout)

        self.addWidget(polyBrowseQLabel, alignment=Qt.AlignCenter)
        # ────────────────────────────────────────────────────


        # ─── Save/Cancel Buttons ────────────────────────────
        saveQLabel = QLabel()
        saveQLabel.setObjectName("cropper_right_browse_label")
        saveQLabel.setAlignment(Qt.AlignCenter)
        saveQLabel.setWordWrap(True)
        saveQLabel.setFixedSize(int(self.w/5), int(self.h/20))

        saveBtnLayout = QHBoxLayout()
        saveBtnLayout.setSpacing(7)
        saveBtnLayout.setContentsMargins(0, 5, 0, 5)
        saveBtnLayout.setAlignment(Qt.AlignTop)

        self.cancelQbtn = QPushButton("Cancel")
        self.cancelQbtn.setObjectName("cropper_browse_btn")
        self.cancelQbtn.setFixedSize(int(self.w/12), int(self.h/30))

        self.saveQbtn   = QPushButton("Save")
        self.saveQbtn.setObjectName("cropper_browse_btn")
        self.saveQbtn.setFixedSize(int(self.w/12), int(self.h/30))

        saveBtnLayout.addWidget(self.cancelQbtn, alignment=Qt.AlignHCenter)
        saveBtnLayout.addWidget(self.saveQbtn,   alignment=Qt.AlignHCenter)
        saveQLabel.setLayout(saveBtnLayout)

        self.addWidget(saveQLabel, alignment=Qt.AlignCenter)
        # ────────────────────────────────────────────────────


        # ─── View Settings Button ───────────────────────────
        SettingQLabel = QLabel()
        SettingQLabel.setObjectName("cropper_right_browse_label")
        SettingQLabel.setAlignment(Qt.AlignCenter)
        SettingQLabel.setWordWrap(True)
        SettingQLabel.setFixedSize(int(self.w/5), int(self.h/20))

        SettingText = """
            <p style="text-align: justify;"><b>Tool View Settings:</b></p>
        """
        SettingTextQLbl = QLabel(SettingText, SettingQLabel)
        SettingTextQLbl.setObjectName("cropper_browse_text")
        SettingTextQLbl.setWordWrap(True)
        SettingTextQLbl.setAlignment(Qt.AlignCenter)
        SettingTextQLbl.setFixedHeight(int(self.h/44))

        self.SettingQbtn = QPushButton("")
        self.SettingQbtn.setObjectName("trans")
        self.SettingQbtn.setFixedHeight(int(self.h/30))
        self.SettingQbtn.setFixedWidth(int(self.w/24))
        icon_path = os.path.join(os.path.dirname(__file__), "color_setting.png")
        self.SettingQbtn.setIcon(QIcon(icon_path))
        self.SettingQbtn.setIconSize(QSize(int(self.w/12), int(self.h/30)))
        self.SettingQbtn.clicked.connect(self.model.runSetting)

        EmptyQbtn = QPushButton("")
        EmptyQbtn.setObjectName("trans")
        EmptyQbtn.setEnabled(False)
        EmptyQbtn.setFixedHeight(int(self.h/30))
        EmptyQbtn.setFixedWidth(int(self.w/12))

        viewLayout = QHBoxLayout()
        viewLayout.setSpacing(15)
        viewLayout.setContentsMargins(0,5,0,5)
        viewLayout.setAlignment(Qt.AlignCenter)
        viewLayout.addWidget(SettingTextQLbl, stretch=10)
        viewLayout.addWidget(self.SettingQbtn,    stretch=10)
        viewLayout.addWidget(EmptyQbtn,           stretch=5)
        SettingQLabel.setLayout(viewLayout)

        self.addWidget(SettingQLabel, alignment=Qt.AlignCenter)
        # ────────────────────────────────────────────────────


    def InitConnects(self):
        """
        Wire up all button and slider signals to model methods.
        """
        self.browseQbtn.clicked.connect(      self.model.ImgLoad)
        self.upQSlider.valueChanged.connect(  lambda val: self.model.setUpLineValue(val))
        self.downQSlider.valueChanged.connect(lambda val: self.model.setDownLineValue(val))
        self.alignQbtn.clicked.connect(       self.model.alignData)
        self.backQbtn.clicked.connect(        self.model.backToSpectrum)
        self.polyBrowseQbtn.clicked.connect(  self.model.polynomLoad)
        self.saveQbtn.clicked.connect(        self.model.save)


    def reset_sliders(self):
        """
        Reset all sliders back to zero.
        """
        self.upQSlider.reset()
        self.downQSlider.reset()
#................................

    def setPolyTextMode(self, mode, txt=None):
        """
        Update the polynomial‐status label on the right panel.

        Parameters:
        - mode=0: no polynomial loaded
        - mode=1: polynomial loaded (txt must be provided)
        - else : error status
        """
        if mode == 0:
            # Style for “not loaded”
            self.polyLoadText.setObjectName("polyLblNotloaded")
            self.polyLoadText.style().unpolish(self.polyLoadText)
            self.polyLoadText.style().polish(self.polyLoadText)
            self.polyLoadText.setText("No Polinomial!")
        elif mode == 1 and txt is not None:
            # Style for “loaded” and display the date/text
            self.polyLoadText.setObjectName("polyLblLoaded")
            self.polyLoadText.style().unpolish(self.polyLoadText)
            self.polyLoadText.style().polish(self.polyLoadText)
            self.polyLoadText.setText("Polynomial on " + str(txt))
        else:
            # Fallback in case of invalid mode or missing txt
            self.polyLoadText.setObjectName("polyLblNotloaded")
            self.polyLoadText.style().unpolish(self.polyLoadText)
            self.polyLoadText.style().polish(self.polyLoadText)
            self.polyLoadText.setText("Error status")


#End of Right ########################################################

# THICKNESS & COLOR DIALOG WINDOW:
class SettingDialog(QMainWindow):
    """
    Dialog window for selecting line thickness, line color, and colormap
    for spectral line display.
    """

    def __init__(self, mainparent=None, model=None):
        """
        Initialize the dialog.

        Parameters:
        - mainparent: the parent QMainWindow
        - model:       the model_align instance to apply settings to
        """
        super().__init__(mainparent)

        # Store reference to the alignment model
        self.Model = model

        # Configure window title and icon
        self.setWindowTitle("View settings")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "color_setting.png")
        self.setWindowIcon(QIcon(icon_path))

        # Determine screen size and set dialog to 20% width, 30% height
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        self.window_width = int(screen_width * 0.2)
        self.window_height = int(screen_height * 0.3)
        self.resize(self.window_width, self.window_height)

        # Central widget and layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(5, 5, 5, 5)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # --- Spectral lines colormap selection ---
        mode_gbox = QGroupBox("Spectral Lines Display Mode")
        mode_gbox.setObjectName("settingframe")
        mode_gbox_layout = QVBoxLayout()
        mode_gbox_layout.setContentsMargins(5, 10, 10, 10)
        mode_gbox.setLayout(mode_gbox_layout)
        central_layout.addWidget(mode_gbox)

        # Spacer to separate label from controls
        mode_gbox_layout.addItem(
            QSpacerItem(1, 20, QSizePolicy.Minimum, QSizePolicy.Fixed)
        )

        # ComboBox of available OpenCV colormaps
        self.color_combo = QComboBox()
        self.colormaps_list = [
            ("AUTUMN",  cv2.COLORMAP_AUTUMN),
            ("BONE",    cv2.COLORMAP_BONE),
            ("JET",     cv2.COLORMAP_JET),
            ("WINTER",  cv2.COLORMAP_WINTER),
            ("RAINBOW", cv2.COLORMAP_RAINBOW),
            ("OCEAN",   cv2.COLORMAP_OCEAN),
            ("SUMMER",  cv2.COLORMAP_SUMMER),
            ("SPRING",  cv2.COLORMAP_SPRING),
            ("COOL",    cv2.COLORMAP_COOL),
            ("HSV",     cv2.COLORMAP_HSV),
            ("PINK",    cv2.COLORMAP_PINK),
            ("HOT",     cv2.COLORMAP_HOT),
            ("PARULA",  cv2.COLORMAP_PARULA),
            ("MAGMA",   cv2.COLORMAP_MAGMA),
            ("INFERNO", cv2.COLORMAP_INFERNO),
            ("PLASMA",  cv2.COLORMAP_PLASMA),
            ("VIRIDIS", cv2.COLORMAP_VIRIDIS),
            ("CIVIDIS", cv2.COLORMAP_CIVIDIS),
        ]
        for name, cmap_id in self.colormaps_list:
            self.color_combo.addItem(name, userData=cmap_id)
        mode_gbox_layout.addWidget(self.color_combo, alignment=Qt.AlignTop)

        # --- Line thickness & color controls ---
        align_gbox = QGroupBox("Alignment Line Style")
        align_gbox.setObjectName("settingframe")
        align_gbox_layout = QHBoxLayout()
        align_gbox_layout.setContentsMargins(5, 40, 40, 5)
        align_gbox_layout.setSpacing(10)
        align_gbox.setLayout(align_gbox_layout)
        central_layout.addWidget(align_gbox)

        # Thickness label + spin box
        align_thick_label = QLabel("Thickness:")
        align_thick_label.setObjectName("settingText")
        align_thick_label.setAlignment(Qt.AlignLeft)
        align_gbox_layout.addWidget(align_thick_label)

        self.align_thick_spinbox = QSpinBox()
        self.align_thick_spinbox.setMinimum(1)
        self.align_thick_spinbox.setMaximum(99)
        self.align_thick_spinbox.setValue(8)
        self.align_thick_spinbox.setSingleStep(1)
        align_gbox_layout.addWidget(self.align_thick_spinbox, alignment=Qt.AlignTop)

        # Color label + color‐picker button
        align_color_label = QLabel("Color:")
        align_color_label.setObjectName("settingText")
        align_color_label.setAlignment(Qt.AlignLeft)
        align_gbox_layout.addWidget(align_color_label)

        self.align_color_button = QPushButton()
        self.align_color_button.setObjectName("trans")
        self.align_color_button.setFixedSize(
            int(self.window_width/4), int(self.window_height/10)
        )
        icon_path = os.path.join(script_dir, "color_btn_4.png")
        self.align_color_button.setIcon(QIcon(icon_path))
        self.align_color_button.setIconSize(
            QSize(int(self.window_width/4), int(self.window_height/10))
        )
        align_gbox_layout.addWidget(self.align_color_button, alignment=Qt.AlignTop)

        # --- Apply & Close button ---
        self.ok_button = QPushButton("Apply")
        self.ok_button.setObjectName("cropper_browse_btn")
        self.ok_button.setFixedSize(int(self.window_width/4), int(self.window_height/10))
        central_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        # Wire up signals
        self.align_color_button.clicked.connect(
            lambda: self.choose_color(self.align_color_button)
        )
        # Default button color
        self.align_color_button.selected_color = QColor("#0000FF")
        self.ok_button.clicked.connect(self.on_ok_clicked)

        # Ensure minimum size
        self.adjustSize()
        self.setMinimumSize(self.size())

    def choose_color(self, button):
        """
        Open a QColorDialog to pick a new color.
        Updates button.selected_color and its stylesheet.
        """
        color = QColorDialog.getColor()
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()};")
            button.selected_color = color

    def on_ok_clicked(self):
        """
        Read selected thickness, color, and colormap,
        apply them to the model_align instance, and
        redraw lines if an image is already loaded.
        Finally, close the dialog.
        """
        # 1) Get chosen colormap ID
        idx = self.color_combo.currentIndex()
        cmap_id = self.color_combo.itemData(idx)

        # 2) Convert selected QColor to OpenCV BGR tuple
        qc = self.align_color_button.selected_color
        rgb = (qc.red(), qc.green(), qc.blue())
        bgr = (rgb[2], rgb[1], rgb[0])

        # 3) Apply to model_align
        self.Model.setLineThickness(self.align_thick_spinbox.value())
        self.Model.setLineColor(bgr)
        self.Model.setColorMap(cmap_id)

        # 4) If an image is displayed, repaint the alignment lines
        if getattr(self.Model, "isImgLoaded", False):
            self.Model.getLeftWidget().PaintLines(
                self.Model.upValue,
                self.Model.downValue
            )

        # 5) Close the dialog
        self.close()

# SELECT POLYNOMIAL DIALOG WINDOW:
class PolynomDialog(QMainWindow):
    """
    Dialog for selecting a polynomial entry from a CSV list.
    Displays a matplotlib preview (if any), the best‐match date,
    and allows the user to choose an alternative entry.
    """

    def __init__(self, mainparent=None, model=None):
        """
        Initialize the Polynomial selection dialog.

        Parameters:
        - mainparent: parent QWidget (typically the main window)
        - model:      model_align instance to which the chosen polynomial will be applied
        """
        super().__init__(mainparent)
        self.Model = model

        # Window title and icon
        self.setWindowTitle("Polynomial Selection")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "color_setting.png")
        self.setWindowIcon(QIcon(icon_path))

        # Resize to 40% width, 90% height of the screen
        screen = QApplication.primaryScreen().availableGeometry()
        self.w = int(screen.width() * 0.4)
        self.h = int(screen.height() * 0.9)
        self.resize(self.w, self.h)

        # Central widget & layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(10, 10, 10, 10)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

        # Matplotlib figure & canvas (for any preview plots)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        central_layout.addWidget(self.canvas)

        # Label to show the “best” polynomial date
        self.bestDateQlbl = QLabel()
        self.bestDateQlbl.setObjectName("cropper_right_browse_label")
        self.bestDateQlbl.setAlignment(Qt.AlignCenter)
        self.bestDateQlbl.setWordWrap(True)
        central_layout.addWidget(self.bestDateQlbl)

        # Table widget for listing all available polynomials
        self.polyTable = QTableWidget()
        self.polyTable.setColumnCount(3)
        self.polyTable.setHorizontalHeaderLabels(["ID", "Date", "Note"])
        self.polyTable.setEditTriggers(QTableWidget.NoEditTriggers)
        self.polyTable.setSelectionBehavior(QTableWidget.SelectRows)
        self.polyTable.setSelectionMode(QTableWidget.SingleSelection)
        self.polyTable.verticalHeader().setVisible(False)
        self.polyTable.horizontalHeader().setStretchLastSection(True)

        # Stack: first show bestDateQlbl, then polyTable on demand
        self.stacked_layout = QStackedLayout()
        self.stacked_layout.addWidget(self.bestDateQlbl)
        self.stacked_layout.addWidget(self.polyTable)
        stack_container = QWidget()
        stack_container.setLayout(self.stacked_layout)
        central_layout.addWidget(stack_container)

        # Buttons: “Choose another date” and “Accept this date”
        btn_layout = QHBoxLayout()
        self.anotherQbtn = QPushButton("Choose another date")
        self.anotherQbtn.setObjectName("cropper_browse_btn")
        self.anotherQbtn.setFixedSize(self.w // 14, self.h // 38)
        self.acceptQbtn = QPushButton("Accept this date")
        self.acceptQbtn.setObjectName("cropper_browse_btn")
        self.acceptQbtn.setFixedSize(self.w // 14, self.h // 38)
        btn_layout.addWidget(self.anotherQbtn, alignment=Qt.AlignHCenter)
        btn_layout.addWidget(self.acceptQbtn, alignment=Qt.AlignHCenter)

        btn_container = QLabel()  # using a label widget as a styled container
        btn_container.setObjectName("cropper_right_browse_label")
        btn_container.setFixedSize(self.w / 5.3, self.h / 24)
        btn_container.setLayout(btn_layout)
        central_layout.addWidget(btn_container, alignment=Qt.AlignCenter)

        # Placeholder for the currently selected polynomial object
        self.POLYnom = None

        # Signal connections
        self.anotherQbtn.clicked.connect(self.showAnotherDate)
        self.acceptQbtn.clicked.connect(lambda: self.Model.setAppliedPolynom(self.POLYnom))

        # Final adjustments
        self.adjustSize()
        self.setMinimumSize(self.size())

    def setPolynoms(self, polynoms):
        """
        Populate the table with a list of polynomial entries.

        Each entry should have attributes: Id, date, note.
        """
        self.polyTable.setRowCount(len(polynoms))
        for row, poly in enumerate(polynoms):
            id_item = QTableWidgetItem(str(poly.Id) if poly.Id is not None else "")
            id_item.setTextAlignment(Qt.AlignCenter)
            date_item = QTableWidgetItem(str(poly.date))
            date_item.setTextAlignment(Qt.AlignCenter)
            note_item = QTableWidgetItem(poly.note or "")
            note_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            self.polyTable.setItem(row, 0, id_item)
            self.polyTable.setItem(row, 1, date_item)
            self.polyTable.setItem(row, 2, note_item)

    def showAnotherDate(self):
        """
        Switch from the best‐date label view to the full table view,
        so the user can pick a different row.
        """
        self.stacked_layout.setCurrentIndex(1)

    def setTheBestDate(self, theBestPoly):
        """
        Display the best‐matched polynomial date and store the object
        for acceptance.
        """
        self.bestDateQlbl.setText(f"Selected Date:\n{theBestPoly.date}")
        self.POLYnom = theBestPoly
