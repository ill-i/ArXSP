
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


class mp_viewer(QWidget):
    def __init__(self, parent=None):
        super(mp_viewer, self).__init__(parent)
        #d
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.panel = QWidget()
        self.panel_layout = QVBoxLayout(self.panel)

        self.dict_files = dict()
        self.dict_id_xposs = dict()

        self.all_Peaks_data = {}
        self.all_Peaks_data_orign = {}
        self.poly_coef = {}
        #d
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        #d
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        #d
        panel_main_layout = QVBoxLayout()
        panel_main_layout.addStretch()
        panel_main_layout.addWidget(self.panel)
        panel_main_layout.addStretch()

        layout.addLayout(canvas_layout)
        layout.addLayout(panel_main_layout)
        self.setLayout(layout)

        self.numer = 1
        self.mag_before_1972 = [0,0.61,1.1,1.47,1.84,2.25,2.66,3.04,0]
        self.mag_after_1972 = [0,0.5,0.97,1.44,1.93,2.43,2.69,3.04,0]

        self.graph_check = None
        self.intensity = [0,1,2]
        self.dark = [0,1,2]
        self.poly_coeffs = None
        self.frames_id_str = ''

        self.saved_poly_coeffs = {}




    def set_all_peaks(self, all_peaks):
        self.all_Peaks_data.clear()
        self.all_Peaks_data_orign.clear()
        self.all_peaks = all_peaks
        for name, peaks in all_peaks.items():
            self.all_Peaks_data[name] = []
            self.all_Peaks_data_orign[name] = []
            for i, (x, y) in enumerate(peaks):
                try:
                    mag = self.mag_after_1972[i] if self.numer == 0 else self.mag_before_1972[i]
                except IndexError:
                    mag = x
                self.all_Peaks_data[name].append((mag, y))
                self.all_Peaks_data_orign[name].append((mag, y))


    def set_dialog_parent(self, dia_parent):
        self.dia_parent = dia_parent

    
    def ImgShow(self):
        if self.graph_check:
            self.graph_check.remove()
        ax = self.figure.add_subplot(111)
        self.graph_check = ax
        ax.clear()
        colors = ['b','g','r','c','m','k']
        for idx, (id, values) in enumerate(self.all_Peaks_data.items()):
            x, y = zip(*values)
            ax.scatter(x, y, c=colors[idx % len(colors)])
            ax.set_xlim(-8, 8)
        self.canvas.draw()

    def show(self):
        lbl = QLabel('Exposition: ')
        self.Expose_box = QSpinBox()
        self.Expose_box.setValue(1)
        self.Expose_box.setRange(0, 1500)
        self.panel_layout.addWidget(lbl)
        self.panel_layout.addWidget(self.Expose_box)

        for file_id, file_name in enumerate(self.all_peaks.keys()):
            label = QLabel(file_name, self.panel)
            self.panel_layout.addWidget(label)
            self.frames_id_str += f" - {file_id};"
            slider = QSlider(Qt.Horizontal, self.panel)
            slider.setRange(-100, 100)
            slider.setValue(0)
            slider.valueChanged.connect(partial(self.slider_value_changed_by_id, file_id))
            self.panel_layout.addWidget(slider)

        btn_convert = QPushButton("Convert")
        btn_convert.clicked.connect(self.convert_button_clicked)
        self.panel_layout.addWidget(btn_convert)

        last_layout = QHBoxLayout()
        self.input_box = QSpinBox()
        self.input_box.setValue(1)
        self.input_box.setRange(1, 20)

        btn_polynom = QPushButton("Polynom")
        btn_polynom.clicked.connect(self.handle_polynom_button_click)

        btn_save_polynom = QPushButton("Save")
        btn_save_polynom.clicked.connect(self.save_polynom)

        last_layout.addWidget(self.input_box)
        last_layout.addWidget(btn_polynom)
        last_layout.addWidget(btn_save_polynom)

        self.panel_layout.addLayout(last_layout)
        super(mp_viewer, self).show()
        self.ImgShow()

    def handle_polynom_button_click(self):
        order = self.input_box.value() or 6
        self.do_polynom(order)

    def do_polynom(self, order=6):
        y = self.intensity
        x = self.dark
        try:
            self.poly_coeffs = np.polynomial.polynomial.polyfit(x, y, order)
            poly = np.polynomial.Polynomial(self.poly_coeffs)

            x_range = np.linspace(x[0], x[-1], 100)
            y_poly = poly(x_range)
            is_monotonic = np.all(np.diff(y_poly) >= 0) or np.all(np.diff(y_poly) <= 0)

            if not is_monotonic:
                return self.do_polynom(order + 1)

            if self.graph_check:
                self.graph_check.remove()
            ax = self.figure.add_subplot(111)
            self.graph_check = ax
            ax.clear()
            ax.set_xlim(x[0], x[-1])
            ax.scatter(x, y, color="blue", s=5, label="Data")
            ax.plot(x_range, y_poly, color="orange", label=f"Polynomial (order={order})")
            ax.legend()
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Optical density")
            self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def get_equation(self):
        terms = [f"{coef:.4f} x^{i}" for i, coef in enumerate(self.poly_coeffs)]
        return " + ".join(terms).replace(" x^0", "").replace(" x^1", " x")

    def save_polynom(self):
        text, ok = QInputDialog.getText(None, "Input Dialog", "Enter polynomial name:")
        if ok and text:
            if text in self.saved_poly_coeffs:
                QMessageBox.information(self, "Duplicate", "Name already exists")
                return
            self.saved_poly_coeffs[text] = {
                "frames_id": self.frames_id_str,
                "coeff": self.poly_coeffs,
                "equation": self.get_equation()
            }
            QMessageBox.information(self, "Saved", self.get_equation())

    def convert_button_clicked(self):
        array = [item for sublist in self.all_Peaks_data.values() for item in sublist]
        x_array, y_array = zip(*array)
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        c = np.abs(np.diff(y_array))
        mean_deviation = c.sum() / len(y_array)
        mask = np.insert(np.abs(np.diff(y_array)) < mean_deviation * 2, 0, True)
        x_array = x_array[mask]
        y_array = y_array[mask]
        intensity = 10000 / 10**(x_array / 2.5)
        sorted_data = sorted(zip(intensity, y_array), key=lambda pair: pair[0])
        self.intensity, self.dark = zip(*sorted_data)
        self.intensity = np.array(self.intensity)
        self.dark = np.array(self.dark)

        if self.graph_check:
            self.graph_check.remove()
        ax = self.figure.add_subplot(111)
        self.graph_check = ax
        ax.clear()
        ax.plot(self.intensity, self.dark, marker='o', markersize=3)
        self.canvas.draw()

    #def slider_value_changed_by_id(self, id, value):
    #    sign = -1 if value < 0 else 1
    #    delta = 2.5 * np.log(self.Expose_box.value() / ((self.Expose_box.value() * max(abs(value), 1)) / 100))
    #    for i in range(len(self.all_Peaks_data_orign[id])):
    #        try:
    #            x_val = self.mag_after_1972[i] if self.numer == 0 else self.mag_before_1972[i]
    #        except IndexError:
    #            x_val = self.all_Peaks_data_orign[id][i][0]
    #        self.all_Peaks_data[id][i] = (x_val + delta * sign, self.all_Peaks_data_orign[id][i][1])
    #    self.ImgShow()         for file_id, file_name in enumerate(self.all_peaks.keys()):

    #def slider_value_changed_by_id(self, id, value):
    #    try:
    #        sign = -1 if value < 0 else 1
    #        delta = 2.5 * np.log(self.Expose_box.value() / ((self.Expose_box.value() * max(abs(value), 1)) / 100))
    #        for i in enumerate(self.all_peaks.keys()):
    #            try:
    #                x_val = self.mag_after_1972[i] if self.numer == 0 else self.mag_before_1972[i]
    #            except IndexError as I:
    #                print("IndexErr",I)
    #                x_val = self.all_Peaks_data_orign[id][i][0]
    #            self.all_Peaks_data[id][i] = (x_val + delta * sign, self.all_Peaks_data_orign[id][i][1])
    #        self.ImgShow()
    #    except Exception as e:
    #        print("err",e)

    def slider_value_changed_by_id(self, id, value):

        sample_name = self.dict_files.get(id)
        if not sample_name:
            return

        if id not in self.all_Peaks_data or not self.all_Peaks_data_orign.get(id):
            return

        peaks = self.all_Peaks_data_orign[id]
        value_of_exposition = self.Expose_box.value()

        if value_of_exposition == 0:
            return

        try:
            denominator = (value_of_exposition * max(abs(value), 1)) / 100
            delta = 2.5 * np.log(value_of_exposition / denominator)
        except Exception as e:
            return

        sign = -1 if value < 0 else 1

        updated_peaks = []
        for i in range(len(peaks)):
            y = peaks[i][1]
            try:
                base_x = self.mag_after_1972[i] if self.numer == 0 else self.mag_before_1972[i]
            except IndexError:
                base_x = peaks[i][0]

            new_x = base_x + delta * sign
            updated_peaks.append((new_x, y))

        self.all_Peaks_data[id] = updated_peaks
        self.ImgShow()

    def add_file_Id_fname(self, id, fname):
        self.dict_files[id] = fname
        self.dict_id_xposs[id] = 0

    def set_date(self, numer):
        print("numer = ", numer)
        self.numer = numer


