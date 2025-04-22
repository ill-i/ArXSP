###############################################
from multiprocessing import Lock
from PyQt5.QtGui import*####################### 
from PyQt5.QtCore import*###################### 
from PyQt5.QtWidgets import*###################
from PyQt5.QtGui import QScreen################
from PyQt5.QtWidgets import QFrame#############
from PyQt5 import QtGui, QtCore, QtWidgets#####
from PyQt5.QtGui import QTextCursor, QColor
###############################################

#system files:
import sys#############
import os #############
import importlib.util #
#######################

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPlainTextEdit, QPushButton, QFileDialog,QDialog
)
from PyQt5.QtGui import QScreen
import sys
import astropy.io.fits as fits
from astropy.io.fits import Header


class FileHeaderEditor(QMainWindow):

    headerSaved = pyqtSignal(Header)

    def __init__(self, arx_data, parent=None, ):
        super().__init__(parent)
        #Your data (It is ArxData object) to work
        self.arx_data = arx_data
        self.lines = None
        self.initializer()
#End of constructor_______


#Initializer of main window:
    def initializer(self):

    #Setting Title & Icon of Main Window:
        self.setWindowTitle("AP22784884")######################
        self.setWindowIcon(QIcon("resources/main_icon.png"))

    #Obtain Screen Sizes:
        screen = QApplication.primaryScreen()#########
        screen_geometry = screen.availableGeometry()##
        screen_width = screen_geometry.width()########
        screen_height = screen_geometry.height()######
        ##############################################
        #Setting the main window size as a% of the screen:
        # 80% of screen width_____________________________
        self.window_width = int((screen_width) * 0.4)#####
        # 80% of screen height____________________________
        self.window_height = int((screen_height) * 0.9)### 
        self.resize(self.window_width,self.window_height)#

# Создаём центральный виджет и layout для него
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 2) Текстовый редактор
        self.text_edit = CustomPlainTextEdit([1,2])
        self.text_edit.setPlaceholderText("Введите текст...")
        main_layout.addWidget(self.text_edit)

        # 3) Нижняя панель с кнопкой «Сохранить» (справа)
        bottom_buttons_layout = QHBoxLayout()
        main_layout.addLayout(bottom_buttons_layout)

        self.save_button = QPushButton("Save header")
        self.save_button.setObjectName("cropper_browse_btn")
        self.save_button.clicked.connect(self.on_save_clicked) #save_file
        self.save_button.setMinimumHeight(100) ######################################
        self.save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bottom_buttons_layout.addWidget(self.save_button, alignment=Qt.AlignRight)
        

        keywords = [
            'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'EXTEND', 'DATE-OBS',
            'EXPTIME', 'TELESCOP', 'OBSERVER', 'OBSERVAT', 'DATES', 'LST'
            ]

        try:
            #hdu = fits.open(filepath) 
            content = self.arx_data.get_header().tostring(sep="\n")
            self.lines = content.splitlines()
            for line in self.lines:
                self.text_edit.appendPlainText(line)
                #self.text_edit.setPlainText(line)

            locked_lines = [ i for i, line in enumerate(self.lines)
                                if any(line.lstrip().startswith(key) for key in keywords)]

            self.text_edit.setLockedLines(locked_lines)

        except Exception as e:
            pass
            #print(f"Ошибка при чтении файла: {e}")




#_______________________________________________________________________________________________

    def save_header(self):
        self.lines.append("END")

        header_from_text = '\n'.join(self.lines)
        
        return Header.fromstring(header_from_text, sep='\n')


    def on_save_clicked(self):
        new_header = self.save_header()
        self.headerSaved.emit(new_header)
        self.close()


    def save_file(self):
        
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить файл",
            "",
            "Текстовые файлы (*.txt);;Все файлы (*)",
            options=options
        )

        self.lines.append('END')

        # Соберём обратно
        safe_header_text = '\n'.join(lines)
        new_header_txt = Header.fromstring(safe_header_text, sep='\n')
        if filepath:
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(self.text_edit.toPlainText())
            except Exception as e:
                print(f"Ошибка при сохранении файла: {e}")




    def on_ok_clicked(self):
        # Сохраняем данные
        self._result_data = self.line_edit.text()
        # Завершаем диалог со статусом "принято"
        self.accept()


    def get_result(self):
        #"""Вернёт сохранённое значение, введённое в поле."""
        return self._result_data





#_________________________________________________________

class CustomPlainTextEdit(QPlainTextEdit):
    def __init__(self, locked_lines=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Набор или список заблокированных номеров строк
        self.locked_lines = set(locked_lines) if locked_lines else set()


    def setLockedLines(self, lines_to_lock):

        self.locked_lines = set(lines_to_lock)
        self._highlight_locked_lines()  # Обновляем подсветку

    def keyPressEvent(self, event):
        cursor = self.textCursor()
        current_block = cursor.block()  # QTextBlock
        current_line_number = current_block.firstLineNumber()
        
        if current_line_number in self.locked_lines:
            # Игнорируем ввод/редактирование
            return  # или event.ignore()
        else:
            # Разрешаем редактирование
            super().keyPressEvent(event)

    def _highlight_locked_lines(self):
        # Список подсветок
        extra_selections = []
        
        # Настраиваем цвет для заблокированных строк
        # (например, светло-серый цвет шрифта)
        locked_format_color = QColor(Qt.lightGray)
        
        for line_number in self.locked_lines:
            block = self.document().findBlockByLineNumber(line_number)
            if block.isValid():
                # Создаем выделение
                selection = QTextCursor(block)
                # Выделяем всю строку
                selection.select(QTextCursor.LineUnderCursor)
                
                # Оформление для заблокированной строки
                extra_sel = QPlainTextEdit.ExtraSelection()
                extra_sel.cursor = selection
                # Меняем цвет текста (foreground)
                extra_sel.format.setForeground(locked_format_color)
                
                extra_selections.append(extra_sel)
        
        # Устанавливаем все выделения разом
        self.setExtraSelections(extra_selections)








