from PyQt5.QtWidgets import QApplication, QLabel
import sys

app = QApplication(sys.argv)
label = QLabel("It works!")
label.resize(200, 100)
label.show()
sys.exit(app.exec_())