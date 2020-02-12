from PyQt5 import QtWidgets, QtGui, QtCore, uic
import time, cfg
import numpy as np

class CalibrateWidget(QtWidgets.QWidget):
	def __init__(self, parent, res):
		super(CalibrateWidget, self).__init__() # Call the inherited classes __init__ method

		self.btn_ind = 0
		self.img_counter = 50

		self.parent = parent
		self.res = res

		self.counter = 10
		self.state = 1

		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.repaint)
		self.timer.start(1000)

		self.initUI()


	def initUI(self):
		self.setGeometry(0, 0, self.res.width, self.res.height)
		self.setWindowTitle("Calibration")
		self.setStyleSheet("background-color: darkgray;")

	def paintEvent(self, e):
		qp = QtGui.QPainter()
		qp.begin(self)

		if self.state:
			qp.setFont(QtGui.QFont('Decorative', 32))
			qp.drawText(int(self.res.width * 0.5),
				int(self.res.height) * 0.5, 'Калибровка начнется через %sс' % self.counter)

			self.counter -= 1
			if self.counter == 0:
				self.state = 0

				self.timer.setInterval(100)

		else:
			with self.parent.lock:
				if self.btn_ind < 8:
					if self.img_counter:
						label = np.zeros(8)
						label[self.btn_ind] = 1.

						if (self.parent.data and
							   np.all(self.parent.data[-1] == self.parent.eyes_roi)):
								pass
						else:
							if self.img_counter <= 10:
								self.parent.data.append(self.parent.eyes_roi)
								self.parent.labels.append(label)
						
							self.img_counter -= 1

					else:
						self.btn_ind += 1
						self.img_counter = 50
				else:
					self.parent.record_dataset()
					self.close()

			if self.btn_ind < 8:
				qp.setBrush(QtGui.QBrush(QtCore.Qt.darkGreen, QtCore.Qt.SolidPattern))

				coords = [self.res.width, self.res.height] * cfg.ball_positions[self.btn_ind]
				coords = tuple(int(coord) for coord in coords)

				qp.drawEllipse(*coords, 50, 50)

				


		qp.end()








