from PyQt5 import QtWidgets, QtGui, QtCore, uic
import time
import numpy as np

class CalibrateWidget(QtWidgets.QWidget):
	def __init__(self, parent, res):
		super(CalibrateWidget, self).__init__() # Call the inherited classes __init__ method

		self.parent = parent
		self.res = res
		self.ball_x = 0.011
		self.ball_y = 0.79

		self.state = 0 # 0 - countdown, +-1-move up/down
		self.counter = 10

		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.repaint)
		self.timer.start(1000)

		self.initUI()


	def initUI(self):
		self.setGeometry(0, 0, self.res.width, self.res.height)
		self.setWindowTitle("Calibration")

	def paintEvent(self, e):
		qp = QtGui.QPainter()
		qp.begin(self)

		if not self.state:
			qp.setFont(QtGui.QFont('Decorative', 32))
			qp.drawText(int(self.res.width * 0.5),
				int(self.res.height) * 0.5, 'Калибровка начнется через %sс' % self.counter)

			self.counter -= 1
			if self.counter == 0:
				self.state = -1

				self.timer.setInterval(100)
				self.parent.record_dataset()

		else:
			with self.parent.lock:

				if abs(self.state) == 1:
					if self.ball_y > 0.8:
						self.state = -1
						self.ball_x += 0.08
					elif self.ball_y < 0.01:
						self.state = 1
						self.ball_x += 0.08

					if self.ball_x > 0.95:
						self.parent.record_dataset()
						self.close()
						return

					self.ball_y += 0.01 * self.state


					self.parent.data.append(self.parent.eyes_roi)
					self.parent.labels.append([self.ball_x, self.ball_y])

			qp.setPen(QtCore.Qt.red)
			qp.setBrush(QtGui.QBrush(QtCore.Qt.red, QtCore.Qt.SolidPattern))
			qp.drawEllipse(int(self.res.width * self.ball_x), 
			int(self.res.height*self.ball_y), 50, 50)	

		qp.end()






