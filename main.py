from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from functools import partial
from imutils import face_utils
from functools import partial
import dlib, cv2, pyautogui, json
import sys, threading, queue, base64
import random as rand
import numpy as np
import cfg, os
from calibrate import CalibrateWidget

class Ui(QtWidgets.QWidget):
	def __init__(self):
		super(Ui, self).__init__() # Call the inherited classes __init__ method
		uic.loadUi('./templates/demo.ui', self) # Load the .ui file

		self.isRecording = False
		self.testing = False
		self.frame = None
		self.q = queue.Queue()
		self.eyes_roi = None

		self.data = []
		self.labels = []
		self.dataset = None
		self.model = None

		self.update_lists()

		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.update_widgets)
		self.timer.start(0)

		self.recordBtn.clicked.connect(self.create_calibration_widget)
		self.trainModelBtn.clicked.connect(self.init_train_thread)
		self.loadDataBtn.clicked.connect(self.load_dataset)
		self.loadModelBtn.clicked.connect(self.load_model)

		self.lock = threading.Lock()

		self.init_cam_thread()
		self.show() # Show the GUI

	def init_cam_thread(self):
		self.t1 = threading.Thread(target=self.cam_thread)
		self.t1.daemon = True
		self.t1.start()

	def load_dataset(self, event):
		s = self.dataList.currentItem() #Selection

		if not s:
			return

		try:
			ds_name = s.text()

			data = np.load(cfg.DATA_PATH + ds_name, allow_pickle=True)
			labels = np.load(cfg.DATA_PATH + ds_name.replace('data', 'labels'), allow_pickle=True)
			
			self.dataset = ds_name
			print("%s succesfully loaded!" % ds_name)

		except FileNotFoundError:
			print("%s or %s not exists" % 
				(ds_name, ds_name.replace('data', 'replace')))

	def load_model(self, event):
		s = self.modelsList.currentItem() #Selection
		if not s:
			return

		try:
			model_name = s.text()
			self.model = load_model(cfg.MODELS_PATH+model_name)

			print("%s succesfully loaded!" % model_name)

		except OSError:
			print("%s not exists" % model_name)

	def save_dataset(self, data, labels):
		file_name = ''.join([rand.choice('qwertyasd123456') for i in range(5)])

		with open(cfg.DATA_PATH + '%s_data_%s.npy' % (file_name, len(labels)), 'wb') as f:
			np.save(f, np.array(data))

		with open(cfg.DATA_PATH + '%s_labels_%s.npy' % (file_name, len(labels)), 'wb') as f:
			np.save(f, np.array(labels))

		print('Data saved in "data/%s_data_%s.npy"' % (file_name, len(self.labels)))
		self.update_lists()

	def init_train_thread(self, event):
		if self.dataset:
			data = np.load(cfg.DATA_PATH + self.dataset, allow_pickle=True)
			labels = np.load(cfg.DATA_PATH + self.dataset.replace("data", "labels"),
				allow_pickle=True)
			train_data = [data, labels]

		self.t2 = threading.Thread(target=partial(self.train_model, train_data))
		self.t2.daemon = True
		self.t2.start()

	def create_calibration_widget(self):
		res = pyautogui.size()
		self.calibrate_widget = CalibrateWidget(self, res)
		self.calibrate_widget.show()

	def record_dataset(self):
		if not self.isRecording:
			self.data, self.labels = [], []
			self.isRecording = True
			print('Record started!')
		else:
			self.isRecording = False
			print('Record stopped!')
			self.save_dataset(self.data, self.labels)

	def train_model(self, dataset):
		model = Sequential()
		model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
		                 activation='relu',
		                 input_shape=(cfg.EYES_ROI_H, cfg.EYES_ROI_W, 3)))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
		model.add(Conv2D(16, (3, 3), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(64, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(64, activation='relu'))
		model.add(Dense(2, activation='sigmoid'))

		model.compile(loss='mean_squared_error', optimizer='adam')

		history = model.fit(x=dataset[0]/255, 
			y=dataset[1],
			validation_split=0.0,
			batch_size=100, 
			epochs=cfg.EPOCHS,
			verbose=1)
		print('loss: %s' % (history.history['loss'][-1]))
		file_name = 'model_%.2f_' % (history.history['loss'][-1])
		file_name += ''.join([rand.choice('qwertyasd123456') for i in range(3)])
		print('Saving model in %s' % file_name)
		model.save(cfg.MODELS_PATH + file_name+'.h5')
		self.update_lists()

	def cam_thread(self):
		cap = cv2.VideoCapture(cfg.CAM)

		face_detector = dlib.get_frontal_face_detector()
		shapes_predictor = dlib.shape_predictor("static/shape_predictor_68_face_landmarks.dat")

		eyes_roi = np.zeros((128, 256), dtype='int8')
		g = self.eyesLabel.geometry()
		w, h = g.width(), g.height()
		out_eyes_roi = cv2.resize(eyes_roi, (w, h))
		
		screen_size = pyautogui.size()
		print("SCREEN: %sx%s" % screen_size)

		while True:
			ret, frame = cap.read()
	
			if not ret:
				print('[ERROR] Camera disconnected!')
				break
				continue

			frame = cv2.resize(frame, (cfg.FRAME_W, cfg.FRAME_H))
			frame = cv2.flip(frame, 1)
			frame_src = np.copy(frame)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			faces = face_detector(gray, 1)

			for face in faces:
				shapes = shapes_predictor(gray, face)
				shapes = face_utils.shape_to_np(shapes)

				for shape in shapes:
					cv2.circle(frame, tuple(shape), 1, (0, 255, 0), -1)

				#Getting eyes
				x0, y0 = shapes[17][0], shapes[17][1]
				x1, y1 = shapes[26][0], shapes[29][1]

				eyes_roi = frame_src[y0:y1, x0:x1]
				try:
					eyes_roi = cv2.resize(eyes_roi, (cfg.EYES_ROI_W, cfg.EYES_ROI_H))
					cv2.rectangle(frame, (x0, y0), 
						(x1, y1), (0, 0, 255), 2)

					if self.testing:
						pred = self.model.predict(np.expand_dims(eyes_roi/255, 0))[0]
						x = pred[0] * screen_size[0]
						y = pred[1] * screen_size[1]
						pyautogui.moveTo(x, y)

					g = self.eyesLabel.geometry()
					w, h = g.width(), g.height()
					out_eyes_roi = cv2.resize(eyes_roi, (w, h))
						
				except cv2.error as e:
					print('[ERROR] Eyes roi is empty!')
				
			g = self.camLabel.geometry()
			w, h = g.width(), g.height()
			out_frame = cv2.resize(frame, (w, h))

			self.q.put({'frame' : np.copy(out_frame), 'eyes' : np.copy(out_eyes_roi)})

	def update_widgets(self):
		if self.q.empty():
			return

		data = self.q.get()
		
		frame = QtGui.QImage(data['frame'], data['frame'].shape[1], data['frame'].shape[0], data['frame'].strides[0], QtGui.QImage.Format_RGB888)
		frame = frame.rgbSwapped()
		self.camLabel.setPixmap(QtGui.QPixmap.fromImage(frame))

		if data['eyes'] is not None:
			eyes_roi = QtGui.QImage(data['eyes'], data['eyes'].shape[1], data['eyes'].shape[0], data['eyes'].strides[0], QtGui.QImage.Format_RGB888)
			eyes_roi = eyes_roi.rgbSwapped()

			self.eyes_roi = cv2.resize(data['eyes'], (cfg.EYES_ROI_W, cfg.EYES_ROI_H))
			self.eyesLabel.setPixmap(QtGui.QPixmap.fromImage(eyes_roi))

	def update_lists(self):
		data = os.listdir(cfg.DATA_PATH)
		datasets = [d for d in data if 'data' in d]

		self.dataList.clear()
		self.dataList.addItems(datasets)

		models = [f for f in os.listdir(cfg.MODELS_PATH) if '.DS' not in f]

		self.modelsList.clear()
		self.modelsList.addItems(models)

	def keyPressEvent(self, event):
		if event.key() == QtCore.Qt.Key_Escape:
			if not self.testing and self.model:
				self.testing = True
				print('Testing')
			else:
				self.testing = False

		if event.key() == QtCore.Qt.Key_Shift:
			self.update_lists()


if __name__ == "__main__":
	app = QtWidgets.QApplication(sys.argv)
	ui = Ui()
	app.exec_()
