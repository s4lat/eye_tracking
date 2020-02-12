import numpy as np
DATA_PATH = 'data/'
MODELS_PATH = 'models/'

FRAME_W, FRAME_H = 640, 360
EYES_ROI_W, EYES_ROI_H = 64, 32
EPOCHS = 40

ball_positions = np.array([[0.1, 0.13], [0.5, 0.13], [0.8, 0.13],
										 [0.1, 0.45], [0.5, 0.45], [0.8, 0.45],
										 [0.1, 0.75], [0.5, 0.75]])

CAM = 0