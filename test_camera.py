import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.real_world.video_recorder import VideoRecorder
from umi.common.usb_util import get_sorted_v4l_paths

video_recorder = VideoRecorder.create_hevc_nvenc(
	shm_manager = SharedMemoryManager(),
	fps = 60,
	input_pix_fmt = 'bgr24',
	bit_rate = 6000*1000)

dev_video_paths = get_sorted_v4l_paths()
for i, path in enumerate(dev_video_paths):
	print(path)
	cap = cv2.VideoCapture(path, cv2.CAP_V4L2)

	if not cap.isOpened():
		print("failed to open cap")
		exit()
		
	#ret, frame = cap.read()
	retl = cap.grab()
	print(retl)
	
	#frame = video_recorder.get_img_buffer()
	frame = np.zeros((1080, 1920, 3))
	print(type(frame))
	print(frame.shape)
	ret, frame = cap.retrieve(frame)
	print(ret)

	#if not ret:
		#print("failed to read frame")
		#break

	#cv2.imshow('Camera Preview', frame)

		
cap.release()
cv2.destroyAllWindows()
print("cameras have been freed")
