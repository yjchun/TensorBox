import tensorflow as tf
import os
import json
from skimage.transform import resize
import sys

import os

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.rect import Rect
from evaluate import add_rectangles

import numpy as np
from scipy.misc import imread
import argparse

from numplate.config import config
import numplate.utils
from ml import image_util
from ml.video_util import VideoCapture, isfile_video
from ml.bounding_box import BBox
import time

# display boxes only above confidence
CONFIDENCE = 0.7
WEIGHT_FILE = config.base_dir+'/TensorBox/output/save.ckpt-1140000'

HYPES_FILE = config.base_dir+'/TensorBox/hypes/overfeat_rezoom.json'
with open(HYPES_FILE, 'r') as f:
	H = json.load(f)


class DetectPlate(object):
	def __init__(self):
		self.pred_boxes, self.pred_confidences = self._load_graph()
		self.image = None
		self.bboxes = []
		self.boxed_image = None

	def _load_graph(self):
		tf.reset_default_graph()
		x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
		if H['use_rezoom']:
			pred_boxes, pred_logits, pred_confidences,\
			pred_confs_deltas, pred_boxes_deltas = build_forward(H,
																tf.expand_dims(x_in, 0),
																'test',
																reuse=None)

			grid_area = H['grid_height'] * H['grid_width']
			pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], H['num_classes']])),
										  [grid_area, H['rnn_len'], H['num_classes']])
			if H['reregress']:
				pred_boxes = pred_boxes + pred_boxes_deltas
		else:
			pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)

		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		saver.restore(sess, WEIGHT_FILE)

		self.sess = sess
		self.x_in = x_in
		return pred_boxes, pred_confidences


	def detect(self, image):
		"""Detect number plate from input images
		:param image: numpy array image in shape of (image_height, image_width, 3 channels) (RGB format).
		:return: list of BBox
		"""
		#sess = tf.get_default_session()
		#x_in = tf.get_default_graph().get_tensor_by_name('x_in:0')

		# resize image...
		resized_img, resize_scale = image_util.resized_aspect_fill(image, (H['image_width'], H['image_height']))

		feed = {self.x_in: resized_img}
		(np_pred_boxes, np_pred_confidences) = self.sess.run([self.pred_boxes, self.pred_confidences], feed_dict=feed)
		# TODO: boxed_image is not needed
		boxed_image, rects = add_rectangles(H, [resized_img], np_pred_confidences, np_pred_boxes,
										use_stitching=True, rnn_len=H['rnn_len'], min_conf=CONFIDENCE,
										show_suppressed=False)
		#print('elapsed: {}'.format(time.time() - t))

		#print(np_pred_boxes)
		#print(np_pred_confidences)
		bboxes = []
		for r in rects[:]:
			#print(r.score)
			if r.score >= CONFIDENCE:
				r.rescale(1/resize_scale)
				bbox = BBox(r.x1, r.y1, x2=r.x2, y2=r.y2)
				bbox.confidence = r.score
				bbox.class_id = r.silhouetteID
				bboxes.append(bbox)

		self.bboxes = bboxes
		self.image = image
		self.boxed_image = boxed_image
		return bboxes


	def get_plates(self, image=None, padding=0.2):
		if image is None:
			bboxes = self.bboxes
			image = self.image
		else:
			bboxes = self.detect(image)

		plate_images = []
		for bbox in bboxes:
			# enlarge bbox by 20%
			bbox.pad(bbox.height*padding)
			cropped = bbox.crop_image(image)
			plate_images.append(cropped)

		return plate_images


def detect_video(path):
	import Tkinter
	from PIL import Image, ImageTk

	d = DetectPlate()
	video = VideoCapture(path)

	def button_click_exit_mainloop(event):
		event.widget.quit()  # this will cause mainloop to unblock.

	root = Tkinter.Tk()
	#root.bind("<Button>", button_click_exit_mainloop)
	root.geometry('+%d+%d' % (1024, 768))
	label = Tkinter.Label(root)
	label.pack()
	tkimg = [None]

	delay = 100  # in milliseconds

	def update_video():
		image = video.get()
		if image is None:
			exit(0)
		d.detect(image)
		tkimg[0] = ImageTk.PhotoImage(Image.fromarray(d.boxed_image))
		label.config(image=tkimg[0])
		root.update_idletasks()
		root.after(delay, update_video)

	update_video()
	root.mainloop()


def process_image(d, path):
	img = imread(path, mode='RGB')
	t = time.time()
	boxes = d.detect(img)
	print('elapsed: {:.3f}'.format(time.time() - t))
	for r in boxes:
		print('class: {}, confidence: {:.2f}'.format(r.class_id, r.confidence))
	# display image and confidence
	numplate.utils.show_image_with_bbox(d.image, d.bboxes)


def main():
	global CONFIDENCE, WEIGHT_FILE, HYPES_FILE, H

	parser = argparse.ArgumentParser()
	parser.add_argument('file', nargs='+')
	parser.add_argument('-w', '--weight_file')
	parser.add_argument('-c', '--confidence',
						help='Display only above confidence threshold. [0 - 1.0]',
						type=float)
	parser.add_argument('-y', '--hype')
	parser.print_help()
	args = parser.parse_args()

	if len(args.file) == 0:
		print('No input file')
		return

	if args.weight_file is not None:
		WEIGHT_FILE = args.weight_file
	if args.confidence is not None:
		CONFIDENCE = args.confidence
	if args.hype is not None:
		HYPES_FILE = args.hype

	print('Loading model: {}'.format(WEIGHT_FILE))
	with open(HYPES_FILE, 'r') as f:
		H = json.load(f)

	# if 'video' in magic.from_file(args.file[0], mime=True):
	if isfile_video(args.file[0]) or args.file[0] == 'video':
		path = args.file[0]
		if args.file[0] == 'video':
			path = 0
		detect_video(path)

	else:
		d = DetectPlate()
		for fn in args.file:
			if fn.endswith('json'):
				with open(fn, 'r') as f:
					testlist = json.load(f)
				for item in testlist:
					image_path = item['image_path']
					process_image(d, image_path)
				continue
			else:
				process_image(d, fn)


if __name__ == '__main__':
	main()

