import tensorflow as tf
import os
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_forward
from evaluate import add_rectangles

from scipy.misc import imread
import argparse

from smarteye import *
import smarteye.misc_util

from ml.bounding_box import BBox
from ml import keras_util
import logging

log = logging.getLogger(__name__)


# display boxes only above confidence
CONFIDENCE = 0.5
WEIGHT_FILE = config.base_dir+'/TensorBox/output/plate/save.ckpt-310000'
#WEIGHT_FILE = config.base_dir+'/TensorBox/output/voc_cars/save.ckpt-490000'

#HYPES_FILE = config.base_dir+'/TensorBox/hypes/overfeat_rezoom.json'
HYPES_FILE = os.path.join(os.path.dirname(WEIGHT_FILE), 'hypes.json')

args = None

class DetectPlate(object):
	def __init__(self, weight_file, hypes_file=None, confidence=CONFIDENCE, per_process_gpu_memory_fraction=None):
		self.weight_file = weight_file
		if hypes_file is None:
			hypes_file = os.path.join(os.path.dirname(weight_file), 'hypes.json')
		print('Loading model: {}'.format(hypes_file))
		with open(hypes_file, 'r') as f:
			self.H = json.load(f)

		self.pred_boxes, self.pred_confidences = self._load_graph(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
		self.image = None
		self.bboxes = []

		self.confidence_max = confidence

		self.warm_up()


	def _load_graph(self, per_process_gpu_memory_fraction=None):
		#tf.reset_default_graph()

		graph = tf.Graph()
		sess = keras_util.create_tf_session(graph=graph, per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
		with sess.as_default():
			with graph.as_default():
				x_in = tf.placeholder(tf.float32, name='x_in', shape=[self.H['image_height'], self.H['image_width'], 3])
				if self.H['use_rezoom']:
					pred_boxes, pred_logits, pred_confidences,\
					pred_confs_deltas, pred_boxes_deltas = build_forward(self.H,
																		tf.expand_dims(x_in, 0),
																		'test',
																		reuse=None)

					grid_area = self.H['grid_height'] * self.H['grid_width']
					pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * self.H['rnn_len'], self.H['num_classes']])),
												  [grid_area, self.H['rnn_len'], self.H['num_classes']])
					if self.H['reregress']:
						pred_boxes = pred_boxes + pred_boxes_deltas
				else:
					pred_boxes, pred_logits, pred_confidences = build_forward(self.H, tf.expand_dims(x_in, 0), 'test', reuse=None)

				saver = tf.train.Saver()
				sess.run(tf.global_variables_initializer())
				saver.restore(sess, self.weight_file)

				self.sess = sess
				self.x_in = x_in
				return pred_boxes, pred_confidences


	def warm_up(self):
		self.detect(image_util.generate_random_image((10, 10, 3)))


	def detect(self, image):
		"""Detect number plate from input images
		:param image: numpy array image in shape of (image_height, image_width, 3 channels) (RGB format).
		:return: list of BBox
		"""
		t = time.time()

		#sess = tf.get_default_session()
		#x_in = tf.get_default_graph().get_tensor_by_name('x_in:0')

		# resize image...
		with self.sess.as_default():
			resized_img, resize_scale = image_util.resized_aspect_fill(image, (self.H['image_width'], self.H['image_height']))

			#log.debug('detect resize image took %f seconds', time.time() - t)
			t = time.time()

			feed = {self.x_in: resized_img}
			(np_pred_boxes, np_pred_confidences) = self.sess.run([self.pred_boxes, self.pred_confidences], feed_dict=feed)
			# boxed_image is not needed
			boxed_image, rects = add_rectangles(self.H, [resized_img], np_pred_confidences, np_pred_boxes,
											use_stitching=True, rnn_len=self.H['rnn_len'],
												min_conf=self.confidence_max,
											show_suppressed=False, boxed_image=False)

			#log.debug('detect image took %f seconds', time.time() - t)

			#print(np_pred_boxes)
			#print(np_pred_confidences)
			bboxes = []
			for r in rects[:]:
				if r.score >= self.confidence_max:
					r.rescale(1/resize_scale)
					bbox = BBox(r.x1, r.y1, x2=r.x2, y2=r.y2)
					bbox.confidence = r.score
					bbox.class_id = r.silhouetteID
					bboxes.append(bbox)

			self.bboxes = bboxes
			self.image = image
			return bboxes


	def get_cropped_images(self, image=None, padding=0.1):
		if image is None:
			bboxes = self.bboxes
			image = self.image
		else:
			bboxes = self.detect(image)

		plate_images = []
		for bbox in bboxes:
			# enlarge bbox by 20%
			bbox.pad(bbox.width*padding, bbox.height*padding)
			cropped = bbox.crop_image(image)
			plate_images.append(cropped)

		return plate_images

	def close(self):
		self.sess.close()

		global DEFAULT_CAR_DETECTOR, DEFAULT_PLATE_DETECTOR
		if DEFAULT_CAR_DETECTOR == self:
			DEFAULT_CAR_DETECTOR = None
		if DEFAULT_PLATE_DETECTOR == self:
			DEFAULT_PLATE_DETECTOR = None


DEFAULT_CAR_DETECTOR = None
def get_car_detector(**kwargs):
	global DEFAULT_CAR_DETECTOR
	if DEFAULT_CAR_DETECTOR is None:
		DEFAULT_CAR_DETECTOR = DetectPlate(config.runtime_dir+'/tb_data/voc_coco_cars/save.ckpt-2900000',
										   confidence=config.car_detect_confidence, **kwargs)
	return DEFAULT_CAR_DETECTOR

DEFAULT_PLATE_DETECTOR = None
def get_plate_detector(**kwargs):
	global DEFAULT_PLATE_DETECTOR
	if DEFAULT_PLATE_DETECTOR is None:
		DEFAULT_PLATE_DETECTOR = DetectPlate(config.runtime_dir+'/tb_data/plate/save.ckpt-680000',
											 confidence=config.plate_detect_confidence, **kwargs)
	return DEFAULT_PLATE_DETECTOR


def detect_video(path):
	from ml.video_util import VideoCapture
	import Tkinter
	from PIL import Image, ImageTk

	d = DetectPlate(WEIGHT_FILE, HYPES_FILE)
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
		boxed_image = image_util.image_with_boxes(d.image, d.bboxes)
		tkimg[0] = ImageTk.PhotoImage(Image.fromarray(boxed_image))
		label.config(image=tkimg[0])
		root.update_idletasks()
		root.after(delay, update_video)

	update_video()
	root.mainloop()


def validate_car_box(width, height):
	return True


def process_image(d, path):
	if not image_util.isfile_image(path):
		print('Skipping {}'.format(path))
		return
	img = imread(path, mode='RGB')
	t = time.time()
	boxes = d.detect(img)
	print('elapsed: {:.3f}'.format(time.time() - t))
	for r in boxes:
		print('class: {}, confidence: {:.2f}'.format(r.class_id, r.confidence))

	if args.extract:
		index = 0
		from scipy.misc import imsave
		if not os.path.exists(args.extract):
			os.mkdir(args.extract)
		fname = os.path.splitext(os.path.basename(path))[0]
		cropped = d.get_cropped_images(padding=args.padding)
		for img in cropped:
			outpath = os.path.join(args.extract, '{}-{}.png'.format(fname, index))
			print('Saving {}'.format(os.path.basename(outpath)))
			imsave(outpath, img)
			index += 1
	else:
		# display image and confidence
		smarteye.misc_util.show_image_with_bbox(d.image, d.bboxes)


def main():
	global CONFIDENCE, WEIGHT_FILE, HYPES_FILE, args
	from ml.video_util import isfile_video

	parser = argparse.ArgumentParser()
	parser.add_argument('file', nargs='+')
	parser.add_argument('-w', '--weight_file')
	parser.add_argument('-c', '--confidence',
						help='Display only above confidence threshold. [0 - 1.0]',
						type=float)
	parser.add_argument('-y', '--hype')
	parser.add_argument('-d', '--detector', choices=['car', 'plate'])
	parser.add_argument('-x', '--extract', help='Output directory for cropped images')
	parser.add_argument('-p', '--padding', default=0.2, type=float)
	args = parser.parse_args()

	if len(args.file) == 0:
		print('No input file')
		return

	if args.weight_file is not None:
		WEIGHT_FILE = args.weight_file
		HYPES_FILE = os.path.join(os.path.dirname(WEIGHT_FILE), 'hypes.json')
	if args.confidence is not None:
		CONFIDENCE = args.confidence
	if args.hype is not None:
		HYPES_FILE = args.hype

	# if 'video' in magic.from_file(args.file[0], mime=True):
	if isfile_video(args.file[0]) or args.file[0] == 'video':
		path = args.file[0]
		if args.file[0] == 'video':
			path = 0
		detect_video(path)

	else:
		if args.detector == 'car':
			d = get_car_detector()
		elif args.detector == 'plate':
			d = get_plate_detector()
		else:
			d = DetectPlate(WEIGHT_FILE, HYPES_FILE)
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

