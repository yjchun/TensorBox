import json
from smarteye import *
import random
from smarteye.annoapi import AnnoDB
from ml import image_util
from scipy.misc import imread, imsave


HYPE_PATH = 'hypes/overfeat_rezoom.json'


param = Namespace()
param.image_size = (0,0)
param.output_dir = 'output/train-images'
param.split_test = 0.1
param.json_path = ''
param.json_test_path = ''
param.debug = False
param.filter_comment = 'night' # ignore images including filter word in comment


def debug_show_image(image, rects):
	from ml.bounding_box import BBox
	from smarteye.misc_util import show_image_with_bbox
	bboxes = []
	for rect in rects:
		bboxes.append(BBox.from_coordinates(rect['x1'], rect['y1'], rect['x2'], rect['y2']))
	show_image_with_bbox(image, bboxes)


def sort_points(keypoints):
	kp = sorted(keypoints, key=lambda p: p[1])
	# if kp[0][0] > kp[1][0]:
	# 	kp[0], kp[1] = kp[1], kp[0]
	return kp


import datetime
unique_filename = str(random.randint(10000, 99999)) + str(datetime.datetime.now().time().microsecond)

def create_path(oldpath, savedir):
	filename = os.path.basename(oldpath)
	filename = os.path.splitext(filename)[0]
	newpath = os.path.join(savedir, '{}-{}.png'.format(filename, unique_filename))
	return newpath


def resize_images(samplelist, savedir):
	if not os.path.exists(savedir):
		os.mkdir(savedir)

	for entry in samplelist:
		image = imread(entry['image_path'], mode='RGB')

		# resize image
		image, scale = image_util.resized_aspect_fill(image, param.image_size)
		# save resized image
		newpath = create_path(entry['image_path'], savedir)
		if not param.debug:
			imsave(newpath, image)

		entry['image_path'] = newpath
		for rect in entry['rects']:
			rect['x1'] *= scale
			rect['y1'] *= scale
			rect['x2'] *= scale
			rect['y2'] *= scale

		# if param.debug:
		# 	debug_show_image(image, entry['rects'])

	return samplelist

def get_entry(imgpath, annolist):
	shapes = {}

	shapes['image_path'] = imgpath
	shapes['rects'] = []

	for annoinfo in annolist:
		#TODO: clip points with image
		points = sort_points(eval(annoinfo['annotation_data']))
		class_name = annoinfo['class_name']

		rect = {}
		rect['x1'] = points[0][0]
		rect['y1'] = points[0][1]
		rect['x2'] = points[1][0]
		rect['y2'] = points[1][1]
		rect['class_name'] = class_name
		rect['class_id'] = 1 # TODO:
		shapes['rects'].append(rect)

	return shapes


def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset_dir', help='Dataset directory', required=True)
	parser.add_argument('-y', '--hype_path', help='json Hype file path', default=HYPE_PATH)
	parser.add_argument('-b', '--dataset_db', help='Dataset Database File Path')
	parser.add_argument('-s', '--debug', action='store_true')
	parser.add_argument('-m', '--merge', action='store_true')
	parser.add_argument('-o', '--output')
	args = parser.parse_args()

	if args.debug:
		param.debug = True
	if args.output:
		param.output_dir = args.output

	# parse parameters
	annodb = AnnoDB(args.dataset_dir, args.dataset_db, create=False)

	# load hype file
	with open(args.hype_path, 'r') as f:
		H = json.load(f)

	param.image_size = (H['image_width'], H['image_height'])
	param.json_path = H['data']['train_idl']
	param.json_test_path = H['data']['test_idl']

	samplelist = []
	image_list = annodb.get_image_list()

	for imginfo in image_list:
		annolist = annodb.get_annotations(imginfo['id'])
		if len(annolist) == 0:
			continue
		if param.filter_comment and param.filter_comment in imginfo['comment']:
			continue

		entry = get_entry(annodb.image_path(imginfo['id']), annolist)
		if entry is None:
			continue
		samplelist.append(entry)


	print('Read {} training images'.format(len(samplelist)))

	# samplelist = filter_list(samplelist, FILTER_CLASSES)
	# print('filtered training images: {}'.format(len(samplelist)))

	print('Resizing images (saving to {})...'.format(param.output_dir))
	try:
		os.makedirs(param.output_dir)
	except OSError as exc:  # Python >2.5
		pass
	samplelist = resize_images(samplelist, param.output_dir)

	if args.merge and os.path.exists(param.json_path):
		with open(param.json_path, 'r') as f:
			l = json.load(f)
			samplelist = samplelist + l
		with open(param.json_test_path, 'r') as f:
			l = json.load(f)
			samplelist = samplelist + l
		print('Merged data (total={})'.format(len(samplelist)))

	random.shuffle(samplelist)
	num_test = int(round(param.split_test * len(samplelist)))
	test_list = samplelist[:num_test]
	train_list = samplelist[num_test:]

	# save resulting json file
	print('Saving data to {}'.format(param.json_path))
	if not param.debug:
		with open(param.json_test_path, 'w') as savefile:
			json.dump(test_list, savefile, indent=2)
		with open(param.json_path, 'w') as savefile:
			json.dump(train_list, savefile, indent=2)


if __name__ == '__main__':
	main()
