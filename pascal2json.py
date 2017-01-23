import json
import os
from xml.etree import ElementTree

from lxml import etree
import random
from numplate.config import config
from ml import image_util
from scipy.misc import imread, imsave


# hype file path
HYPE_PATH = 'hypes/overfeat_rezoom.json'

# run in data/ dir
PASCAL_XML_DIR = config.base_dir + '/data/pascal-def-xml/'
SPLIT_TEST = 0.1 # 10% is for test set


# load hype file
with open(HYPE_PATH, 'r') as f:
	H = json.load(f)

IMAGE_SIZE = (H['image_width'], H['image_height'])
IMAGE_DIR = 'output/train-images'
JSON_PATH = H['data']['train_idl']
JSON_TEST_PATH = H['data']['test_idl']

assert H['num_classes'] == len(config.class_list)


# one of
# plate_new_small, plate_new_large, plate_new_com_small, plate_new_com_large, plate_new_2004, plate_old, plate_con, plate_temp
# defined in tools/labelimg/data/predefined_classes.txt
FILTER_CLASSES = []


def read_pascal(filepath):
	"""
	Read single pascal xml file
	:param filepath: xml file path
	:return: (image-filename, [[xmin,ymin,xmax,ymax,one-hot encoding of class-name],...])
	xmin,... is in relative coordinate.
	"""
	assert filepath.endswith('.xml'), "Unsupport file format"
	parser = etree.XMLParser(encoding='utf-8')
	xmltree = ElementTree.parse(filepath, parser=parser).getroot()
	path = xmltree.find('path').text
	filename = os.path.split(path)[-1]
	path = '../data/training/' + filename
	if not os.path.isfile(path):
		print('File not exist, Ignored: {}'.format(filename))
		return None

	shapes = {}
	shapes['image_path'] = path
	shapes['rects'] = []

	for object_iter in xmltree.findall('object'):
		bndbox = object_iter.find("bndbox")
		if bndbox is None:
			raise Exception('failed to retrieve bndbox')

		label = object_iter.find('name').text

		rect = {}
		rect['x1'] = float(bndbox.find('xmin').text)
		rect['x2'] = float(bndbox.find('xmax').text)
		rect['y1'] = float(bndbox.find('ymin').text)
		rect['y2'] = float(bndbox.find('ymax').text)
		rect['class_name'] = label
		rect['class_id'] = config.class_list.index(label)
		shapes['rects'].append(rect)

	return shapes


def read_pascal_dir(path):
	listdata = []
	filelist = os.listdir(path)
	for file in filelist:
		if not file.endswith('.xml'):
			continue
		entry = read_pascal(os.path.join(path, file))
		if entry is None:
			continue
		listdata.append(entry)
	return listdata


def resize_images(samplelist, savedir):
	if not os.path.exists(savedir):
		os.mkdir(savedir)

	for entry in samplelist:
		image = imread(entry['image_path'], mode='RGB')

		# resize image
		image, scale = image_util.resized_aspect_fill(image, IMAGE_SIZE)
		# save resized image
		newpath = os.path.split(entry['image_path'])[-1]
		newpath = os.path.join(savedir, newpath)
		imsave(newpath, image)


		entry['image_path'] = newpath
		for rect in entry['rects']:
			rect['x1'] *= scale
			rect['y1'] *= scale
			rect['x2'] *= scale
			rect['y2'] *= scale

		# utils.show_image(image, bboxes)
	return samplelist


def filter_list(samplelist, classlist):
	newlist = []
	if len(classlist) == 0:
		return samplelist
	for entry in samplelist:
		rects = []
		for rect in entry['rects']:
			# add bounding box only if it is in classlist
			label = rect['class_name']
			if label in classlist:
				rects.append(rect)
		# if there is a bounding box left
		if len(rects) > 0:
			entry['rects'] = rects
			newlist.append(entry)

	return newlist


def main():

	print('Loading pascal XML dir: {}'.format(PASCAL_XML_DIR))
	samplelist = read_pascal_dir(PASCAL_XML_DIR)

	print('Read {} training images'.format(len(samplelist)))

	samplelist = filter_list(samplelist, FILTER_CLASSES)
	print('filtered training images: {}'.format(len(samplelist)))

	print('Resizing images (saving to {})...'.format(IMAGE_DIR))
	try:
		os.makedirs(IMAGE_DIR)
	except OSError as exc:  # Python >2.5
		pass
	samplelist = resize_images(samplelist, IMAGE_DIR)

	random.shuffle(samplelist)
	num_test = int(round(SPLIT_TEST * len(samplelist)))
	test_list = samplelist[:num_test]
	train_list = samplelist[num_test:]

	# save resulting json file
	print('Saving data to {}'.format(JSON_PATH))
	with open(JSON_TEST_PATH, 'w') as savefile:
		json.dump(test_list, savefile, indent=2)
	with open(JSON_PATH, 'w') as savefile:
		json.dump(train_list, savefile, indent=2)


if __name__ == '__main__':
	main()
