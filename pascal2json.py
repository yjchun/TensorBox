import json
import os
import struct
import imghdr
import numpy as np
from xml.etree import ElementTree

from lxml import etree
import random
from numplate import utils
from numplate import image_utils
# from skimage import io
from scipy.misc import imread, imsave

# run in data/ dir
base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../'
base_dir = os.path.abspath(base_dir)
PASCAL_XML_DIR = base_dir + '/data/pascal-def-xml/'
# JSON_PATH = base_dir + 'data/sample-list.json'
JSON_PATH = 'train-list.json'
JSON_TEST_PATH = 'test-list.json'
SPLIT_TEST = 0.1 # 10% is for test set

IMAGE_SIZE = (1024, 768)
IMAGE_DIR = 'images'



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

		# label = object_iter.find('name').text

		rect = {}
		rect['x1'] = float(bndbox.find('xmin').text)
		rect['x2'] = float(bndbox.find('xmax').text)
		rect['y1'] = float(bndbox.find('ymin').text)
		rect['y2'] = float(bndbox.find('ymax').text)
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
		bboxes = []
		for rect in entry['rects']:
			bboxes.append([rect['x1'],rect['y1'],rect['x2'],rect['y2']])
		image = imread(entry['image_path'], mode='RGB')

		image, bboxes = image_utils.resized_aspect_fill(image, IMAGE_SIZE, bboxes)

		newpath = os.path.split(entry['image_path'])[-1]
		newpath = os.path.join(savedir, newpath)
		imsave(newpath, image)

		entry['image_path'] = newpath
		entry['rects'] = []
		for bbox in bboxes:
			rect = {}
			rect['x1'] = bbox[0]
			rect['y1'] = bbox[1]
			rect['x2'] = bbox[2]
			rect['y2'] = bbox[3]
			entry['rects'].append(rect)

		# utils.show_image(image, bboxes)
	return samplelist


def main():
	print('Loading pascal XML dir: {}'.format(PASCAL_XML_DIR))
	samplelist = read_pascal_dir(PASCAL_XML_DIR)

	print('Read {} training images'.format(len(samplelist)))

	print('Resizing images (saving to {})...'.format(IMAGE_DIR))
	samplelist = resize_images(samplelist, IMAGE_DIR)

	random.shuffle(samplelist)
	num_test = int(round(SPLIT_TEST * len(samplelist)))
	test_list = samplelist[:num_test]
	train_list = samplelist[num_test:]

	# save resulting json file
	print('Saving data to {}'.format(JSON_PATH))
	with open(JSON_TEST_PATH, 'w') as savefile:
		json.dump(test_list, savefile)
	with open(JSON_PATH, 'w') as savefile:
		json.dump(train_list, savefile)


if __name__ == '__main__':
	main()
