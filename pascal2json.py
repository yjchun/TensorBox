import json
import os
import struct
import imghdr
import numpy as np
from xml.etree import ElementTree

from lxml import etree


# run in data/ dir
base_dir = os.path.dirname(os.path.abspath(__file__)) + '/../'
base_dir = os.path.abspath(base_dir)
PASCAL_XML_DIR = base_dir + '/data/pascal-def-xml/'
# JSON_PATH = base_dir + 'data/sample-list.json'
JSON_PATH = 'sample-list.json'


# http://stackoverflow.com/questions/8032642/how-to-obtain-image-size-using-standard-python-class-without-using-external-lib
def get_image_size(fname):
	'''Determine the image type of fhandle and return its size.
	from draco'''
	with open(fname, 'rb') as fhandle:
		head = fhandle.read(24)
		if len(head) != 24:
			return None,  None
		if imghdr.what(fname) == 'png':
			check = struct.unpack('>i', head[4:8])[0]
			if check != 0x0d0a1a0a:
				return None, None
			width, height = struct.unpack('>ii', head[16:24])
		elif imghdr.what(fname) == 'gif':
			width, height = struct.unpack('<HH', head[6:10])
		elif imghdr.what(fname) == 'jpeg':
			try:
				fhandle.seek(0) # Read 0xff next
				size = 2
				ftype = 0
				while not 0xc0 <= ftype <= 0xcf:
					fhandle.seek(size, 1)
					byte = fhandle.read(1)
					while ord(byte) == 0xff:
						byte = fhandle.read(1)
					ftype = ord(byte)
					size = struct.unpack('>H', fhandle.read(2))[0] - 2
				# We are at a SOFn block
				fhandle.seek(1, 1)  # Skip `precision' byte.
				height, width = struct.unpack('>HH', fhandle.read(4))
			except Exception: #IGNORE:W0703
				return
		else:
			return
		return width, height


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

	shapes = {}
	shapes['image_path'] = '../data/training/' + filename
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
		listdata.append(entry)
	return listdata


def main():
	print('Loading pascal XML dir: {}'.format(PASCAL_XML_DIR))
	samplelist = read_pascal_dir(PASCAL_XML_DIR)
	print('Read {} training images'.format(len(samplelist)))

	# save resulting json file
	print('Saving data to {}'.format(JSON_PATH))
	with open(JSON_PATH, 'w') as savefile:
		json.dump(samplelist, savefile)


if __name__ == '__main__':
	main()
