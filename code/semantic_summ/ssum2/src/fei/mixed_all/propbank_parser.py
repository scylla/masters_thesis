#!/usr/bin/python

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import re
import pickle

frames_dict = {} # store propbank frames concept and arg roles

class FrameProperties(object):
	def __init__(self, name = '', num_roles = 0):
		self.name = name
		self.num_roles = num_roles
		self.roles = {}

	def __repr__(self):
		return ("%s || %s" % (self.name, str(self.num_roles)))

if __name__ == '__main__':
	input_dir = '/Users/amit/Desktop/Thesis/propbank_frames/propbank-frames-master/frames'
	for filename in os.listdir(input_dir):
		if filename.endswith('.xml'):
			print "parsing ", filename
			file_path = os.path.join(input_dir, filename)
			DOMTree = xml.dom.minidom.parse(file_path)
			collection = DOMTree.documentElement
			rolesets = collection.getElementsByTagName('roleset')
			for roleset in rolesets:
				attr = re.sub(r'\.', '-', roleset.getAttribute('id'))
				name = roleset.getAttribute('name')
				roles = roleset.getElementsByTagName('role')
				frames_dict[attr] = FrameProperties(name, len(roles))
				for role in roles:
					descr = role.getAttribute('descr')
					number = role.getAttribute('n')
					frames_dict[attr].roles[number] = descr

	propbank_file = open("propbank_frames.pkl","wb")
	pickle.dump(frames_dict, propbank_file)
	propbank_file.close()
