#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
from utils import getLogger
from collections import namedtuple

logger = getLogger()

def separate_sum_body(input_filename, aligned):
	body_data = []
	summary_data = []
	curr_data = []
	is_summary = False
	with codecs.open(input_filename, 'r', 'utf-8') as infile:
		for line in infile:
			line = line.rstrip()
			if line == '':
				if is_summary:
					summary_data.append(curr_data)
				else:
					body_data.append(curr_data)
				is_summary = False
				curr_data = []
			else:
				if 'summary' in line:
					is_summary = True
				curr_data.append(line)

	return body_data, summary_data
	# print summary_data
	print "body_len , summary_len in %s\n " % input_filename
	print len(body_data), len(summary_data)

def write_to_file(output_filename, amr_data):
	with codecs.open(output_filename, 'w', 'utf-8') as outfile:
		for amr_list_items in amr_data:
			for amrs in amr_list_items:
				outfile.write('%s\n' % amrs)
			outfile.write('\n')


if __name__ == '__main__':

  gold_input_dir = '/Users/amit/Desktop/Thesis/jamr/biocorpus/amr_parsing/data/gold/'
  jamr_input_dir = '/Users/amit/Desktop/Thesis/jamr/biocorpus/amr_parsing/data/jamr/'

  gold_input_dirs = [os.path.join(gold_input_dir,o) for o in os.listdir(gold_input_dir) if os.path.isdir(os.path.join(gold_input_dir,o))]
  jamr_input_dirs = [os.path.join(jamr_input_dir,o) for o in os.listdir(jamr_input_dir) if os.path.isdir(os.path.join(jamr_input_dir,o))]

  for dirs in gold_input_dirs:
  	body_data, summary_data = separate_sum_body(dirs + '/amr-release-1.0-proxy.txt', False)
  	write_to_file(dirs + '/amr-release-1.0-cleaned-proxy.txt', body_data)
  	write_to_file(dirs + '/amr-release-1.0-cleaned-summary.txt', summary_data)

  for dirs in jamr_input_dirs:
  	body_data, summary_data = separate_sum_body(dirs + '/amr-release-1.0-proxy.aligned', True)
  	write_to_file(dirs + '/amr-release-1.0-cleaned-proxy.aligned', body_data)
  	write_to_file(dirs + '/amr-release-1.0-cleaned-summary.aligned', summary_data)