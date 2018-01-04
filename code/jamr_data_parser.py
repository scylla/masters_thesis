#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import codecs
import argparse

def filter_amrs(jamr_file_path):
	jamr_out_file = "/Users/amit/Desktop/Thesis/jamr_data_parser/amrs_data_10000_"
	with codecs.open(jamr_file_path, 'rb', 'utf-8') as in_file:
		with codecs.open(jamr_out_file + 'raw', 'wb', 'utf-8') as out_file_raw:
			with codecs.open(jamr_out_file + 'linear', 'wb', 'utf-8') as out_file_linear:
				amr_list = []
				amr_string = ''
				for line in in_file:
					line = line.rstrip()
					if line == '' and amr_list:
						for amr_pieces in amr_list:
							amr_string += amr_pieces.lstrip() + " "
							out_file_raw.write("%s \n" % (amr_pieces.lstrip()))
						out_file_linear.write("%s \n" % (amr_string))
						out_file_raw.write("\n")
						amr_string = ''
						amr_list = []
					else:
						if not line.startswith('#'):
							amr_list.append(line)
	print "process completed"

def filter_tokens(jamr_file_path):
	tokens_out_file = "/Users/amit/Desktop/Thesis/jamr_data_parser/amrs_data_10000_tokens"
	with codecs.open(jamr_file_path, 'rb', 'utf-8') as in_file:
		with codecs.open(tokens_out_file + 'raw', 'wb', 'utf-8') as out_file:
			for line in in_file:
				line = line.rstrip()
				if line.startswith('# ::tok'):
					out_file.write("%s \n" % line[8:].lstrip())

if __name__ == '__main__':
	argparser = argparse.ArgumentParser()
	argparser.add_argument("-jf", "--jamr_file", type=str, help="jamr input file", required=True)
	argparser.add_argument("-s", "--switch", type=str, help="amr, tok", required=True)
	args = argparser.parse_args()
	if args.switch == 'amr':
		filter_amrs(args.jamr_file)
	elif args.switch == 'tok':
		filter_tokens(args.jamr_file)
