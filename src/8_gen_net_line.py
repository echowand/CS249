import numpy as np

import itertools
from tqdm import tqdm
# run with pypy

import csv
import pandas as pd




def copy_columnwise():
	colnames = ['display_id', 'ad_id', 'clicked','ad_document_id']
	train_file = 'tmp/small_join.csv'
	data = pd.read_csv(train_file, usecols=colnames)
	data.to_csv('tmp/selected_small_join.csv', index=False)



def gen_displayid_adid():
	# inputfile = "../data/events.csv"
	inputfile = "tmp/selected_small_join.csv"
	fin = open(inputfile)
	u_dict = {}

	i = -1
	for l in tqdm(fin):
		i=i+1
		# print i
		if i == 0:
			continue
		l = l.rstrip().split(',')
		# print l
		uuid = l[3]
		displayid = l[0]
		adid = l[1]
		clicked = l[2]
		nodeid = str(displayid) + '_' + str(adid) + '_' + str(clicked)

		# print uuid, nodeid
		if uuid not in u_dict:
			u_dict[uuid]=[nodeid]
		else:
			u_dict[uuid].append(nodeid)
	# print u_dict
	# print "Done dict"
	fout = open("./tmp/nodeid_net.txt", 'w+')
	for k in tqdm(u_dict):
		mylist = u_dict[k]
		if len(mylist) > 1:
			for item in itertools.combinations(mylist,2):
				# print item[0], item[1]
				str1 = item[0] + " " + item[1] + "\n"
				fout.write(str1)

def mapping():
	with open('./tmp/nodeid_net.txt') as fin:
		fout = open('./tmp/nodeid_net_indexed.txt', 'w+')
		mapping = {}
		for l in fin:
			# print l
			l = l.rstrip().split(' ')
			# print l
			for node in l:
				label = node.split('_')[2]
				if node not in mapping:
					mapping[node] = [len(mapping), label]
				print mapping[node][0], 
				fout.write(str(mapping[node][0]) + ' ')
			print
			fout.write('\n')
	# print len(mapping)
	# break

# copy_columnwise()
gen_displayid_adid()	
mapping()
