#!/usr/bin/python
import sys
import os
from random import randrange

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print "usage: ./extract_10k.py <path_to_google_trace_data>.csv"
	elif not os.path.exists(sys.argv[1]):
		print "Invalid path. %r does not exist." %(sys.argv[1])
	else:
		with open(sys.argv[1],'r') as cluster_file:
			ls = [line for line in cluster_file]
			randnum = randrange(len(ls)-10000)
			with open('google_cluster_slice_10k_from_'+`randnum`+'.csv','w') as dumpfile:
				dumpfile.write(ls[0])
				for count in xrange(10000):
					dumpfile.write(ls[count+randnum])