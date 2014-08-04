#!/usr/bin/python
import os
import sys
from traceAnalyzer.traceAnalyzer import traceAnalyzer
from traceAnalyzer.create_dir import *
from KDE.kde_analysis import main as kde_main
from KDE.kde_analysis import main as kde_main

#x = traceAnalyzer()
#x = traceAnalyzer('raw_CPU_usg_slice_2035742/raw_CPU_usg_slice_2035742_avg_1.5408333333333304_std_5.4967778931658016')
#x = traceAnalyzer('raw_CPU_usg_slice_1880356/raw_CPU_usg_slice_1880356_avg_2.9888571428571127_std_7.4830828471923034')
if len(sys.argv) != 3:
	print "Please provide the path to load or reload trace with load/reaload arguments.\
	\ni.e. <path/to/trace_dump> --l to load trace\
	\n<path/to/trace_dump> --r to reload trace"
elif len(sys.argv) == 3 and sys.argv[2]=="--l" or sys.argv[2]=="--r":
	x = traceAnalyzer(sys.argv[1], sys.argv[2])
	print "Please choose one of the following functions to perform on the loaded trace\n\
	1: Holt-Winter's Triple Exponential Smoothing\n\
	2: Kernel Density Analysis\n\
	3: Match Kernel Density Distribution sections\n\
	4: Match outlier Kernel Density Distribution sections\n"
	choice_two = raw_input("---------------------------------------------------------------------------------\n> ")
	if choice_two == "1":
		print "Holt-Winter's Triple Exponential Smoothing chosen"
		a_b_g = []
		while True:
			a_b_g =  raw_input("Enter values of 'alpha', 'beta', 'gamma' and number of forecast points separated by commas\n\
Default is alpha = 0.2, beta = 0.1, gamma = 0.05, forecast ="+ str(len(x.a_slice)/4) +"\n>").split(",")
			if a_b_g[0] == "q":
				sys.exit()
			if len(a_b_g) == 4:
				print "Executing analysis: holtwinters(%r,%r,%r,%r)" %(float(a_b_g[0]), float(a_b_g[1]), float(a_b_g[2]), int(a_b_g[3]))
				x.holtwinters(float(a_b_g[0]), float(a_b_g[1]), float(a_b_g[2]), int(a_b_g[3]))
				break
			else:
				chk =  raw_input("Wrong input!\nExecute with default values (Y/N)?\n> ")
				if chk == "Y" or "y":
					x.holtwinters(0.2,0.2,0.2,len(x.a_slice)/4)
					sys.exit()
	elif choice_two == "2":
		kde_main()
	elif choice_two == "3":
		#x.dist_predictor("above")
		x.general_kde_analysis()
	elif choice_two == "4":
		x.dist_predictor("above")
else:
	print "Please provide the path to load or re-load trace with load/realod arguments.\
	\ni.e. <path/to/trace_dump> --l to load trace\
	\n<path/to/trace_dump> --r to reload trace"


#x.holtwinters(0.2, 0.1, 0.05, 2000)