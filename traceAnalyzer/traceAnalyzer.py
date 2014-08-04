import os
import csv
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from create_dir import Dir
from threading import Thread
from scipy.stats.stats import pearsonr
from scipy.stats import gaussian_kde

class traceAnalyzer:
	def __init__(self, *args):
		#data = []
		self.data = []
		self.smoothed_data = []
		self.a_slice = []
		self.dumpDir = None
		self.isloaded = False
		self.inpt = "" 
		if args[1] == "--l":
			with open(args[0], 'rb') as input_file:
				index = 0
				tracereader =  csv.reader(input_file, delimiter='\n', quoting=csv.QUOTE_NONE)
				#to skip the first line, traceviewer.next() is used
				tracereader.next()
				for row in tracereader:
					self.data.append(row[0].split()[4])
					#self.data[index] = float(row[0].split()[4])
				print "size of loaded trace file: %d" %(len(self.data))
		    	self.get_rand_datadump()
		elif args[1] == "--r":
			self.isloaded = True
			with open(args[0], 'rb') as list_file:
				self.a_slice = list_file.read().splitlines()
				self.dumpDir = Dir(args[0].rsplit('/',1)[0])
				index = 0
				json_dump = {}
				for item in self.a_slice:
					self.a_slice[index] = float(item)
					json_dump[index] = item
					index += 1
				#self.a_slice = [float(value) for value in self.a_slice]
				print "size of re-loaded trace file: %d" %(len(self.a_slice))
				with open("KDE/_to_analyze_.json", 'wb') as json_file:
					json_file.write(json.dumps(json_dump))


	def get_rand_datadump(self):
		slice_initial_index = random.randint(0, len(self.data)-10001)
		print "\n--------------------------------------------------------------------------------\n"
		#slice_initial_index = 2316696
		print "slice starts from %d" %(slice_initial_index)
		__max = 0.0
		for i in xrange(0,10000):
			self.a_slice.append(self.data[slice_initial_index + i])
			#getting max value of the slice to use for scaling in the plot
			__max = float(self.a_slice[i]) if float(self.a_slice[i]) > __max else __max
		__scaler = 100.0/__max
		print "max value : %f\nand scaler is %f" %(__max, __scaler)
		
		self.a_slice = [float(value)*__scaler for value in self.a_slice]
		#print " self.a_slice[%d] = %f" % (key, self.a_slice[key])
		mean = np.mean(self.a_slice)
		std = np.std(self.a_slice)
		print "mean of the slice : %f\nstandard deviation of the slice : %f" %(mean, std)
		print "\n--------------------------------------------------------------------------------\n"
		self.dumpDir = Dir(os.getcwd() + '/raw_CPU_usg_slice_'+`slice_initial_index`)
		json_dump = {}
		index = 0
		with open(self.dumpDir.getDir() + '/raw_CPU_usg_slice_'+`slice_initial_index`+'_avg_'+`mean`+'_std_'+`std`, 'w') as outfile:
			for item in self.a_slice:
				outfile.write("%s\n" %item)
				json_dump[index] = item
				index += 1
		with open(self.dumpDir.getDir().rsplit('/',1)[0]+"/KDE/_to_analyze_.json", 'wb') as json_file:
			json_file.write(json.dumps(json_dump))



	def holtwinters(self, alpha, beta, gamma, c, debug=False):
	    """
	    y - time series data.
	    alpha , beta, gamma - exponential smoothing coefficients 
	                                      for level, trend, seasonal components.
	    c -  extrapolated future data points.
	          4 quarterly
	          7 weekly.
	          12 monthly
	 	The length of y must be a an integer multiple  (> 2) of c.
	    """
	    #Compute initial b and intercept using the first two complete c periods.
	    
	    #y=self.a_slice[0:10000] if self.isloaded else self.a_slice
	    y=self.a_slice
	    ylen =len(y)
	    self.smoothed_data = [0] * (ylen+c)
	    if debug:
	    	print "entered holtwinters", ylen
	    if ylen % c !=0:
	        return None
	    fc =float(c)
	    ybar2 =sum([y[i] for i in range(c, 2 * c)])/ fc
	    ybar1 =sum([y[i] for i in range(c)]) / fc
	    b0 = 0.001 if (ybar2 - ybar1)/fc ==0 else (ybar2 - ybar1) / fc
	    if debug: print "b0 = ", b0
	 
	    #Compute for the level estimate a0 using b0 above.
	    tbar  =sum(i for i in range(1, c+1)) / fc
	    a0 =ybar1  - b0 * tbar
	    if debug: print "a0 = ", a0
	 
	    #Compute for initial indices
	    I =[y[i] / (a0 + (i+1) * b0) for i in xrange(0, ylen)]
	    if debug: print "Initial indices = ", I
	 
	    S=[0] * (ylen+ c)
	    for i in range(c):
	        S[i] =(I[i] + I[i+c]) / 2.0
	 
	    #Normalize so S[i] for i in [0, c)  will add to c.
		div = 0.001 if sum([S[i] for i in range(c)])==0 else sum([S[i] for i in range(c)])
	    tS =c / div
	    for i in range(c):
	        S[i] *=tS
	        if debug: print "S[",i,"]=", S[i]
	 
	    # Holt - winters proper ...
	    if debug: print "Use Holt Winters formulae"
	    F =[0] * (ylen+ c)   
	    At =a0
	    Bt =b0
	    for i in range(ylen):
	        Atm1 =At
	        Btm1 =Bt
	        S[i] = 0.001 if S[i] == 0 else S[i]
	        At =alpha * y[i] / S[i] + (1.0-alpha) * (Atm1 + Btm1)
	        Bt =beta * (At - Atm1) + (1- beta) * Btm1
	        S[i+c] = (gamma * y[i] / At + (1.0 - gamma) * S[i]) if At != 0 else (gamma * y[i] / 0.001 + (1.0 - gamma) * S[i])
	        F[i]=(a0 + b0 * (i+1)) * S[i]
	        if F[i] >= 100:
	        	self.smoothed_data[i] = 100
	        elif F[i] < 0:
	        	self.smoothed_data[i] = 0
        	else:
	        	self.smoothed_data[i] = F[i]
        	#print "i=", i+1, "y=", y[i], "S=", S[i], "Atm1=", Atm1, "Btm1=",Btm1, "At=", At, "Bt=", Bt, "S[i+c]=", S[i+c], "F=", F[i]
	        #print i,y[i],F[i]

	    #Forecast for next c periods:
	    for m in range(c):
	    	forecast = (At + Bt* (m+1))* S[ylen + m]
	    	if forecast > 100:
	    		forecast = 100
	    	elif forecast < 0 :
	    		forecast = 0
	    	self.smoothed_data[i+m] = forecast
	        #print "forecast at point ",i+m,":", self.smoothed_data[i+m]

	    with open(self.dumpDir.getDir() + '/forecast','w') as outfile:
			for value in self.smoothed_data:
				outfile.write("%f\n" %value)
	    plt.plot(self.a_slice, 'x')
	    plt.axis([0,len(self.a_slice)+c,-10,120])
	    plt.plot(self.smoothed_data)
	    """
	    t = Thread(target=self.temps)
	    t.start()
	    if self.inpt != "":
	    	t.join()
	    """
	    plt.show()
	

	def dist_predictor(self, *args):
		dict_kde_comparer = {}
		if args[0]=="above":
			dist_file=open("KDE/outlier_data_above", 'r')
		elif args[0]=="below":
			dist_file=open("KDE/outlier_data_below", 'r')
		else:
			print "Missing/wrong argument in 'dist_predictor'"
		
		prev = []
		next = []
		count = 0
		dict_kde_comparer = json.load(dist_file)
		sorted_keys = sorted(dict_kde_comparer.keys())
		
		for i in range(len(sorted_keys)-1):
			prev = [0.0]*(len(self.a_slice)/4)
			next = [0.0]*(len(self.a_slice)/4)
			length = len(next)
			
			for key in dict_kde_comparer[sorted_keys[i]]:
				prev[int(key)%length] = float(dict_kde_comparer[sorted_keys[i]][key])
				
			for key in dict_kde_comparer[sorted_keys[i+1]]:
				next[int(key)%length] = float(dict_kde_comparer[sorted_keys[i+1]][key])
			
			grid = np.linspace(0,100,len(self.a_slice)/4)
			prev_kde = gaussian_kde(np.asarray(prev)).evaluate(grid).tolist()
			next_kde = gaussian_kde(np.asarray(next)).evaluate(grid).tolist()
			correlation_value = pearsonr(prev_kde,next_kde)[0]*100
			print "Correlation between outlier windows "+`str(sorted_keys[i])`+" and "+`str(sorted_keys[i+1])`+" is "+`pearsonr(prev_kde,next_kde)[0]*100`
			fig = plt.figure(1)
			fig.suptitle('Correlation between the slices: '+`correlation_value`, fontsize=15)
			plt.subplot(211)
			plt.plot(grid, prev_kde)
			plt.subplot(212)
			plt.plot(grid, next_kde)
			plt.xlabel("CPU utilization")
			plt.ylabel("Outlier distribution")
			plt.show()
			
			

	def match_outlier_plots(self):
		pass

	def show_outlier_plots(self):
		pass

	def to_print_in_thread(*args):
		for line in args:
			print line

	def general_kde_analysis(self, number_of_slices = 4, debug=False):
		number_of_slices = 4
		default_length = len(self.a_slice)/number_of_slices
		grid = np.linspace(0,100,len(self.a_slice)/number_of_slices)
		for i in range(number_of_slices-1):
			prev = self.a_slice[ (i*default_length) : ((i+1)*default_length)]
			next = self.a_slice[ ((i+1)*default_length) : ((i+2)*default_length)]
			if debug:
				print "\n\n", len(prev), len(next), "\n\n"
			prev_kde = gaussian_kde(np.asarray(prev)).evaluate(grid).tolist()
			next_kde = gaussian_kde(np.asarray(next)).evaluate(grid).tolist()
			correlation_value = pearsonr(prev_kde,next_kde)[0]*100
			print "Correlation between slice windows '"+`(i*default_length)`+"-"+`(i+1)*default_length`+"' and '"+`(i+1)*default_length`+"-"+`(i+2)*default_length`+"' is "+`correlation_value`
			fig = plt.figure(1)
			fig.suptitle('Correlation between the slices: '+`correlation_value`, fontsize=15)
			plt.subplot(211)
			plt.plot(grid, prev_kde)
			plt.subplot(212)
			plt.plot(grid, next_kde)
			plt.xlabel("CPU utilization")
			plt.ylabel("Outlier distribution")
			plt.show()