import os
import csv
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from create_dir import Dir 

class csvSlicer:
	def __init__(self, *args):
		#data = []
		self.data = {}
		self.smoothed_data = {}
		self.a_slice = {}
		self.dumpDir = None
		self.isloaded = False 
		if len(args) == 0 :
			with open('google-cluster-data-1.csv', 'rb') as input_file:
				index = 0
				tracereader =  csv.reader(input_file, delimiter='\n', quoting=csv.QUOTE_NONE)
				#to skip the first line, traceviewer.next() is used
				tracereader.next()
				for row in tracereader:
					#self.__data.append(float(row[0].split()[5]))
					self.data[index] = float(row[0].split()[4])
					index = index + 1					
		    	print "size of loaded trace file: %d" %(len(self.data))
		    	self.get_rand_datadump()
		else:
			self.isloaded = True
			import json
			with open(args[0], 'rb') as json_data:
				self.a_slice = json.load(json_data)
				self.dumpDir = Dir(args[0].rsplit('/',1)[0])
				print len(self.data)

	def get_rand_datadump(self):
		self.a_slice = {}
		slice_initial_index = random.randint(0, len(self.data)-10001)
		#slice_initial_index = 2316696
		print "slice starts from %d" %(slice_initial_index)
		__max = 0.0
		for i in range(0,10000):
			self.a_slice[i] = self.data[slice_initial_index + i]
			#getting max value of the slice to use for scaling in the plot
			__max = float(self.a_slice[i]) if float(self.a_slice[i]) > __max else __max
		__scaler = 100.0/__max
		print "max value : %f\nand scaler is %f" %(__max, __scaler)
		for key in self.a_slice:
			self.a_slice[key] *= __scaler
			#print " self.a_slice[%d] = %f" % (key, self.a_slice[key])
		mean = np.mean(self.a_slice.values())
		std = np.std(self.a_slice.values())
		print "mean of the slice : %f\nstandard deviation of the slice : %f" %(mean, std)
		self.dumpDir = Dir(os.getcwd() + '/raw_CPU_usg_slice_'+`slice_initial_index`)
		
		import json
		with open(self.dumpDir.getDir() + '/raw_CPU_usg_slice_'+`slice_initial_index`+'_avg_'+`mean`+'_std_'+`std`+'.json', 'w') as outfile:
			json.dump(self.a_slice, outfile)


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
	    y=[]
	    length = len(self.a_slice)
	    for key in xrange(0,length):
	    	y.append(self.a_slice[str(key)])
	    #y = self.a_slice.values()
	    list_f = open(self.dumpDir.getDir() + '/list_f','wb') if not self.isloaded else open(self.dumpDir.getDir() + '/list_f_E','wb')
	    for item in y:
	    	print>> list_f, item
	    list_f.close()
	    
	    ylen =len(y)
	    if ylen % c !=0:
	        return None
	    fc =float(c)
	    ybar2 =sum([y[i] for i in range(c, 2 * c)])/ fc
	    ybar1 =sum([y[i] for i in range(c)]) / fc
	    b0 = 0.001 if (ybar2 - ybar1)/fc ==0 else (ybar2 - ybar1) / fc
	    if debug: print "b0 = ", b0
	 
	    #Compute for the level estimate a0 using b0 above.
	    tbar  =sum(i for i in range(1, c+1)) / fc
	    print tbar
	    a0 =ybar1  - b0 * tbar
	    if debug: print "a0 = ", a0
	 
	    #Compute for initial indices
	    I =[y[i] / (a0 + (i+1) * b0) for i in range(0, ylen)]
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
	        S[i+c] =gamma * y[i] / At + (1.0 - gamma) * S[i]
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
	    	self.smoothed_data[i+m] = (At + Bt* (m+1))* S[ylen + m]
	        #print "forecast at point ",i+m,":", self.smoothed_data[i+m]

	    import json
	    with open(self.dumpDir.getDir() + '/forecast.json','w') as outfile:
			json.dump(self.smoothed_data, outfile)
	    plt.plot(self.a_slice.values(), 'x')
	    plt.axis([0,10500,-10,120])
	    plt.plot(self.smoothed_data.values())
	    plt.show()


"""
x = csvSlicer()
print len(x.data)
x.get_rand_datadump()
print len(x.data)

x.holtwinters(0.2, 0.1, 0.05, 500)
plt.plot(x.data, 'x')
plt.axis([0,10500,-10,120])
plt.plot(x.smoothed_data.values())
plt.show()
"""