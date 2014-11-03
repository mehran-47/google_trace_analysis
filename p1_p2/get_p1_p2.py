#!/usr/bin/python
import sys
import numpy as np
import matplotlib.pyplot as plt

def median(lst):
    sortedLst = sorted(lst)
    lstLen = len(lst)
    index = (lstLen - 1) // 2
    if (lstLen % 2):
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1])/2.0

def mean(lst):
	return sum(lst)/len(lst)

def std(lst):
	avg = mean(lst)
	return (mean([(element-avg)**2 for element in lst]))**(0.5)

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def get_outliers(sliceLoaded, mva ,**kwargs):
	static_outliers_above = {}
	static_outliers_below = {}
	elastic_outliers_above = {}
	elastic_outliers_below = {}
	med = median(sliceLoaded)
	stddev = std(sliceLoaded)
	s_ceiling = med+stddev if med+stddev < 100.0 else 100.0
	s_floor = med-stddev if med-stddev > 0.0 else 0.0
	for index in xrange(len(mva)):
		ceiling = mva[index]+stddev if mva[index]+stddev < 100.0 else 100.0
		floor = mva[index]-stddev if mva[index]-stddev > 0.0 else 0.0
		if sliceLoaded[index] > ceiling:
			elastic_outliers_above[index] = sliceLoaded[index]
		elif sliceLoaded[index] < floor:
			elastic_outliers_below[index] = sliceLoaded[index]
		if sliceLoaded[index] > s_ceiling:
			static_outliers_above[index] = sliceLoaded[index]
		elif sliceLoaded[index] < s_floor:
			static_outliers_below[index] = sliceLoaded[index]
	gap = len(sliceLoaded) - len(mva)
	for index in xrange(len(sliceLoaded)):
		if sliceLoaded[index] > s_ceiling:
			static_outliers_above[index] = sliceLoaded[index]
		elif sliceLoaded[index] < s_floor:
			static_outliers_below[index] = sliceLoaded[index]
	if kwargs.get('verbose')==True:
		print('ceiling : %f\nfloor : %f\nmedian : %f\nsandard deviation : %f\n'%(ceiling,floor,med,stddev))
		print('P1:%f'%(float(len(elastic_outliers_above.keys()))/len(sliceLoaded)))
		print('P2:%r'%(float(len(elastic_outliers_below.keys()))/len(sliceLoaded)))
	return {'elastic_outliers_above':elastic_outliers_above,\
	'elastic_outliers_below':elastic_outliers_below,\
	'static_outliers_above':static_outliers_above,\
	'static_outliers_below':static_outliers_below,\
	'P0':float(len(static_outliers_below.keys())+len(static_outliers_above.keys()))/len(sliceLoaded),\
	'P1':float(len(elastic_outliers_below.keys())+len(elastic_outliers_above.keys()))/len(sliceLoaded)}

def get_alerts(vUp, sliceLoaded):
	alertflag = False
	alertCount = 0
	overprov_alert = 0
	underprov_alert = 0
	mva = movingaverage(sliceLoaded,window_size)
	outliers = get_outliers(sliceLoaded, mva)
	for index in range(len(sliceLoaded)):
		if outliers['elastic_outliers_above'].get(index)!=None:
			'''
			consec_list = [outliers['elastic_outliers_above'].get(count) for count in range(index,index+vUp) 
				if outliers['elastic_outliers_above'].get(count)!=None]
			print(len(consec_list))
			'''
			if len([outliers['elastic_outliers_above'].get(count) for count in range(index,index+vUp) 
				if outliers['elastic_outliers_above'].get(count)!=None])>=vUp:
				underprov_alert+=1
		elif outliers['elastic_outliers_below'].get(index)!=None:
			if len([outliers['elastic_outliers_below'].get(count) for count in range(index,index+vUp) 
				if outliers['elastic_outliers_below'].get(count)!=None])>=vUp:
				overprov_alert+=1
	return { 'overprov_alerts':overprov_alert,\
	'underprov_alerts':underprov_alert,\
	'num_alerts':overprov_alert+underprov_alert,\
	'P2':float(overprov_alert+underprov_alert)/len(sliceLoaded) }


if __name__ == '__main__':
	if len(sys.argv) < 2:
		raise TypeError('Usage: "./get_p1_p2.py <location of the usage slice as list> <vUp (integer)> <exponential smoothing window size (integer)>"')
	with open(str(sys.argv[1]), 'rb') as sliceFile:
		sliceLoaded = [float(element) for element in sliceFile.readlines()]
	vUp = int(sys.argv[2]) if len(sys.argv)>=3 else 250
	window_size = int(sys.argv[3]) if len(sys.argv)==4 else 250
	mva = movingaverage(sliceLoaded,window_size)
	outliers = get_outliers(sliceLoaded, mva)
	alerts = get_alerts(vUp, sliceLoaded)
	print('P0:%r\nP1:%r\nP2:%r' %(outliers['P0'],outliers['P1'],alerts['P2']))
	print(get_alerts(vUp,sliceLoaded))
	plt.plot(sliceLoaded[:len(mva)],'.')
	plt.plot(mva,'r')
	plt.show()