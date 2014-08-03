from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

readList = []
with open('good_slices/raw_CPU_usg_slice_1294851_avg_14.006923076923895_std_20.996635513643344', 'r') as rd:
	tempRd = rd.read()
	for i in xrange(10000):
		readList.append(float(tempRd.split("\n")[i]))
readList = np.asarray(readList)
grid = np.linspace(0,100,10000)
kde = gaussian_kde(readList)
print kde.evaluate(grid).shape, readList.shape
with open('good_slices/kde_.txt','w') as kdeOut:
	for line in kde.evaluate(grid).tolist():
		kdeOut.write(`line`+"\n")
plt.plot(grid,kde.evaluate(grid))
plt.show()