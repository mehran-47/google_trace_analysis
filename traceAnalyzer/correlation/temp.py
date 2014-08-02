import random as rand
from pydoc import help
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
"""
pearsonr(x, y)
    Calculates a Pearson correlation coefficient and the p-value for testing
    non-correlation.
    
    The Pearson correlation coefficient measures the linear relationship
    between two datasets. Strictly speaking, Pearson's correlation requires
    that each dataset be normally distributed. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear
    relationship. Positive correlations imply that as x increases, so does
    y. Negative correlations imply that as x increases, y decreases.
    
    The p-value roughly indicates the probability of an uncorrelated system
    producing datasets that have a Pearson correlation at least as extreme
    as the one computed from these datasets. The p-values are not entirely
    reliable but are probably reasonable for datasets larger than 500 or so.
    
    Parameters
    ----------
    x : 1D array
    y : 1D array the same length as x
    
    Returns
    -------
    (Pearson's correlation coefficient,
     2-tailed p-value)
    
    References
    ----------
    http://www.statsoft.com/textbook/glosp.html#Pearson%20Correlation
"""

x1 = []
x2 = []
x3 = []
x4 = []
"""
for i in range(200):
	x1.append(rand.uniform(0,99))
	x2.append(rand.uniform(0,99))
	if i < 90:
		x3.append(rand.uniform(0,99))
	else:
		x3.append(x1[i])
	x4.append(x2[i]**1.5)

print "similarity between x1 and x2:", pearsonr(x1,x2)[0]*100, "%"
print "similarity between x1 and x3:", pearsonr(x1,x3)[0]*100, "%"
print "similarity between x2 and x4:", pearsonr(x2,x4)[0]*100, "%"
plt.figure(1)
plt.subplot(221)
plt.plot(x1,)
plt.subplot(222)
plt.plot(x2)
plt.subplot(223)
plt.plot(x3)
plt.subplot(224)
plt.plot(x4)
plt.show()
"""
x1= np.linspace(-10,10,1000)
y1= np.sin(x1)
y2= (np.sin(x1))**2
y3 = np.cos(x1)

print "similarity between y1= sin(x1) and y2=sin(x)^2:", pearsonr(y1,y2)[0]*100, "% , p-value: ", pearsonr(y1,y2)[1]
print "similarity between y1= sin(x1) and y3=cos(x1):", pearsonr(y1,y2)[0]*100, "% , p-value: ", pearsonr(y1,y2)[1]
plt.figure(1)
plt.subplot(311)
plt.plot(y1)
plt.subplot(312)
plt.plot(y2)
plt.subplot(313)
plt.plot(y3)
plt.show()


"""

"""
