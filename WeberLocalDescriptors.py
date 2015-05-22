import numpy as np 
import matplotlib.pyplot as plt

from scipy import misc
from scipy.ndimage import filters

from math import pi


image1 = misc.imread('images.bmp')


def mask1(x):
	return (x.sum() - 9*x[4]) / x[4]

def mask2(x):
	v11 = (x[3] - x[5])
	v10 = (x[7] - x[1])
	s =  np.arctan2(v11, v10) + pi
	return s

diffExcit = np.arctan(filters.generic_filter(image1, mask1, 3))

#print diffExcit.shape

theta = np.arctan(filters.generic_filter(image1,mask2,3))
T = 8 
M = 6 
S = 20 
phi = 2 * ( ((theta*T)/(2*pi) + 1/2) % T) * pi /T       
        

H_t = []
for i in range(T):
    p = 2*i*pi/T
    indicies = np.where(phi == p)
    values = theta[indicies[0], indicies[1]]
    H_t.append(np.histogram(values))
       

length = H_t[0][0].len()
H_tm = []
for i in range(T):
	h = []
	for j in range(M):
        h.append(H_t[i][0][j*M/length:(j+1)*M/length], H_t[i][1][j*M/length:(j+1)*M/length])
    H_tm.append(h)

H_m = []
for i in range(M):
	m = []
	for j in range(T):
		m.append(H_tm[j][i])
	H_m.append(m)

H = []
for i in range(M):
	l = H_m[i]


for i in range(M):
       for j in range(T):
               H_m[i] = H_t[j]


#plt.hist(diffExcit)
#plt.show()


#print diffExcit.shape
#print diffExcit 

#plt.imshow(image1)
#plt.show()
