import matplotlib.pyplot as plt
import numpy as np
import log_bin as lb
import time
import json
import sys
import matplotlib.lines as mlines
from networks import *
from scipy.optimize import curve_fit
input = sys.argv[1]
if input =='1':
	N=10000
	col=['rx', 'bx', 'gx', 'mx']
	for m in range(1,5):

		degree, nei=WalkGraph1(N,m, 0, 100)
		degree=degree.flatten()
		deg, freq= lb.frequency(degree)
		x=np.linspace(1,max(deg),100)
		norm=float(sum(freq))
		prob= freq/norm
		fit1 =lambda k: m**(k-m)/(m+1)**(k-m+1)
		plt.loglog(deg, prob, col[m-1], zorder=2)
		plt.loglog(x, fit1(x), 'k-', lw=.5,zorder=1)
		plt.xlabel('$k$')
		plt.ylabel('$p(k)$')
		black = mlines.Line2D([], [], color='k', linestyle='-',lw=.5,label='Theoretical Fit for Respective $m$')
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x',label='m=1')
		blue = mlines.Line2D([], [], color='b', linestyle=' ', marker='x', label='m=2')
		green = mlines.Line2D([], [], color='g', linestyle=' ', marker='x',label='m=3')
		purple = mlines.Line2D([], [], color='m', linestyle=' ', marker='x',label='m=4')
		plt.legend(handles=[black, red, blue, green, purple])	
	plt.show()

if input =='2':
	N=10000
	col=['rx', 'bx', 'gx', 'mx']
	for m in range(1,5):
		degree, nei=WalkGraph1(N,m,1,100)
		degree=degree.flatten()
		deg,freq=lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm
		plt.loglog(deg, prob, col[m-1], zorder=2)
		plt.show()		

if input == '3':
	N=10000
	# l= 100
	# l= int(np.log(N)/(np.log(np.log(N))))
	col=['rx', 'bx', 'gx', 'mx']
	for m in range(4,5):
		for l in [1,10, 50]:
			degree, nei=WalkGraph1(N,m,l,100)
			degree=degree.flatten()
			deg,freq=lb.frequency(degree)
			norm=float(sum(freq))
			prob= freq/norm
			plt.loglog(deg, prob, col[m-1], zorder=2)
		plt.show()		
