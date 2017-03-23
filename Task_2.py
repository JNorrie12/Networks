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
		degree=RanGraph1(N,m, 100)
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

if input == '2':
	k1=[]
	error=[]
	def fit(x,a):
		return a*np.log(x)
	Npower= np.array(range(2,6))
	for i in Npower:
		degree =RanGraph1(10**i, 4, 100)
		kmax=np.array([max(j) for j in degree])
		k1.append(np.mean(kmax))
	N=10**Npower
	
	fit= curve_fit(fit, N, k1)
	print fit[0][0]
	x=np.linspace(0, 100000, 10000)
	y= fit[0][0]*np.log(x)
	plt.plot(N, k1, 'ko', zorder=2)
	plt.plot(x, y, 'b--', zorder=1)
	blue = mlines.Line2D([], [], color='b', linestyle='--', label='Fit, '+ str(round(fit[0][0],1))+ 'log(x)')
	black = mlines.Line2D([], [], color='k', linestyle=' ', marker='o', label='Data')
	plt.legend(handles=[black, blue])
	plt.xlabel('Number of Nodes in Network ($N$)')
	plt.ylabel('Largest Degree Size ($k_1$)')
	plt.show()

if input == '3': 
	m=4
	col=['rx', 'bx', 'gx', 'mx']
	for i in range(2,4):
		degreeL= RanGraph1(10**i,m,100)
		degree=degreeL[0]
		deg, freq= lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm

		x= np.linspace(0, max(deg),500)

		fit =lambda k: m**(k-m)/(m+1)**(k-m+1)

		collapse= prob/fit(deg)
		plt.figure(1)
		plt.loglog(x, fit(x), 'k--' ,lw=0.5)
		# plt.loglog(deg2, prob2, col[i-2])
		plt.loglog(deg, prob, col[i-2], zorder=6-i)
		black = mlines.Line2D([], [], color='k', linestyle='--', label='Discrete Theoretical Fit')
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x' ,label='N=$10^2$')
		blue = mlines.Line2D([], [], color='b', linestyle='', marker='x', label='N=$10^3$')
		green = mlines.Line2D([], [], color='g', linestyle='', marker='x', label='N=$10^4$')
		purple = mlines.Line2D([], [], color='m', linestyle='', marker='x', label='N=$10^5$')
		plt.legend(handles=[black, red, blue, green, purple])
		plt.xlabel('Degree Sizes ($k$)')
		plt.ylabel('Probability ($p(k)$)')
		plt.figure(2)
		plt.loglog(deg, collapse, col[i-2], zorder=6-i)
		plt.loglog(x, fit(x)/fit(x), 'k--')
		black = mlines.Line2D([], [], color='k', linestyle='--', label='Discrete Theoretical Fit')
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x' ,label='N=$10^2$')
		blue = mlines.Line2D([], [], color='b', linestyle='', marker='x', label='N=$10^3$')
		green = mlines.Line2D([], [], color='g', linestyle='', marker='x', label='N=$10^4$')
		purple = mlines.Line2D([], [], color='m', linestyle='', marker='x', label='N=$10^5$')
		plt.legend(handles=[black, red, blue, green, purple])
		plt.xlabel('$k$')
		plt.ylabel('$p_{data}(k)/p_{theory}(k)$')
		
		plt.figure(3)
		plt.loglog(x/np.log(10**i), fit(x)/fit(x), 'k--')
		plt.loglog(deg/np.log(10**i), collapse, col[i-2], zorder=6-i)
		black = mlines.Line2D([], [], color='k', linestyle='--', label='Discrete Theoretical Fit')
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x' ,label='N=$10^2$')
		blue = mlines.Line2D([], [], color='b', linestyle='', marker='x', label='N=$10^3$')
		green = mlines.Line2D([], [], color='g', linestyle='', marker='x', label='N=$10^4$')
		purple = mlines.Line2D([], [], color='m', linestyle='', marker='x', label='N=$10^5$')
		plt.legend(handles=[black, red, blue, green, purple])
		plt.xlabel('$k/{\sqrt{N}}$')
		plt.ylabel('$p_{data}(k)/p_{theory}(k)$')	
	plt.show()