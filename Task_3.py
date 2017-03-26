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
	N=1000
	col=['rx', 'bx', 'gx', 'mx']
	for m in range(1,5):

		degree, nei=WalkGraph1(N,m, 0, 100)
		degree=degree.flatten()
		deg, freq= lb.frequency(degree)
		x=np.linspace(1,max(deg),100)
		norm=float(sum(freq))
		prob= freq/norm
		fit1 =lambda k: m**(k-m)/(m+1)**(k-m+1)
		plt.loglog(deg, prob, col[m-1],zorder=2)
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
	for m in [10]:
		degree, nei=WalkGraph2(N,m,1,100)
		degree1, nei1=WalkGraph1(N,m, 1,100)
		degree=degree.flatten()
		degree1=degree1.flatten()
		deg,freq=lb.frequency(degree)
		deg1, freq1=lb.frequency(degree1)
		norm=float(sum(freq))
		prob= freq/norm
		prob1= freq1/float(sum(freq1))
		plt.loglog(deg, prob, 'x', zorder=2)
		plt.loglog(deg1, prob1, 'x', zorder=1)
	plt.show()		

if input =='3':
	N=10000
	col=['rx', 'bx', 'gx', 'mx']
	for m in [1,2,10]:
		degree=WalkGraph2(N,m,1,100)
		degree=degree.flatten()
		deg,freq=lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm
		plt.loglog(deg, prob, 'x', zorder=2)
	blue = mlines.Line2D([], [], color='C0', linestyle=' ', marker='x', label='m=1')
	orange = mlines.Line2D([], [], color='C1', linestyle=' ', marker='x',label='m=2')
	green = mlines.Line2D([], [], color='C2', linestyle=' ', marker='x',label='m=10')
	plt.legend(handles=[blue, orange ,green])
	plt.xlabel('$k$')
	plt.ylabel('$p(k)$')
	plt.show()		

if input =='4':
	N=10000
	col=['rx', 'bx', 'gx', 'mx']
	for m in [1,2,10]:
		degree=MDA(N,m,100)
		degree=degree.flatten()
		deg,freq=lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm
		plt.loglog(deg, prob, 'x', zorder=2)
	blue = mlines.Line2D([], [], color='C0', linestyle=' ', marker='x', label='m=1')
	orange = mlines.Line2D([], [], color='C1', linestyle=' ', marker='x',label='m=2')
	green = mlines.Line2D([], [], color='C2', linestyle=' ', marker='x',label='m=10')
	plt.legend(handles=[blue, orange ,green])
	plt.xlabel('$k$')
	plt.ylabel('$p(k)$')
	plt.show()		


if input == '5':
	N=10000
	col=['rx', 'bx', 'gx', 'mx', 'rx' , 'bx']
	l=[2,4,100]
	for m in range(1,2):
		for i in range(len(l)):
			degree, nei=WalkGraph2(N,m,l[i],100)
			degree=degree.flatten()
			deg,freq=lb.frequency(degree)
			norm=float(sum(freq))
			prob= freq/norm
			plt.loglog(deg, prob, 'x',zorder=2)
			
			A= 2*m*(m+1)
			y=np.linspace(1, max(deg), 1000)
			fit2= lambda x: A/(x*(x+1)*(x+2))  
			plt.figure(1)
			plt.loglog(y, fit2(y), 'k--' ,zorder=1)
		blue = mlines.Line2D([], [], color='C0', linestyle=' ', marker='x', label='L=2')
		orange = mlines.Line2D([], [], color='C1', linestyle=' ', marker='x',label='L=4')
		green = mlines.Line2D([], [], color='C2', linestyle=' ', marker='x',label='L=100')
		black = mlines.Line2D([], [], color='k', linestyle='--',label='Theoretical Fit')
		plt.legend(handles=[blue, orange ,green, black])
		plt.xlabel('$k$')
		plt.ylabel('$p(k)$')
		plt.show()		


if input == '6':
	N=[1000, 10000, 100000]
	col=['rx', 'bx', 'gx', 'mx', 'rx' , 'bx']
	l=[7, 9, 11]
	for m in range(1,2):
		for i in range(len(l)):
			degree=WalkGraph2(N[i],m,l[i],100)
			degree=degree.flatten()
			deg,freq=lb.frequency(degree)
			norm=float(sum(freq))
			prob= freq/norm
			plt.loglog(deg, prob, col[i],zorder=5-i)
			A= m
			y=np.linspace(1, max(deg), 1000)
			fit2= lambda x: A/(x**2.5)  
			plt.loglog(y, fit2(y), 'k--' ,zorder=1)
			plt.figure(1)
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x', label='$N=10^3$')
		blue = mlines.Line2D([], [], color='b', linestyle=' ', marker='x',label='$N=10^4$')
		green = mlines.Line2D([], [], color='C2', linestyle=' ', marker='x',label='$N=10^5$')
		black = mlines.Line2D([], [], color='k', linestyle='--',label='Possible Power Law Fit')
		plt.legend(handles=[red, blue ,green, black])
		plt.xlabel('$k$')
		plt.ylabel('$p(k)$')
		plt.show()

