import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import log_bin as lb
import time
import json
import scipy as sp
import scipy.stats as stats
import sys
import matplotlib.lines as mlines
from networks import *

input = sys.argv[1]
if input == '1':
	##Showing different G0s:
	# #m=1
	# G=nx.complete_graph(3)
	# nx.draw_circular(G)
	# plt.title('$\mathcal{G}_0$ for m=1')
	# plt.show()
	# #m=2
	# G=nx.complete_graph(5)
	# nx.draw_circular(G)
	# plt.title('$\mathcal{G}_0$ for m=2')
	# plt.show()
	# #m=5
	# G=nx.complete_graph(7)
	# nx.draw_circular(G)
	# plt.title('$\mathcal{G}_0$ for m=3')
	# plt.show()
	
	# d,e= GenGraph(10, 3, repeat=True, draw=True, G0=False)
	# d,e= GenGraph(10, 3, repeat=False, draw=True, G0=False)
	

	for m in range(1,4):
		degreeL = LoadData(10000,m, 1)
		degree=degreeL[0]
		plt.plot(degree, zorder=4-m)
	blue_line = mlines.Line2D([], [], color='C0', label='m=1')
	orange_line = mlines.Line2D([], [], color='C1', label='m=2')
	green_line = mlines.Line2D([], [], color='C2', label='m=3')
	plt.legend(handles=[blue_line, orange_line, green_line])
	plt.title('Degree Vs. Node number')
	plt.xlabel('Node Number')
	plt.ylabel('Degree')
	plt.show()

if input == '2':
	for m in range(3,4):
		degreeL= LoadData(10000,m,100)
		degree=degreeL[0]
		deg, freq= lb.frequency(degree)
		norm=float(sum(freq))
		print norm
		prob= freq/norm
		A= 2*m*(m+1)
		fit1= lambda x: A*x**-3 
		fit2= lambda x: A/(x*(x+1)*(x+2))  
		fit1a = fit1(deg)
		fit2a = fit2(deg) 
		
		plt.loglog(deg,prob, 'k+', zorder=3)
		plt.loglog(deg, fit1a, 'r--',zorder=1)
		plt.loglog(deg, fit2a, 'b-' ,zorder=2)
		plt.show()

		c, e, bins=lb.log_bin(degree, bin_start=m-0.39, first_bin_width=1, a=1.25)
	
		boundaries=[]
		for i in range(len(bins)-1):
			singlebin=np.array([bins[i], bins[i+1]]) 
			boundaries.append(singlebin)
		boundaries=np.log(np.array(boundaries)).T
		# error=np.log(width)
		d=np.array(c)
		fit1b=fit1(d)
		fit2b=fit2(d)
		# plt.plot(/c,e, 'gs', markerfacecolor='None', zorder=4)
		plt.plot(np.log(deg), np.log(fit1a), 'r--',zorder=1)
		plt.plot(np.log(deg), np.log(fit2a), 'b-' ,zorder=2)
		plt.errorbar(np.log(c), np.log(e), yerr=boundaries, fmt='ks',elinewidth='.5', ecolor='r')
		plt.show()
		
		# c, e=lb.log_bin(degree, bin_start=float(m) ,a=1.7)
		ks1, pvalks1 =stats.ks_2samp(e, fit1b)
		ks2, pvalks2 =stats.ks_2samp(e, fit2b)
		# ks1, pvalks1 =stats.kstest(fit1(d), fit1, N=10000)
		# ks2, pvalks2 =stats.kstest(e, fit2, N=10000)
		print ks1, pvalks1
		print ks2, pvalks2
	# s, i, r_value, p_value, std_err = scipy.
		chi1, pvalchi1= stats.chisquare(e, f_exp=fit1b, ddof=2)
		chi2, pvalchi2= stats.chisquare(e, f_exp=fit2b, ddof=2)
		print chi1, chi2
		print pvalchi1, pvalchi2
if input == '3':
	k1=[]
	Npower= np.array(range(2,6))
	for i in Npower:
		degreeL= LoadData(10**i,1, 1 )
		degree=degreeL[0]
		kmax=max(degree)
		k1.append(kmax)
		N=10**Npower
	plt.plot(N, k1, 'ro')
	plt.show()

if input == '4': 
	m=3
	col=['rx', 'bx', 'gx']
	for i in range(2,5):
		degreeL= LoadData(10**i,m,100)
		degree=degreeL[0]
		deg, freq= lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm
		k1=max(deg)
		# plt.loglog(deg, prob)
		x= np.linspace(0, 550,500)
		A= 2*m*(m+1) 
		fit3= lambda x: A/(x*(x+1)*(x+2))
		deg, prob, bins=lb.log_bin(degree, bin_start=m-0.39, first_bin_width=1. ,a=1.25)
		deg= np.array(deg)
		fit3a = fit3(deg)
		collapse= prob/fit3a
		plt.figure(1)
		plt.loglog(x, fit3(x), 'k--' ,lw=0.5)
		plt.loglog(deg, prob, col[i-2])
		plt.figure(2)
		plt.loglog(deg, collapse, col[i-2])
		plt.loglog(x, fit3(x)/fit3(x), 'k--')
		plt.figure(3)
		plt.loglog(x/(10**i)**.5, fit3(x)/fit3(x), 'k--')
		plt.loglog(deg/(10**i)**.5, collapse, '.')
	# axes = plt.gca()
	# axes.set_xlim([0,4])
	# axes.set_ylim([0,4])
	plt.show()
if input =='5':
	N=10000
	m=3
	degree, e =RanGraph(N,m)
	# degree=np.array(degreeL[0])
	deg, freq= lb.frequency(degree)
	norm=float(sum(freq))
	prob= freq/norm
	plt.loglog(deg, prob)
	fit =lambda k: 1.2*np.exp(-k/3)
	fit1= fit(deg)
	plt.loglog(deg, fit1)
	plt.show()