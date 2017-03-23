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
	for m in range(4,5):
		degreeL= LoadData(100000,m,1000)
		degree=degreeL[0]
		deg, freq= lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm
		A= 2*m*(m+1)
		fit1= lambda x: A*x**-3 
		fit2= lambda x: A/(x*(x+1)*(x+2))  
		fit1a = fit1(deg)
		fit2a = fit2(deg) 
	# 	plt.figure(m)
	# 	plt.loglog(deg,prob, 'k+', zorder=1)
	# 	plt.loglog(deg, fit1a, 'r--',zorder=2)
	# 	plt.loglog(deg, fit2a, 'b-' ,zorder=2)
	# 	blue = mlines.Line2D([], [], color='b', linestyle='-', label='Discrete Theoretical Fit')
	# 	red = mlines.Line2D([], [], color='r', linestyle='--', label='Continuous Theoretical Fit')
	# 	black = mlines.Line2D([], [], color='k', linestyle=' ', marker='+', label='Data')
	# 	plt.legend(handles=[blue, red, black])	
	# 	plt.xlabel('Avalanche Size')
	# 	plt.ylabel('Probability')

	# plt.show()
		###R^2 TEST######
# if input == 'a':
		d, e, bins=lb.log_bin(degree, bin_start=m-0.39, first_bin_width=1, a=1.25) 
		c=np.array(d)
		plt.plot(fit1(c), e ,'r+')
		plt.plot(fit2(c), e,'b+') 	
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='+', label='Continuous Theoretical Fit')
		blue = mlines.Line2D([], [], color='b', linestyle=' ', marker='+', label='Discrete Theoretical Fit')
		plt.legend(handles=[blue, red])
		plt.show()
		# print stats.linregress(prob,fit1a)
		# print stats.linregress(prob, fit2a)
		# print stats.linregress(e, fit1(c))
		# print stats.linregress(e, fit2(c))
		# boundaries=[]
		# for i in range(len(bins)-1):
		# 	singlebin=np.array([bins[i], bins[i+1]]) 
		# 	boundaries.append(singlebin)
		# boundaries=np.log(np.array(boundaries)).T
		# # error=np.log(width)
		
		plt.loglog(c,e, 'ks', markerfacecolor='None', zorder=4)
		plt.loglog(deg, fit1a, 'r--',zorder=1)
		plt.plot(deg, fit2a, 'b-' ,zorder=2)
		# plt.errorbar(np.log(c), np.log(e), yerr=boundaries, fmt='ks',elinewidth='.5', ecolor='r')
		# plt.show()
		
		# ks1, pvalks1 =stats.ks_2samp(prob, fit1a)
		# ks2, pvalks2 =stats.ks_2samp(prob, fit2a)
		# print ks1, pvalks1
		# print ks2, pvalks2
		# chi1, pvalchi1= stats.chisquare(prob, f_exp=fit1a, ddof=2)
		# chi2, pvalchi2= stats.chisquare(prob, f_exp=fit2a, ddof=2)
		# print chi1, chi2
		# print pvalchi1, pvalchi2
		# mwu1, pvalmwu1= stats.mannwhitneyu(prob, fit1a)
		# mwu2, pvalmwu2= stats.mannwhitneyu(prob, fit2a)	
		# print mwu1, mwu2
		# print pvalmwu1, pvalmwu2
	plt.show()
if input == '3':
	k1=[]

	error=[]
	Npower= np.array(range(2,6))
	for i in Npower:
		# degreeL= LoadData(10**i,1, 1000 )
		degree =GenGraph3(10**i, 1, 100)
		kmax=np.array([max(j) for j in degree[0]])
		# degree=degreeL[0]
		# print len(degree)
		# breakup= [degree[i:i + 1000] for i in xrange(0, len(degree), 1000)]
		# kmax=np.array([max(i) for i in breakup])
		# print kmax
		k1.append(np.median(kmax))
		error.append([min(kmax), max(kmax)])
		# plt.show()
		# kmax=max(degree)
	error=np.array(error).T
	N=10**Npower
	coeff =np.polynomial.polynomial.polyfit(k1, N,[2] )
	print coeff
	coeff2=1/(coeff[-1])**.5
	x=np.linspace(0,100000,10000)
	print coeff2
	fit= coeff2*x**0.5
	plt.plot(x, fit ,'r--', zorder=1)
	# plt.plot(N, k1, 'ro', zorder=2)
	plt.errorbar(N, k1 ,yerr=3*error**0.5, barsabove=True ,linestyle=' ', marker='o', color='k', elinewidth='0.5')
	red = mlines.Line2D([], [], color='r', linestyle='--', label='Fit, '+ str(round(coeff2,1))+ '$x^0.5$')
	black = mlines.Line2D([], [], color='k', linestyle=' ', marker='o', label='Data')
	# plt.legend(handles=[black, red])
	plt.xlabel('Number of Nodes in Network ($N$)')
	plt.ylabel('Largest Degree Size ($k_1$)')
	plt.show()

if input == '4': 
	m=4
	col=['rx', 'bx', 'gx', 'mx']
	for i in range(2,6):
		degreeL= LoadData(10**i,m,1000)
		degree=degreeL[0]
		deg, freq= lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm

		x= np.linspace(0, 550,500)
		A= 2*m*(m+1) 
		fit3= lambda x: A/(x*(x+1)*(x+2))
		# deg2, prob2, bins=lb.log_bin(degree, bin_start=m-0.39, first_bin_width=1. ,a=1.2)
		# deg2= np.array(deg2)
		collapse= prob/fit3(deg)
		plt.figure(1)
		plt.loglog(x, fit3(x), 'k--' ,lw=0.5)
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
		plt.loglog(x, fit3(x)/fit3(x), 'k--')
		black = mlines.Line2D([], [], color='k', linestyle='--', label='Discrete Theoretical Fit')
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x' ,label='N=$10^2$')
		blue = mlines.Line2D([], [], color='b', linestyle='', marker='x', label='N=$10^3$')
		green = mlines.Line2D([], [], color='g', linestyle='', marker='x', label='N=$10^4$')
		purple = mlines.Line2D([], [], color='m', linestyle='', marker='x', label='N=$10^5$')
		plt.legend(handles=[black, red, blue, green, purple])
		plt.xlabel('$k$')
		plt.ylabel('$p_{data}(k)/p_{theory}(k)$')
		
		plt.figure(3)
		plt.loglog(x/(10**i)**.5, fit3(x)/fit3(x), 'k--')
		plt.loglog(deg/(10**i)**.5, collapse, col[i-2], zorder=6-i)
		black = mlines.Line2D([], [], color='k', linestyle='--', label='Discrete Theoretical Fit')
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x' ,label='N=$10^2$')
		blue = mlines.Line2D([], [], color='b', linestyle='', marker='x', label='N=$10^3$')
		green = mlines.Line2D([], [], color='g', linestyle='', marker='x', label='N=$10^4$')
		purple = mlines.Line2D([], [], color='m', linestyle='', marker='x', label='N=$10^5$')
		plt.legend(handles=[black, red, blue, green, purple])
		plt.xlabel('$k/{\sqrt{N}}$')
		plt.ylabel('$p_{data}(k)/p_{theory}(k)$')	
	

	plt.show()
if input =='5':
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
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x',label='N=$10^2$')
		blue = mlines.Line2D([], [], color='b', linestyle=' ', marker='x', label='N=$10^3$')
		green = mlines.Line2D([], [], color='g', linestyle=' ', marker='x',label='N=$10^4$')
		purple = mlines.Line2D([], [], color='m', linestyle=' ', marker='x',label='N=$10^5$')
		plt.legend(handles=[black, red, blue, green, purple])	
	plt.show()