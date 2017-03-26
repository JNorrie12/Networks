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

##########Implementation and Testing###############
if input == '0':
	#Figure 1, Showing different G0'a
	#m=1
	G=nx.complete_graph(3)
	nx.draw_circular(G)
	plt.title('$\mathcal{G}_0$ for m=1')
	plt.show()
	#m=2
	G=nx.complete_graph(5)
	nx.draw_circular(G)
	plt.title('$\mathcal{G}_0$ for m=2')
	plt.show()
	#m=5
	G=nx.complete_graph(7)
	nx.draw_circular(G)
	plt.title('$\mathcal{G}_0$ for m=3')
	plt.show()
	
	#Figure 2, comparision of double edges vs. no double egdes.
	degree1=GenGraph3(1000, 10, 1) #No Double edges
	degree2=GenGraph4(1000, 10, 1) #Double edges
	degree1=degree1.flatten()
	degree2=degree2.flatten()
	deg1, freq1= lb.frequency(degree1)
	deg2, freq2= lb.frequency(degree2)
	norm1=float(sum(freq1))
	norm2=float(sum(freq2))

	prob1= freq1/norm1
	prob2= freq2/norm2

	plt.loglog(deg1, prob1, 'x')
	plt.loglog(deg2,prob2, 'x')
	plt.xlabel('$k$')
	plt.ylabel('$p(k)$')
	green = mlines.Line2D([], [], color='C0', linestyle=' ', marker='x',label='Double Edges Allowed')
	purple = mlines.Line2D([], [], color='C1', linestyle=' ', marker='x',label='Double Edges Not Allowed')
	plt.legend(handles=[green, purple])	

	plt.show()

	# #Figure 3 testing power law distribution.
	# for i in [1,1000]:
	# 	for m in range(1,5):
	# 		degreeL = LoadData(10000,m, i)
	# 		degree=degreeL[0]
	# 		plt.plot(degree, zorder=4-m)
	# 	blue_line = mlines.Line2D([], [], color='C0', label='m=1')
	# 	orange_line = mlines.Line2D([], [], color='C1', label='m=2')
	# 	green_line = mlines.Line2D([], [], color='C2', label='m=3')
	# 	plt.legend(handles=[blue_line, orange_line, green_line])
	# 	plt.title('Degree Vs. Node number')
	# 	plt.xlabel('Node Number')
	# 	plt.ylabel('Degree')
	# 	plt.show()
	

if input == '1':
	#Figure 3 testing power law distribution.
	for i in [1,1000]:
		for m in range(1,5):
			degreeL= LoadData(10000,m,i)
			degree=degreeL[0]
			deg, freq= lb.frequency(degree)
			norm=float(sum(freq))
			prob= freq/norm
			A= 2*m*(m+1) 
			y=np.linspace(1, max(deg), 1000)
			fit2= lambda x: A/(x**3) 
			plt.figure(1)
			plt.loglog(deg,prob, '.', zorder=2)
			if m==4:
				plt.loglog(y, fit2(y), 'k--', lw=.5 ,zorder=1)
			blue = mlines.Line2D([], [], color='C0', linestyle=' ', marker='.' ,label='m=1')
			orange = mlines.Line2D([], [], color='C1', linestyle=' ', marker='.', label='m=2')
			green = mlines.Line2D([], [], color='C2', linestyle=' ', marker='.', label='m=3')
			red = mlines.Line2D([], [], color='C3', linestyle=' ', marker='.', label='m=4')
		 	black = mlines.Line2D([], [], color='k', linestyle='--', label='Fit, $f(k) \propto k^{-\gamma}$')
			plt.legend(handles=[blue, orange, green, red, black])	
			plt.xlabel('Avalanche Size')
			plt.ylabel('Probability')

		plt.show()

		
if input == '2':
	#Figure 4 theoretical fits
	for m in range(4,5):
		degreeL= LoadData(100000,m,1000)
		degree=degreeL[0]
		deg, freq= lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm
		A= 2*m*(m+1)
		fit1= lambda x: A*x**-3 
		fit2= lambda x: A/(x*(x+1)*(x+2))  
		plt.figure(m)
		plt.loglog(deg,prob, 'k+', zorder=1)
		plt.loglog(deg, fit1(deg), 'r--',zorder=2)
		plt.loglog(deg, fit2(deg), 'b-' ,zorder=2)
		blue = mlines.Line2D([], [], color='b', linestyle='-', label='Discrete Theoretical Fit')
		red = mlines.Line2D([], [], color='r', linestyle='--', label='Continuous Theoretical Fit')
		black = mlines.Line2D([], [], color='k', linestyle=' ', marker='+', label='Data')
		plt.legend(handles=[blue, red, black])	
		plt.xlabel('Avalanche Size')
		plt.ylabel('Probability')
		plt.show()


		d, e, bins=lb.log_bin(degree, bin_start=m-0.39, first_bin_width=1, a=1.25) 
		c=np.array(d)
		

	#Figure 6 Linear regression graph
		if m==4:
			x = np.linspace(0, max(e), 5)
			y=np.linspace(0, max(fit1(c)), 5)
			plt.plot(y, 0.5719*y, color='darkred', linestyle='--', lw=.5, zorder=1)
			plt.plot(x, 1.0312*x, color='darkblue', linestyle='--', lw=.5, zorder=1)
			plt.plot(fit1(c), e ,'r+', zorder=2)
			plt.plot(fit2(c), e,'b+', zorder=2)
			plt.plot() 	
			red = mlines.Line2D([], [], color='r', linestyle=' ', marker='+', label='Continuous Theoretical Fit')
			blue = mlines.Line2D([], [], color='b', linestyle=' ', marker='+', label='Discrete Theoretical Fit')
			redline = mlines.Line2D([], [], color='darkred', linestyle='--', lw=.5, label='Regression Line(Continuous)')
			blueline = mlines.Line2D([], [], color='darkblue', linestyle='--', lw=.5,label='Regression Line(Discrete)')
			
			plt.legend(handles=[blue, red, blueline, redline])
			plt.xlabel('Theoretical Data')
			plt.ylabel('Actual Data')
			plt.show()

	#Statistics as quoted in figure 6.
		print stats.linregress(fit1(c), e)
		print stats.linregress(fit2(c), e)

if input == '3':
	#Figure 5 log-bins
	col=['r-', 'b-', 'g-', 'm-']
	for m in range(1,5):
		degreeL= LoadData(100000,m,1000)
		degree=degreeL[0]
		deg, freq= lb.frequency(degree)
		norm=float(sum(freq))
		prob= freq/norm
		A= 2*m*(m+1)
		
		d, e, bins=lb.log_bin(degree, bin_start=m-0.39, first_bin_width=1, a=1.25) 
		c=np.array(d)
		
		fit1= lambda x: A*x**-3 
		fit2= lambda x: A/(x*(x+1)*(x+2))  
		
		plt.loglog(c , e, col[m-1], zorder=4)
		plt.loglog(deg, fit1(deg), 'y--', lw=.5 ,zorder=1)
		plt.loglog(deg, fit2(deg), 'k-', lw=.5 ,zorder=2)

		black = mlines.Line2D([], [], color='k', linestyle='-',lw=.5,label='Discrete Theoretical Fit for Respective $m$')
		yellow = mlines.Line2D([], [], color='k', linestyle='--',lw=.5,label='Continuous Respective $m$')
		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x',label='m=1')
		blue = mlines.Line2D([], [], color='b', linestyle=' ', marker='x', label='m=2')
		green = mlines.Line2D([], [], color='g', linestyle=' ', marker='x',label='m=3')
		purple = mlines.Line2D([], [], color='m', linestyle=' ', marker='x',label='m=4')
		plt.legend(handles=[black, yellow ,red, blue, green, purple])	

		plt.xlabel('Theoretical Data')
		plt.ylabel('Actual Data')
	plt.show()

if input == '4':
	#Figure 7, Largest degree 
	k1=[]
	error=[]
	Npower= np.array(range(2,6))
	for i in Npower:
		m=4
		degree =GenGraph3(10**i, m, 100)
		kmax=np.array([max(j) for j in degree])
		k1.append(np.mean(kmax))
		error.append(np.std(kmax))
	error=np.array(error).T
	N=10**Npower
	coeff =np.polynomial.polynomial.polyfit(k1, N,[2] )

	coeff2=1/(coeff[-1])**.5
	x=np.linspace(0,100000,10000)
	print coeff2
	# y=np.sqrt(m*(m+1))
	fit= 0.5*(-1+np.sqrt(1+4*x*m*(m+1)))
	plt.plot(x, fit ,'r--', zorder=1)
	# plt.plot(N, k1, 'ro', zorder=2)
	plt.errorbar(N, k1 ,yerr=error, barsabove=True ,linestyle=' ', marker='o', color='k', elinewidth='0.5')
	red = mlines.Line2D([], [], color='r', linestyle='--', label='Fit, '+ str(round(coeff2,1))+ '$x^0.5$')
	black = mlines.Line2D([], [], color='k', linestyle=' ', marker='o', label='Data')
	# plt.legend(handles=[black, red])
	plt.xlabel('Number of Nodes in Network ($N$)')
	plt.ylabel('Largest Degree Size ($k_1$)')
	plt.show()

	figK1, ax1 = plt.subplots()
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.loglog(x, fit, 'r--', zorder=1)
	ax1.loglog(N, k1, 'ks')
	ax1.errorbar(N, k1 ,yerr=error, barsabove=True ,linestyle=' ', marker='o', color='k', elinewidth='0.5')
	figK1.xlabel('Number of Nodes in Network ($N$)')
	figK1.ylabel('Largest Degree Size ($k_1$)')
	figK1.show()

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
# if input =='5':
# 	N=10000
# 	col=['rx', 'bx', 'gx', 'mx']
# 	for m in range(1,5):
# 		degree=RanGraph1(N,m, 100)
# 		degree=degree.flatten()
# 		deg, freq= lb.frequency(degree)
# 		x=np.linspace(1,max(deg),100)
# 		norm=float(sum(freq))
# 		prob= freq/norm
# 		fit1 =lambda k: m**(k-m)/(m+1)**(k-m+1)
# 		plt.loglog(deg, prob, col[m-1], zorder=2)
# 		plt.loglog(x, fit1(x), 'k-', lw=.5,zorder=1)
# 		plt.xlabel('$k$')
# 		plt.ylabel('$p(k)$')
# 		black = mlines.Line2D([], [], color='k', linestyle='-',lw=.5,label='Theoretical Fit for Respective $m$')
# 		red = mlines.Line2D([], [], color='r', linestyle=' ', marker='x',label='N=$10^2$')
# 		blue = mlines.Line2D([], [], color='b', linestyle=' ', marker='x', label='N=$10^3$')
# 		green = mlines.Line2D([], [], color='g', linestyle=' ', marker='x',label='N=$10^4$')
# 		purple = mlines.Line2D([], [], color='m', linestyle=' ', marker='x',label='N=$10^5$')
# 		plt.legend(handles=[black, red, blue, green, purple])	
# 	plt.show()
# 	