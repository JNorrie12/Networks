

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import log_bin as lb
import time
import json
import scipy as sp
import scipy.stats as stats
import sys
import random
from collections import Counter
#####################################################
def GenGraph3(N,m, T):
	t0=time.time()
	GTotal=[]
	degreeTotal=[]
	
	G0=[] #Making initial graph combinatorically 
	g=2*m+1
	for j in range(g-1): 
		for k in range(j+1,g):
			G0.extend([j,k])
	for t in range(T):
		G=G0[:] 					#Slicing to avoid change G0
		# print G0	
		for k in range(2*m+1, N):
			x=[]					
			while len(x)<m:			#No repeated edges
				value=random.choice(G)
				if value in x:
					pass
				else:
					x.append(value)
			for i in x:
				G.append(k)
				G.append(i)
		GTotal.append(G)
	for i in GTotal:
		degree = Counter(i) #returns a list of degrees for each sample separately
		degreeTotal.append(degree.values())
	print time.time()-t0
	return np.array(degreeTotal)


def GenGraph4(N,m, T):
	t0=time.time()
	GTotal=[]
	degreeTotal=[]
	
	G0=[] #Making initial graph combinatorically 
	g=2*m+1
	for j in range(g-1): 
		for k in range(j+1,g):
			G0.extend([j,k])
	for t in range(T):
		G=G0[:] 					#Slicing to avoid change G0
		# print G0	
		for k in range(2*m+1, N):
			x=[]					
			for i in range(m):			#No repeated edges
				value=random.choice(G)
				G.append(k)
				G.append(value)
		GTotal.append(G)
	for i in GTotal:
		degree = Counter(i) #returns a list of degrees for each sample separately
		degreeTotal.append(degree.values())
	print time.time()-t0
	return np.array(degreeTotal)


def RanGraph1(N,m, T):
	t0=time.time()
	GTotal=[]
	degreeTotal=[]
	
	G0=[] #Making initial graph combinatorically 
	g=2*m+1
	for j in range(g-1): 
		for k in range(j+1,g):
			G0.extend([j,k])
	for t in range(T):
		G=G0[:] 					#Slicing to avoid change G0	
		for k in range(2*m+1, N):
			x=[]					
			while len(x)<m:			#No repeated edges
				value=random.randint(0,k-1)
				if value in x:
					pass
				# elif value == k:
					# pass
				else:
					x.append(value)
			for i in x:
				G.append(k)
				G.append(i)
		GTotal.append(G)
	for i in GTotal:
		degree = Counter(i) #returns a list of degrees for each sample separately
		degreeTotal.append(degree.values())
	print time.time()-t0
	return np.array(degreeTotal)
# degree=list(np.array(GenGraph3(1000, 3, 100)).flatten())
def WalkGraph1(N, m, L, T):
	t0=time.time()
	GTotal=[]
	degreeTotal=[]
	
	G0=[] 
	g=2*m+1
	for j in range(g-1): 
		for k in range(j+1,g):
			G0.extend([j,k])
	
	Neighbourhood0=[[] for i in range(N)] #Neighbourhood of node i	
	
	for j in range(g):							#Initialisng complete graph in terms of neighbourhoods
		Neighbourhood0[j].extend(range(j)) 		#Adds all nodes less than j to neighbourhood
		Neighbourhood0[j].extend(range(j+1, g)) 	#Adds all nodes more than j to neighbourhood
	for t in range(T):
		G=G0[:] 
		Neighbourhood=[nei[:] for nei in Neighbourhood0]				#Slicing to avoid change G0	
		for k in range(2*m+1, N):	#2*m+1			
			x=[]
			while len(x)<m:			#No repeated edges
				value=random.randint(0,k-1)
				for i in range(L):
					value=random.choice(Neighbourhood[value]) #Randon walk through graph				
				if value in x:
					pass
				else:
					x.append(value)
			
			for i in x:
				G.append(k)
				G.append(i)
				Neighbourhood[i].append(k) #Adding the 2 nodes to eachother's neighbourhood
				Neighbourhood[k].append(i)
		GTotal.append(G)
	for i in GTotal:
		degree = Counter(i) #returns a list of degrees for each sample separately
		degreeTotal.append(degree.values())
	print time.time()-t0
	return np.array(degreeTotal), Neighbourhood

def WalkGraph2(N, m, L, T):
	t0=time.time()
	NeiTotal=[]
	Neighbourhood0=[[] for i in range(N)] #Neighbourhood of node i	
	degreeTotal=[]
	g=2*m+1
	for j in range(g):							#Initialisng complete graph in terms of neighbourhoods
		Neighbourhood0[j].extend(range(j)) 		#Adds all nodes less than j to neighbourhood
		Neighbourhood0[j].extend(range(j+1, g)) 	#Adds all nodes more than j to neighbourhood
	for t in range(T):

		Neighbourhood=[nei[:] for nei in Neighbourhood0]				#Slicing to avoid change G0	
		for k in range(g, N):	#2*m+1			
			x=[]
			while len(x)<m:			#No repeated edges
				value=random.randint(0,k-1)
				for i in range(L):
					value=random.choice(Neighbourhood[value]) #Randon walk through graph				
				if value in x:
					pass
				else:
					x.append(value)
			
			for i in x:
				Neighbourhood[i].append(k) #Adding the 2 nodes to eachother's neighbourhood
				Neighbourhood[k].append(i)
		NeiTotal.append(Neighbourhood)
	for i in NeiTotal:
		degree=[len(j) for j in i]
		degreeTotal.append(degree)
	print time.time()-t0
	return np.array(degreeTotal)

def MDA(N, m, T):
	t0=time.time()
	NeiTotal=[]
	Neighbourhood0=[[] for i in range(N)] #Neighbourhood of node i	
	degreeTotal=[]
	g=2*m+1
	for j in range(g):							#Initialisng complete graph in terms of neighbourhoods
		Neighbourhood0[j].extend(range(j)) 		#Adds all nodes less than j to neighbourhood
		Neighbourhood0[j].extend(range(j+1, g)) 	#Adds all nodes more than j to neighbourhood
	for t in range(T):

		Neighbourhood=[nei[:] for nei in Neighbourhood0]				#Slicing to avoid change G0	
		for k in range(g, N):	#2*m+1			
			value=random.randint(0,k-1)
			if len(Neighbourhood[value]) <= m:
				x=Neighbourhood[value][:]
			else:
				x=random.sample(Neighbourhood[value], m)		#No repeated edges
			for i in x:
				Neighbourhood[i].append(k) #Adding the 2 nodes to eachother's neighbourhood
				Neighbourhood[k].append(i)
		NeiTotal.append(Neighbourhood)
	for i in NeiTotal:
		degree=[len(j) for j in i]
		degreeTotal.append(degree)
	print time.time()-t0
	return np.array(degreeTotal)

####################SAVING AND LOADING#####################################
def SaveData(N, m):
	degree, edges =GenGraph(N, m)

	file_path_d = 'Database/Degree-m=' + str(m) + 'Nodes=' + str(N)+'.json'
	file_path_e = 'Database/Edges-m=' + str(m) + 'Nodes=' + str(N) +'.json'
		
	with open(file_path_d, 'w') as fp:
		json.dump(degree, fp)
	with open(file_path_e, 'w') as fp:
		json.dump(edges, fp)

def LoadData(N,m, T, Edges=False):
	file_path_d = 'Database/Degree-m=' + str(m) + 'Nodes=' + str(N)+'Trials='+str(T)+'.json'
	with open(file_path_d) as fp:
		degree = [json.load(fp)]
	
	if Edges==True:	
		file_path_e = 'Database/Edges-m=' + str(m) + 'Nodes=' + str(N)+'Trials='+str(T)+'.json'
		with open(file_path_e) as fp:
			edges = [json.load(fp)]
		return degree, edges
	else:
		return degree

def SaveDataTrials(N, m, T):
	degree =list(GenGraph3(N, m, T).flatten())
	file_path_d = 'Database/Degree-m=' + str(m) + 'Nodes=' + str(N)+'Trials='+str(T)+'.json'
		
	with open(file_path_d, 'w') as fp:
		json.dump(degree, fp)

##########################################################################
# GenGraph3(100000, 3,100)
# for m in range(1,5):
	# SaveDataTrials(100000,m,1)

# m=1
# degree=GenGraph3(1000,1,100).flatten()
# degree=list(np.array(GenGraph3(1000, 3, 100)).flatten())
# print degree
# m=3
# degree =LoadData(100000,m, 1000)
# deg, freq= lb.frequency(degree)
# norm=float(sum(freq))
# print len(deg)
# prob= freq/norm
# print len(prob)
# A= 2*m*(m+1)
# fit1= lambda x: A*x**-3 
# fit2= lambda x: A/(x*(x+1)*(x+2))  
# fit1a = fit1(deg)
# fit2a = fit2(deg) 
# plt.loglog(deg,prob, 'k+', zorder=3)
# plt.loglog(deg, fit1a, 'r--',zorder=1)
# plt.loglog(deg, fit2a, 'b-' ,zorder=2)
# plt.show()

# 	#CREATES ORIGINAL GRAPH
# 	#G=[1,2,1,3,2,3]
# 	#G=[1,2,1,3,1,4,1,5,2,3,2,4,2,5,3,4,34]
# m=2
# g=2*m+1
# for j in range(g-1): 
# 	for k in range(j+1,g):
# 		G.extend([j,k])

# print G
# print min(GenGraph3(10,3,1)[0])
# GenGraph3(10**5, 1, 1)
# GenGraph3(10**5,2,1)
# degree1 =GenGraph3(10000, 3,  1)
# print min(degree1)
# print min(degree2)
# plt.plot(degree1[0])
# plt.plot(degree2, alpha=.5)
# plt.show(