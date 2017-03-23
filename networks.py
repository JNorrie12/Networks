

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
#Slower way of building graph, however remembers edges
# def GenGraph(N, m, repeat=False, draw=False, G0=True):
# 	#2000=5secs 5000=13sec 10,000=51, 20,000=130
# 	t0=time.time()
# 	if G0==True:
# 		g= 2*m+1
# 		G=nx.complete_graph(g)
# 	#This gives G(0) with E(0)=m*N(0) satisfied
# 	else:
# 		g=0
# 		G=nx.Graph()
# 		G=nx.complete_graph(m)
		
# 	prob=np.ones(len(G))/(len(G))
# 	x=np.random.choice(np.arange(len(G)), p=prob)
# 	for k in range((g)+1, N+1):
# 		G.add_node(k)
# 		degree=np.array(G.degree(G).values())
# 		prob=(degree)/float(sum(degree))
# 		x=np.random.choice(G.nodes(), m, p=prob, replace=repeat)
# 		#Making m new edges
# 		y=list((k)*np.ones(m))
# 		z=zip(x,y)
# 		for i in z:
# 			G.add_edge(*i)
# 	if draw==True:
# 		nx.draw_circular(G)
# 		plt.show()
# 		####################
# 			# degree=np.array(G.degree(G).values())
# 			# prob=(degree)/float(sum(degree))
# 			# x=np.random.choice(G.nodes(), p=prob, replace=False)
# 	print time.time()-t0
# 	return G.degree(G).values(), G.edges(G)

# def RanGraph(N, m):
# 	t0=time.time()
# 	G=nx.Graph()
# 	g= 2*m+1
# 	G=nx.complete_graph(g)
# 	#This gives G(0) with E(0)=m*N(0) satisfied
# 	x=np.random.choice(np.arange(len(G)))
# 	for k in range((g)+1, N):
# 		G.add_node(k)
# 		degree=np.array(G.degree(G).values())
# 		x=np.random.choice(G.nodes(), m, replace=False)
# 		#Making m new edges
# 		y=list((k)*np.ones(m))
# 		z=zip(x,y)
# 		for i in z:
# 			G.add_edge(*i)
# 		####################
# 			# degree=np.array(G.degree(G).values())
# 			# prob=(degree)/float(sum(degree))
# 			# x=np.random.choice(G.nodes(), p=prob, replace=False)
# 	print time.time()-t0
# 	return G.degree(G).values(), G.edges(G)
###########################################################################
# def GenGraph2(N,m):
# 	t0=time.time()
# 	G=2*m*np.ones(2*m+1)
# 	prob=G/float(sum(G))
# 	nodes=np.array(range(len(G)))
# 	x=np.random.choice(np.arange(len(G)), p=prob)
# 	for k in range(2*m+1, N):
# 		G=np.append(G,0)
# 		nodes=np.append(nodes, k)
# 		prob=G/float(sum(G))
# 		x=np.random.choice(nodes, m, p=prob, replace=False)
# 		G[x]+=1
# 		G[-1]+=m
# 	print time.time()-t0
# 	return G


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
	return np.array(degreeTotal), G


# def RanGraph1(N,m, T):
# 	t0=time.time()
# 	GTotal=[]
# 	degreeTotal=[]
	
# 	G0=[] #Making initial graph combinatorically 
# 	g=2*m+1
# 	for j in range(g-1): 
# 		for k in range(j+1,g):
# 			G0.extend([j,k])
# 	for t in range(T):
# 		G=G0[:] 					#Slicing to avoid change G0	
# 		for k in range(2*m+1, N):
# 			x=[]					
# 			while len(x)<m:			#No repeated edges
# 				value=random.randint(0,k)
# 				if value in x:
# 					pass
# 				else:
# 					x.append(value)
# 			for i in x:
# 				G.append(k)
# 				G.append(i)
# 		GTotal.append(G)
# 	for i in GTotal:
# 		degree = Counter(i) #returns a list of degrees for each sample separately
# 		degreeTotal.append(degree.values())
# 	print time.time()-t0
# 	return np.array(degreeTotal)

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
	
	Neighbourhood=[[] for i in range(N)] #Neighbourhood of node i	
	
	for j in range(g):							#Initialisng complete graph in terms of neighbourhoods
		Neighbourhood[j].extend(range(j)) 		#Adds all nodes less than j to neighbourhood
		Neighbourhood[j].extend(range(j+1, g)) 	#Adds all nodes more than j to neighbourhood
	
	for t in range(T):
		G=G0[:] 					#Slicing to avoid change G0	
		for k in range(2*m+1, N):				
			x=[]
			while len(x)<m:			#No repeated edges
			# print Neighbourhood
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
WalkGraph1(100, 3, 1, 1)

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
	degree =list(np.array(GenGraph3(N, m, T)).flatten())
	file_path_d = 'Database/Degree-m=' + str(m) + 'Nodes=' + str(N)+'Trials='+str(T)+'.json'
		
	with open(file_path_d, 'w') as fp:
		json.dump(degree, fp)

##########################################################################
# GenGraph3(100000, 3,100)
# for m in range(1,5):
# m=1
	# SaveDataTrials(100000,m,1000)

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