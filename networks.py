import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import log_bin as lb
import time
import json
import scipy as sp
import scipy.stats as stats
def GenGraph(N, m):
	#2000=5secs 5000=13sec 10,000=51, 20,000=130
	t0=time.time()
	G=nx.Graph()
	G=nx.complete_graph(2*m +1)
	#This gives G(0) with E(0)=m*N(0) satisfied
	prob=np.ones(2*m+1)/(2*m+1)
	x=np.random.choice(np.arange(len(G)), p=prob)
	for k in range((2*m+1)+1, N):
		G.add_node(k)
		degree=np.array(G.degree(G).values())
		prob=(degree)/float(sum(degree))
		x=np.random.choice(G.nodes(), m, p=prob, replace=False)
		# print x
		y=list((k)*np.ones(m))
		z=zip(x,y)
		for i in z:
			G.add_edge(*i)
			# print G.degree(k)
		####################
			# degree=np.array(G.degree(G).values())
			# prob=(degree)/float(sum(degree))
			# x=np.random.choice(G.nodes(), p=prob, replace=False)
#If we have 2 edges between 2 nodes then networkx counts it as one.
#Maybe allow this to happen or not
#Maybe look to if it makes a difference for big t
	print time.time()-t0
	return G.degree(G).values(), G.edges(G)
####################SAVING AND LOADING#####################################
def SaveData(N, m):
	degree, edges =GenGraph(N, m)

	file_path_d = 'Database/Degree-m=' + str(m) + 'Nodes=' + str(N)+'.json'
	file_path_e = 'Database/Edges-m=' + str(m) + 'Nodes=' + str(N) +'.json'
		
	with open(file_path_d, 'w') as fp:
		json.dump(degree, fp)
	with open(file_path_e, 'w') as fp:
		json.dump(edges, fp)

def LoadData(N,m):
	file_path_d = 'Database/Degree-m=' + str(m) + 'Nodes=' + str(N)+'.json'
	with open(file_path_d) as fp:
		degree = [json.load(fp)]
		
	file_path_e = 'Database/Edges-m=' + str(m) + 'Nodes=' + str(N)+'.json'
	with open(file_path_e) as fp:
		edges = [json.load(fp)]
	return degree, edges		

# d,e =GenGraph(100,1)
# print d
# SaveData(100000,1)

degreeL, edges= LoadData(10000,2)
degree=degreeL[0]
num_bin=max(degree)
deg, probs= lb.lin_bin(degree, num_bin)
plt.plot(deg, probs, 'r.', zorder=1)
d, e=lb.log_bin(degree, a=1.5)
# plt.show()


# fit1= 10*(np.array(range(1,float(max(degree)))))**-3
# x=np.linspace
fit=9*(deg)**-3 
plt.plot(deg, fit)
plt.show()

plt.loglog(deg, fit, 'r--')
plt.loglog(deg, probs, 'b.')
plt.show()
# fitting=sp.optimize.curve_fit(fit, probs, deg)
# print fitting
# plt.plot(x, fit[0]*x+fit[1], 'r-')
# print fit
# plt.plot(np.log(b), np.log(c), 'b-')
# print len(G)
# nx.draw_networkx(G, node_size=2*degree, with_labels=False)
# plt.show()

# t = time.time()-t0
# print t