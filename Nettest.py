import numpy as np
import matplotlib.pyplot as plt
import time
import log_bin as lb
def GenGraph1(N,m):
	t0=time.time()
	G=np.concatenate((2*m*np.ones(2*m+1),np.zeros(N-2*m-1)))
	prob=G/float(sum(G))
	nodes=np.array(range(N))
	x=np.random.choice(np.arange(len(G)), p=prob)
	for k in range(2*m+1, N):
		# G.extend(0)
		# G=np.append(G,0)
		# nodes.extend(k)
		# nodes=np.append(nodes, k)
		prob=G/float(sum(G))
		x=np.random.choice(nodes, m, p=prob, replace=False)
		for i in x:
			G[i]+=1
		G[k]+=m
		# y=list((k)*np.ones(m))
		# z=zip(x,y)
		# for i in z:
			# G.add_edge(*i)
	print time.time()-t0
	return G

def GenGraph2(N,m):
	t0=time.time()
	G=2*m*np.ones(2*m+1)
	prob=G/float(sum(G))
	nodes=np.array(range(len(G)))
	x=np.random.choice(np.arange(len(G)), p=prob)
	for k in range(2*m+1, N):
		# G.extend(0)
		G=np.append(G,0)
		# nodes.extend(k)
		nodes=np.append(nodes, k)
		prob=G/float(sum(G))
		x=np.random.choice(nodes, m, p=prob, replace=False)
		G[x]+=1
		# for i in x:
			# G[i]+=1
		G[-1]+=m
		# y=list((k)*np.ones(m))
		# z=zip(x,y)
		# for i in z:
			# G.add_edge(*i)
	print time.time()-t0
	return G



# Gengraph(100,1)
# g = GenGraph1(10000,1)
m=2
degree = GenGraph2(10000, m)
deg, freq= lb.frequency(degree)
norm=float(sum(freq))
print norm
prob= freq/norm
plt.loglog(deg,prob, 'r.')
A= 2*m*(m+1)
# fit1=A*(deg)**-3
# fit3=A/(deg*(deg+1)*(deg+2))

fit1= lambda x: A*x**-3 
fit3= lambda x: A/(x*(x+1)*(x+2))  
fit1a = fit1(deg)
fit3a = fit3(deg) 
plt.loglog(deg, fit1a)
plt.loglog(deg, fit3a)
c, e=lb.log_bin(degree, bin_start=m-0.5 ,first_bin_width=1, a=1.8)
plt.loglog(c,e)
# print c[0]-c[1]
plt.show()
# pr]int min(g)
plt.plot(g)
plt.show()