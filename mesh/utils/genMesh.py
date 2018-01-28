#!/bin/python

import math, numpy, sys
from matplotlib import pyplot

rr = 0.6
dd = 0.1
xC = 0.0
yC = 0.0

s = 8
n = 3
save = False

if len(sys.argv)>1:
	s = int(sys.argv[1])
	n = int(sys.argv[2])
	save = ( int(sys.argv[3]) == 1 )
dd = dd/n

if s%8!=0:
	raise Exception("Error: Non-symmetric mesh!")

P = []
phi = 2*math.pi/s
for m in range(n):
	for i in range(s):
		P.append([xC+(rr+m*dd)*math.cos(phi*i),yC+(rr+m*dd)*math.sin(phi*i)])

for p in enumerate(P):
	print(p)


T = []
for m in range(n-1):
	for t in range(int(s/2)-1):
		t = 2*t
		T.append([t+m*s,t+(m+1)*s,t+m*s+1])
		T.append([t+(m+1)*s+1,t+m*s+1,t+(m+1)*s])

		T.append([t+(m+1)*s+2,t+m*s+2,t+m*s+1])
		T.append([t+m*s+1,t+(m+1)*s+1,t+(m+1)*s+2])
	T.append([(m+1)*s,m*s,(m+1)*s-1])
	T.append([(m+1)*s-1,(m+2)*s-1,(m+1)*s])
	T.append([(m+1)*s+s-2,m*s+s-2,m*s+s-1])
	T.append([m*s+s-1,(m+1)*s+s-1,(m+1)*s+s-2])

print("")
for t in T:
	print(t)


fig, ax = pyplot.subplots(1,1)
fig.set_tight_layout(True)

ax.cla()
X = list(map(lambda x: x[0],P))
Y = list(map(lambda x: x[1],P))
#Z = list(map(lambda x: x[2],P))
ax.plot(X,Y,"b.")
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

for t in T:
	tX = [P[t[0]][0],P[t[1]][0],P[t[2]][0],P[t[0]][0]]
	tY = [P[t[0]][1],P[t[1]][1],P[t[2]][1],P[t[0]][1]]
	#tZ = [P[t[0]][2],P[t[1]][2],P[t[2]][2],P[t[0]][2]]
	ax.plot(tX,tY,'b-',linewidth=0.2)


pyplot.show()

if save:
	f = open('pS_'+str(s)+'_'+str(n)+'.mat', 'w')
	for p in P:
		f.write(str(p[0])+'\t'+str(p[1])+'\t\n')
	f.close()

	f = open('tS_'+str(s)+'_'+str(n)+'.mat', 'w')
	for t in T:
		f.write(str(t[0])+'\t'+str(t[1])+'\t'+str(t[2])+'\t\n')
	f.close()

