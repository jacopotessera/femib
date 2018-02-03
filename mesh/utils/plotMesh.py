#!/bin/python

import math, numpy, sys
from functools import reduce
from matplotlib import pyplot

prefix = sys.argv[1]
suffix = sys.argv[2]

p_ = prefix+"p"+suffix
t_ = prefix+"t"+suffix
e_ = prefix+"e"+suffix

p_file = open(p_,"r").read().split("\n")
t_file = open(t_,"r").read().split("\n")
e_file = open(e_,"r").read().split("\n")

P = []
T = []
E = []

for p_line in p_file:
	temp = []
	for p_word in p_line.split("\t"):
		if p_word != '':
			temp.append(float(p_word))
	if temp != []:
		P.append(temp)	

for t_line in t_file:
	temp = []
	for t_word in t_line.split("\t"):
		if t_word != '':
			temp.append(int(t_word))
	if temp != []:
		T.append(temp)

for e_line in e_file:
	for e_word in e_line.split("\t"):
		if e_word != '':
			E.append(int(e_word))

#L = list(map(lambda x : list(map(lambda y : (P[y[1]][0]-P[y[0]][0])**2+(P[y[1]][1]-P[y[0]][1])**2 ,x)) , list(map(lambda x : list(zip(x,numpy.roll(x,-1))),T))))
#LL = list(map(lambda x : (min(x),max(x),min(x)/max(x)) , L ))

#min_min = math.sqrt(min(list(map(lambda x : x[0],LL))))
#max_min = math.sqrt(min(list(map(lambda x : x[0],LL))))
#min_max = math.sqrt(min(list(map(lambda x : x[1],LL))))
#max_max = math.sqrt(min(list(map(lambda x : x[1],LL))))
#min_ = math.sqrt(min(list(map(lambda x : x[2],LL))))
#max_ = math.sqrt(min(list(map(lambda x : x[2],LL))))

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

for e in E:
	ax.plot(P[e][0],P[e][1],"r.")

ax.set_xlabel(p_)#+": "+str(min_)+" - "+str(max_max))
pyplot.show()

