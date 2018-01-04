#!/bin/python

from matplotlib import pyplot

n = "5"
p = "perugiamesh/p"+n+".mat"
t = "perugiamesh/t"+n+".mat"
e = "perugiamesh/e"+n+".mat"

p_obj = open(p,"r").read().split("\n")
t_obj = open(t,"r").read().split("\n")
e_obj = open(e,"r").read().split("\n")

PP = []#[None]*len(p_obj)
TT = []#[None]*len(t_obj)
EE = []#[None]*len(e_obj)

for p_line in p_obj:
	for p_word in p_line.split("\t"):
		if p_word != '':
			PP.append(float(p_word))
for t_line in t_obj:
	for t_word in t_line.split("\t"):
		if t_word != '':
			TT.append(int(t_word))
for e_line in e_obj:
	for e_word in e_line.split("\t"):
		if e_word != '':
			EE.append(int(e_word))
P = []
dim=len(p_obj)-1
for i in range(int(len(PP)/dim)):
	P.append([])
	for j in range(dim):
		P[i].append(PP[i+j*int(len(PP)/dim)])
T = []
dim=len(t_obj)-1
for i in range(int(len(TT)/dim)):
	T.append([])
	for j in range(dim):
		T[i].append(TT[i+j*int(len(TT)/dim)])
E = EE

#print(P)
#print(T)
#print(E)

fig, ax = pyplot.subplots(1,1)
fig.set_tight_layout(True)

ax.cla()
X = list(map(lambda x: x[0],P))
Y = list(map(lambda x: x[1],P))
ax.plot(X,Y,"b.")
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
for tri in T:
	triX = [P[tri[0]][0],P[tri[1]][0],P[tri[2]][0],P[tri[0]][0]]
	triY = [P[tri[0]][1],P[tri[1]][1],P[tri[2]][1],P[tri[0]][1]]
	ax.plot(triX,triY,'b-',linewidth=0.2)

for ed in E:
	ax.plot(P[ed][0],P[ed][1],"r.")

#ax[0][0].set_xlabel(label)
pyplot.show()


