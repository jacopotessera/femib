#!/bin/python

import os
import sys
import csv
import numpy
import matplotlib.pyplot as pyplot
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from functools import reduce
from matplotlib import colors, ticker, cm
import matplotlib

from pymongo import  MongoClient
import pymongo
import pandas
import pprint

ffw = 40
area0 = numpy.pi*0.6*0.6
x_min = -1
x_max = 1
y_min = -1
y_max = 1
steps = 20

db = MongoClient('localhost', 27017)['testSimulation']
simCollection = db['sim']
timestepCollection = db['plotData']
id_ = "1"

sim = simCollection.find({"id" : id_},{"S.mesh" : 1,"parameters" : 1})
deltat = sim[0]["parameters"]["deltat"]
timesteps = timestepCollection.find({"id" : id_},{"time" : 1, "x" : 1})
TMAX = timestepCollection.find({"id" : id_},{"time" : 1, "x" : 1}).sort([("time", pymongo.DESCENDING)]).limit(1)[0]["time"]

T = []
X = []
Y = []
A = []
E = []

U = []
V = []
XX, YY = numpy.mgrid[x_min:x_max:steps*1j, y_min:y_max:steps*1j]
XXX = numpy.linspace(x_min,x_max,steps+1)
YYY = numpy.linspace(y_max,y_min,steps+1)
P = []

'''
for i in range(TMAX):
	U.append([])
	V.append([])
	P.append([])

streamline = ["x","y","v_x","v_y"]
pressure = ["x","y","p"]
for dirname, dirnames, filenames in os.walk('plot/streamline'):
	for filename in filenames:
		print(filename)
		if filename.split('_')[0] == 'sl':
			with open(os.path.join(dirname, filename)) as csvfile:
				reader = csv.DictReader(csvfile,delimiter=';',fieldnames=streamline)
				t = filename.split('_')[1].split('.')[0]
				if int(t)<TMAX:
					for row in reader:
						U[int(t)].append(float(row['v_x']))
						V[int(t)].append(float(row['v_y']))
					U[int(t)] = numpy.asarray(U[int(t)]).reshape(41,41)
					V[int(t)] = numpy.asarray(V[int(t)]).reshape(41,41)
		if filename.split('_')[0] == 'p':
			with open(os.path.join(dirname, filename)) as csvfile:
				reader = csv.DictReader(csvfile,delimiter=';',fieldnames=pressure)
				t = filename.split('_')[1].split('.')[0]
				if int(t)<TMAX:
					for row in reader:
						P[int(t)].append(float(row['p']))
					P[int(t)] = numpy.asarray(P[int(t)]).reshape(41,41)
'''
for t in timesteps:

	if t["time"]%(numpy.ceil(TMAX/ffw)) == 0:
		#xx = t["x"][:int(len(t["x"])/2)]
		#yy = t["x"][int(len(t["x"])/2):]
		xx = list(map(lambda x : x[1][0],t["x"] ))
		yy = list(map(lambda x : x[1][1],t["x"] ))
		
		T.append(t["time"]*deltat)
		X.append(xx)
		Y.append(yy)
		size = len(xx)

		area = abs (reduce( (lambda x,y : x+y ) , map( lambda x : 0.5*(x[0]*x[3]-x[1]*x[2]) ,zip(xx,yy,numpy.roll(xx,-1),numpy.roll(yy,-1)) ) ) )
		A.append(area/area0)

		_min = reduce((lambda v,w : v if v<w else w),map(lambda x :  max(list(map(lambda y : (y[0]-x[0])**2+(y[1]-x[1])**2,zip(xx,yy)))) , zip(xx,yy)))
		_max = reduce((lambda v,w : v if v>w else w),map(lambda x :  max(list(map(lambda y : (y[0]-x[0])**2+(y[1]-x[1])**2,zip(xx,yy)))) , zip(xx,yy)))

		aa = numpy.sqrt( (xx[0]-xx[int(size/2)])**2+(yy[0]-yy[int(size/2)])**2 )
		AA = numpy.sqrt( (xx[int(size/4)]-xx[int(3*size/4)])**2+(yy[int(size/4)]-yy[int(3*size/4)])**2 )
		E.append({"a":aa,"A":AA,"e":_min/_max})
	
		delta = str( (E[-1]["e"]) - (E[-2]["e"])) if len(E)>1 else "0"

		print(str(t["time"]) + ": area = " + str(area/area0) + " " +str(aa) + " " +str(AA) + " " +str(aa/AA)+ " " +str(_min/_max) + " " +  delta )

fig, ax = pyplot.subplots(2,3)
fig.set_tight_layout(True)

print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

def update(i):

	ax[0][0].cla()
	ax[0][0].plot(X[i],Y[i],".-")
	ax[0][0].set_xlim([-1,1])
	ax[0][0].set_ylim([-1,1])
	#label = 'timestep {0},time {3:.2f}, area {1:.4f}, ratio {2:.4f}'.format(ffw*i,area/area0,_min/_max,ffw*i*0.01)
	label = 'time {0:.2f}'.format(T[i])	
	ax[0][0].set_xlabel(label)
	#ax[0][0].quiver(XXX,YYY,U[i],V[i],pivot='mid', units='inches')
	'''
	ax[0][1].cla()
	ax[0][1].set_xlim([-1,1])
	ax[0][1].set_ylim([-1,1])
	label = 'time {0:.2f}'.format(T[i])	
	ax[0][1].set_xlabel(label)
	ax[0][1].quiver(XXX,YYY,U[i],V[i],pivot='mid',width=0.005)
	
	ax[0][2].cla()
	ax[0][2].set_xlim([-1,1])
	ax[0][2].set_ylim([-1,1])
	label = 'time {0:.2f}'.format(T[i])	
	ax[0][2].set_xlabel(label)
	matplotlib.rcParams['xtick.direction'] = 'out'
	matplotlib.rcParams['ytick.direction'] = 'out'
	matplotlib.rcParams['contour.negative_linestyle'] = 'dashed'
	cs = ax[0][2].contour(XXX,YYY,P[i],10,colors='k')
	ax[0][2].clabel(cs, inline=1, fontsize=10)
	'''
	ax[1][0].cla()
	ax[1][0].plot(T[:i],A[:i],"-")
	ax[1][0].set_xlim([0,T[-1]])
	ax[1][0].set_ylim([0.7,1])

	ax[1][1].cla()
	ax[1][1].plot(T[:i],list(map(lambda x: x["a"]/x["A"],E))[:i],"-")
	ax[1][1].set_xlim([0,T[-1]])
	ax[1][1].set_ylim([0.78,1.22])

	ax[1][2].cla()
	ax[1][2].plot(T[:i],list(map(lambda x: x["a"]/(x["a"]+x["A"]),E))[:i],"-")
	ax[1][2].plot(T[:i],list(map(lambda x: x["A"]/(x["a"]+x["A"]),E))[:i],"-")
	ax[1][2].set_xlim([0,T[-1]])
	ax[1][2].set_ylim([0.4,0.6])

	return ax

if __name__ == '__main__':
    anim = FuncAnimation(fig, update, frames=numpy.arange(0, len(T)), interval=1)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('plot.gif', dpi=80, writer='imagemagick')
    else:
        pyplot.show()

