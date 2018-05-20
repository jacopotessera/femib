#!/bin/python

#
#	plotUtils.py
#

import sys, pymongo, numpy
from functools import reduce

#class PlotUtils():

#	@staticmethod
def test():
	print("test")
	T = [
			None,
			"",
			"A",
			"1",
			"2,B",
			"1,2,1:2,3",
			"2:B",
			"5:3",
			"1,2,3:5:9",
			"5:5","1,2",
			"1:3",
			"1:4,6,8,10:15",
			"1,3:5,7"
		]
	for t in T:
		try:
			print(t,parse_id(t))
		except Exception as e:
			print(t)
			pass
	sims = [{"id": "a"},{"id": "b"},{"id": "c"},{"id": "d"},{"id": "e"}]
	inputs = [
				['plotUtils.py'],
				['plotUtils.py','1'],
				['plotUtils.py','1:2'],
				['plotUtils.py','A'],
				['plotUtils.py','1','op'],
				['plotUtils.py','1:2','op'],
				['plotUtils.py','A','op'],
				['plotUtils.py','A','op','wut'],
		]
	for i in inputs:
		sys.argv = i
		print(sys.argv,parse_input(sims))

#	@staticmethod
def parse_id(i):
	ret = []
	try:
		if i is None or i == "":
			return []
		J = i.split(",")
		for j in J:
			K = j.split(":")
			if len(K)==1:
				ret.append(int(j))				
			elif len(K)==2:
				k0 = int(K[0])
				k1 = int(K[1])+1
				if k0 < k1:
					for j in range(k0,k1):
						ret.append(j)
				else:
					raise Exception("invalid interval: " + j)
			else:
				raise Exception("invalid interval: " + j)
	except Exception as e:
		print(e)
		#return []
		raise
	return list(set(ret))

def parse_input(sims):
	try:
		if len(sys.argv)==1:
			return {"op": "list"}
		elif len(sys.argv)==2:
			return {"op": "plot", "id_" : sims[int(sys.argv[1])-1]["id"]}
		elif len(sys.argv)==3:
			ids_ = parse_id(sys.argv[1])
			ids = []
			for id_ in ids_:
				ids.append(sims[id_-1]["id"])
			op = sys.argv[2]
			return {"op": op, "id_" : ids}
		else:
			raise Exception("cant parse input :|")
	except Exception as e:
		print(e)
		return {"op": "list"}

def calcPlotData(timesteps,parameters,TMAX,plotConfig):
	T = []
	X = []
	Y = []
	U = []
	V = []
	Z = []
	P = []
	A = []
	E = []
	W = []

	for t in timesteps:
		if t["time"]%(numpy.ceil(TMAX/plotConfig["ffw"])) == 0:
			T.append(t["time"]*parameters["deltat"])

			xx = list(map(lambda x : x[1][0],t["x"] ))
			yy = list(map(lambda x : x[1][1],t["x"] ))
			X.append(xx)
			Y.append(yy)

			uu = list(map(lambda x : x[1][0],t["u"] ))
			vv = list(map(lambda x : x[1][1],t["u"] ))
			zz = list(map(lambda x : numpy.sqrt(x[1][0]*x[1][0]+x[1][1]*x[1][1]),t["u"] ))

			uuu = [0]*len(uu)
			for i,u in enumerate(uu):
				ii = i//21
				qq = i%21
				uuu[i] = uu[qq*21+ii]

			vvv = [0]*len(vv)
			for i,v in enumerate(vv):
				ii = i//21
				qq = i%21
				vvv[i] = vv[qq*21+ii]

			zzz = [[0]*21 ]*21
			for i,v in enumerate(zz):
				ii = i//21
				qq = i%21
				zzz[ii][qq] = zz[ii*21+qq]


			U.append(uu)
			V.append(vv)
			Z.append(zz)
			W.append({	"x":list(map(lambda x : x[0][0],t["u"] )),"y":list(map(lambda x : x[0][1],t["u"] )),
						"u":list(map(lambda x : x[1][0],t["u"] )),"v":list(map(lambda x : x[1][1],t["u"] ))})

			p0 = list(filter(lambda x : abs(x[0][0])<0.001 and abs(x[0][1])<0.001, t["q"]))
			print("p(0,0) = ",p0[0][1][0])

			p1 = list(filter(lambda x : abs(x[0][0]-1)<0.001 and abs(x[0][1]-1)<0.001, t["q"]))
			print("p(1,1) = ",p1[0][1][0])

			pp = []
			for ii in range(len(t["q"])):
				if ii%(plotConfig["steps"]+1)==0:
					pp.append([])
				pp[-1].append(t["q"][ii][1][0])
			P.append(pp)
			
			area = abs (reduce( (lambda x,y : x+y ) , map( lambda x : 0.5*(x[0]*x[3]-x[1]*x[2]) ,zip(xx,yy,numpy.roll(xx,-1),numpy.roll(yy,-1)) ) ) )
			A.append(area/plotConfig["area0"])

			size = len(xx)
			aa = numpy.sqrt( (xx[0]-xx[int(size/2)])**2+(yy[0]-yy[int(size/2)])**2 )
			AA = numpy.sqrt( (xx[int(size/4)]-xx[int(3*size/4)])**2+(yy[int(size/4)]-yy[int(3*size/4)])**2 )

			E.append({"a":aa,"A":AA,"e":aa/AA})
		
			print(str(t["time"]) + ": area = " + str(area/plotConfig["area0"]) + "%, e = " +str(aa/AA))

	return {"T":T,"X":[X,Y],"U":[U,V,Z],"P":P,"A":[A,E],"W":W}
	
if __name__ == '__main__':
	#PlotUtils.test()
	test()

