#!/bin/python

#
#	plotSimulation.py
#

from plotUtils import parse_id, parse_input, calcPlotData
import sys, pymongo, numpy, matplotlib, matplotlib.pyplot as pyplot
from matplotlib.animation import FuncAnimation

class PlotSimulation():

	def __init__(self,db_name):
		self.db = pymongo.MongoClient('localhost', 27017)[db_name]
		self.simCollection = self.db['sim']
		self.timestepCollection = self.db['timestep']
		self.plotDataCollection = self.db['plotData']
		self.sims = self.simCollection.find({},{"id": 1}).sort([("id", pymongo.DESCENDING)])

		self.plotData = {}
		self.plotConfig = {"ffw": 40, "steps": 20, "area0": numpy.pi*0.6*0.6, "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1}
	
	def do(self):
		op = parse_input(self.sims)
		if op["op"] == "list":
			self.list()
		elif op["op"] == "plot":
			self.plot(op["id_"])
		elif op["op"] == "save":
			self.save(op["id_"])
		elif op["op"] == "delete":
			self.delete(op["id_"])
		else:
			print("invalid op :|")

	def list(self):
		print("Available Simulations:")
		n = 1
		for n,sim in enumerate(self.sims):
			id_ = str(sim["id"])
			try:
				TMAX = self.timestepCollection.find({"id" : id_},{"time" : 1}).sort([("time", pymongo.DESCENDING)]).limit(1)[0]["time"]
			except:
				TMAX = 0
			print("\t* ["+str(n+1)+"] "+ id_ +" - "+str(TMAX)+" timesteps")
		print("")
		print("plot with:")
		print("\t$ make plot ID=n")
		sys.exit()

	def delete(self,ids):
		for id_ in ids:
			print("Deleting Simulation with id "+id_+" ...")
			self.simCollection.delete_many({"id" : id_})
			self.timestepCollection.delete_many({"id" : id_})
			self.plotDataCollection.delete_many({"id" : id_})
		sys.exit()

	def save(self,ids):
		for id_ in ids:
			print("Saving Simulation with id "+id_+" ...")
			self.plot(id_,True)
		sys.exit()

	def plot(self,id_,save=False):

		sim = self.simCollection.find({"id" : id_},{"S":1,"S.mesh" : 1,"parameters" : 1,"full": 1})
		parameters = sim[0]["parameters"]
		deltat = sim[0]["parameters"]["deltat"]
		TMAX = self.plotDataCollection.find({"id" : id_},{"time" : 1, "x" : 1}).sort([("time", pymongo.DESCENDING)]).limit(1)[0]["time"]
		timesteps = self.plotDataCollection.find({"id" : id_},{"time" : 1, "x" : 1,"u": 1,"q": 1})

		if not save:
			print("Plotting Simulation with id "+id_+" ...")
			print("full:",sim[0]["full"])
			print(parameters)

		grid_x, grid_y = numpy.meshgrid(	numpy.arange(self.plotConfig["x_min"],self.plotConfig["x_max"]+0.1,0.1),
											numpy.arange(self.plotConfig["y_min"],self.plotConfig["y_max"]+0.1,0.1))

		self.plotData = calcPlotData(timesteps,parameters,TMAX,self.plotConfig)

		fig, ax = pyplot.subplots(2,3)
		fig.set_tight_layout(True)
		#print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))


		#grid_x = self.plotData["W"][0]["x"]
		#grid_y = self.plotData["W"][0]["y"]

		def update(i):

			ax[0][0].cla()
			ax[0][0].plot(self.plotData["X"][0][i],self.plotData["X"][1][i],".-")
			ax[0][0].set_xlim([-1,1])
			ax[0][0].set_ylim([-1,1])
			#label = 'timestep {0},time {3:.2f}, area {1:.4f}, ratio {2:.4f}'.format(ffw*i,area/area0,_min/_max,ffw*i*0.01)
			label = 'time {0:.2f}'.format(self.plotData["T"][i])	
			ax[0][0].set_xlabel(label)
			ax[0][0].quiver(grid_x,grid_y,self.plotData["U"][0][i],self.plotData["U"][1][i],pivot='mid',units='inches')
			circle = pyplot.Circle((0,0), 0.72, color='r',fill=False)
			ax[0][0].add_artist(circle)
			circle = pyplot.Circle((0,0), 0.4, color='r',fill=False)
			ax[0][0].add_artist(circle)
			ax[0][0].axis('equal')

			ax[0][1].cla()
			ax[0][1].set_xlim([-1,1])
			ax[0][1].set_ylim([-1,1])
			label = 'time {0:.2f}'.format(self.plotData["T"][i])
			ax[0][1].set_xlabel(label)
			ax[0][1].quiver(grid_x,grid_y,self.plotData["U"][0][i],self.plotData["U"][1][i],pivot='mid',width=0.005)

			ax[0][2].cla()
			ax[0][2].set_xlim([-1,1])
			ax[0][2].set_ylim([-1,1])
			label = 'time {0:.2f}'.format(self.plotData["T"][i])	
			ax[0][2].set_xlabel(label)
			matplotlib.rcParams['xtick.direction'] = 'out'
			matplotlib.rcParams['ytick.direction'] = 'out'
			matplotlib.rcParams['contour.negative_linestyle'] = 'dashed'
			cs = ax[0][2].contour(grid_x,grid_y,self.plotData["P"][i])
			ax[0][2].clabel(cs, inline=1, fontsize=10)

			ax[1][0].cla()
			ax[1][0].plot(self.plotData["T"][:i],self.plotData["A"][0][:i],"-")
			ax[1][0].set_xlim([0,self.plotData["T"][-1]])
			ax[1][0].set_ylim([0.0,1])

			ax[1][1].cla()
			ax[1][1].plot(self.plotData["T"][:i],list(map(lambda x: x["e"],self.plotData["A"][1]))[:i],"-")
			ax[1][1].set_xlim([0,self.plotData["T"][-1]])
			ax[1][1].set_ylim([0.78,1.22])

			ax[1][2].cla()
			ax[1][2].plot(self.plotData["T"][:i],list(map(lambda x: x["a"]/(x["a"]+x["A"]),self.plotData["A"][1]))[:i],"-")
			ax[1][2].plot(self.plotData["T"][:i],list(map(lambda x: x["A"]/(x["a"]+x["A"]),self.plotData["A"][1]))[:i],"-")
			ax[1][2].set_xlim([0,self.plotData["T"][-1]])
			ax[1][2].set_ylim([0.4,0.6])

			return ax

		anim = FuncAnimation(fig, update, frames=numpy.arange(0, len(self.plotData["T"])), interval=1)
		
		if save:
			anim.save('gifs/Simulation_'+id_+'.gif', dpi=600, writer='imagemagick')
		else:
			pyplot.show()

if __name__ == '__main__':
	db_name = 'testSimulation'
	plotSimulation = PlotSimulation(db_name)
	plotSimulation.do()	
	
