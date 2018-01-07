/*
*	Simulation.h
*/

#ifndef SIM_H_INCLUDED_
#define SIM_H_INCLUDED_

#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <mutex>

#include <Eigen/Eigen>

#include "../../lib/mini-book.h"

#include "../dmat/dmat.h"
#include "../dmat/dmat_impl.h"

#include "../utils/utils.h"
#include "../Gauss/Gauss.h"
#include "../Gauss/GaussService.h"
#include "../utils/Mesh.h"

#include "../TriangleMesh/TriangleMesh.h"
#include "../FiniteElement/FiniteElementService.h"
#include "../FiniteElementSpace/FiniteElementSpace.h"
#include "../FiniteElementSpace/FiniteElementSpaceV.h"
#include "../FiniteElementSpace/FiniteElementSpaceQ.h"
#include "../FiniteElementSpace/FiniteElementSpaceS.h"
#include "../FiniteElementSpace/FiniteElementSpaceL.h"

#include "../tensorAlgebra/tensorAlgebra.h"
#include "../mongodb/struct.h"
#include "../mongodb/dbconfig.h"
#include "../mongodb/mongodb.h"
#include "../Cuda/Cuda.h"

#include <fstream>

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> edmat;
typedef std::vector<Eigen::Triplet<double>> etmat;
typedef Eigen::SparseMatrix<double> esmat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> evec;

class Simulation
{
	public:

		Cuda c;
		GaussService gaussService;
		FiniteElementService finiteElementService;

		dbconfig db;
		std::string id;
		Parameters parameters;
		bool full;

		FiniteElementSpaceV V;
		FiniteElementSpaceQ Q;
		FiniteElementSpaceS S;
		FiniteElementSpaceL L;

		std::vector<int> edge;
		std::vector<int> notEdge;

		int time; //TODO
		std::vector<timestep> timesteps;

		Simulation();
		Simulation(std::string id,dbconfig db,Parameters parameters,FiniteElementSpaceV V,FiniteElementSpaceQ Q,FiniteElementSpaceS S,FiniteElementSpaceL L,timestep t0,timestep t1);
		Simulation(std::string id,dbconfig db,Parameters parameters,FiniteElementSpaceV V,FiniteElementSpaceQ Q,FiniteElementSpaceS S,timestep t0,timestep t1);
		~Simulation();

		void setParameters(Parameters parameters);
		void setTime(int time);
		void setInitialValues(timestep t0, timestep t1);

		miniSim sim2miniSim();
		void saveSimulation();
		void getSimulation(dbconfig db,std::string id);

		timestep eigen2timestep(evec a);
		void saveTimestep(int time);
		void saveTimestep();
		timestep getTimestep(int time);
		plotData timestep2plotData(timestep t);
		void savePlotData(int time);
		void savePlotData();


		void saveStreamline();

		void buildEdge();
		void buildFluidMatrices();
		void buildStructureMatrices();
		void buildMultiplierMatrices();

		void prepare();
		void clear();

		void buildLs();
		void buildK2f();
		void buildLf();
		void buildF();

		void buildb();

		void advance();
		void advance(int steps);
		void solve();
		void save();
		void updateX();
		void triplet2sparse();

	//private:

		std::vector<std::vector<int>> M;
		std::vector<std::vector<int>> MM;

		etmat C;
		etmat Ct;

		esmat sC;
		esmat sCt;
		evec b;

		etmat Mf;
		etmat K1f;
		etmat K2f;
		etmat MB;

		etmat B;

		etmat Ms;
		etmat Ks;
		etmat Ls;
		etmat Lf;

		evec FF;

		esmat sMB;
		esmat sMf;
		esmat sMs;
		esmat sLs;
		esmat sK2f;
		esmat sLf;

		//TODO solver in constructor
		//Eigen::SparseSolverBase<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>,Eigen::Lower|Eigen::Upper> solver;
};

#endif

