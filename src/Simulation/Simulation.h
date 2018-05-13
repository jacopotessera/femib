/*
*	Simulation.h
*/

#ifndef SIM_H_INCLUDED_
#define SIM_H_INCLUDED_

#include <string>
#include <vector>
#include <iostream>

#include <Eigen/Eigen>

#include "../../lib/mini-book.h"
#include "../../lib/Log.h"

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

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> edmat;
typedef std::vector<Eigen::Triplet<double>> etmat;
typedef Eigen::SparseMatrix<double> esmat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> evec;

typedef FiniteElementSpaceV<Triangular> FiniteElementSpaceV_;
typedef FiniteElementSpaceQ<Triangular> FiniteElementSpaceQ_;
typedef FiniteElementSpaceS<Triangular> FiniteElementSpaceS_;
typedef FiniteElementSpaceL<Triangular> FiniteElementSpaceL_;

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

		FiniteElementSpaceV_ V;
		FiniteElementSpaceQ_ Q;
		FiniteElementSpaceS_ S;
		FiniteElementSpaceL_ L;

		std::vector<int> edge;
		std::vector<int> notEdge;

		std::vector<timestep> timesteps;

		Simulation();
		~Simulation();

		double getEnergy(timestep t);

		Simulation(std::string id,dbconfig db,Parameters parameters,
			FiniteElementSpaceV_ V,FiniteElementSpaceQ_ Q,
			FiniteElementSpaceS_ S,FiniteElementSpaceL_ L,
			timestep t0,timestep t1,bool full);

		Simulation(std::string id,dbconfig db,Parameters parameters,
			FiniteElementSpaceV_ V,FiniteElementSpaceQ_ Q,FiniteElementSpaceS_ S,FiniteElementSpaceL_ L,
			timestep t0,timestep t1) : Simulation(id,db,parameters,V,Q,S,L,t0,t1,true){};

		Simulation(std::string id,dbconfig db,Parameters parameters,
			FiniteElementSpaceV_ V,FiniteElementSpaceQ_ Q,FiniteElementSpaceS_ S,
			timestep t0,timestep t1) : Simulation(id,db,parameters,V,Q,S,FiniteElementSpaceL_(),t0,t1,false){};

		Simulation(dbconfig db,Parameters parameters,
			FiniteElementSpaceV_ V,FiniteElementSpaceQ_ Q,FiniteElementSpaceS_ S,FiniteElementSpaceL_ L,
			timestep t0,timestep t1) : Simulation(getTimestamp(),db,parameters,V,Q,S,L,t0,t1,true){};

		Simulation(dbconfig db,Parameters parameters,
			FiniteElementSpaceV_ V,FiniteElementSpaceQ_ Q,FiniteElementSpaceS_ S,
			timestep t0,timestep t1) : Simulation(getTimestamp(),db,parameters,V,Q,S,FiniteElementSpaceL_(),t0,t1,false){};

		void setParameters(Parameters parameters);
		void setTime(int time);
		void setInitialValues(timestep t0, timestep t1);

		miniSim sim2miniSim();
		void saveSimulation();
		void getSimulation(dbconfig db,std::string id);

		int getTime();

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

