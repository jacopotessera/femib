/*
*	FiniteElementSpaceS.cu
*/

#include "FiniteElementSpaceS.h"

FiniteElementSpaceS& FiniteElementSpaceS::operator=(const FiniteElementSpace &finiteElementSpace)
{
	T = finiteElementSpace.T;
	gauss = finiteElementSpace.gauss;
	finiteElement = finiteElementSpace.finiteElement;
	buildFiniteElementSpace();
	buildEdge();
	return *this;
}

void FiniteElementSpaceS::buildEdge()
{
	bool thin = true; //TODO
	bool thick = !thin;
	if(thin)
	{
		LOG_INFO("finiteElemenSpaceS: thin.");
		edge = join(nodes.E,nodes.E+spaceDim/ambientDim);//TODO 1d? 3d?
		nBT = edge.size();
		notEdge = setdiff(linspace(spaceDim),edge);

		std::vector<Eigen::Triplet<double>> tE;
		Eigen::SparseMatrix<double> sE(2,spaceDim);
		Eigen::Matrix<double,2,Eigen::Dynamic> dE(2,spaceDim);

		for(int i=0;i<edge.size();++i)
		{
			tE.push_back({i,i*(spaceDim/ambientDim),1.0});
			tE.push_back({i,edge[i],-1.0});
		}
		sE.setFromTriplets(tE.begin(),tE.end());
		dE = Eigen::Matrix<double,2,Eigen::Dynamic>(sE);
		E = dE;
	}
	if(thick)
	{
		LOG_INFO("finiteElemenSpaceS: thick.");
		nBT = 0;
		notEdge = linspace(spaceDim);

		std::vector<Eigen::Triplet<double>> tE;
		Eigen::SparseMatrix<double> sE(nBT,spaceDim);
		Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> dE(nBT,spaceDim);

		for(int i=0;i<nBT;++i)
		{
			tE.push_back({i,edge[i],-1.0});
		}
		sE.setFromTriplets(tE.begin(),tE.end());
		dE = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>(sE);
		E = dE;
	}
}

