/*
*	FiniteElementSpaceS.cu
*/

#include "FiniteElementSpaceS.h"

FiniteElementSpaceS::FiniteElementSpaceS(){}

void FiniteElementSpaceS::buildEdge()
{
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

