/*
*	FiniteElementSpaceL.h
*/

#include "FiniteElementSpaceL.h"

FiniteElementSpaceL::FiniteElementSpaceL(){}

void FiniteElementSpaceL::buildEdge()
{
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

