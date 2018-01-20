/*
*	FiniteElementSpaceV.cu
*/

#include "FiniteElementSpaceV.h"

FiniteElementSpaceV& FiniteElementSpaceV::operator=(const FiniteElementSpace &finiteElementSpace)
{
	T = finiteElementSpace.T;
	gauss = finiteElementSpace.gauss;
	finiteElement = finiteElementSpace.finiteElement;
	buildFiniteElementSpace();
	return *this;
}

void FiniteElementSpaceV::buildEdge()
{
	edge = join(nodes.E,nodes.E+spaceDim/ambientDim);//TODO 1d? 3d?
	nBT = edge.size();
	notEdge = setdiff(linspace(spaceDim),edge);

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

