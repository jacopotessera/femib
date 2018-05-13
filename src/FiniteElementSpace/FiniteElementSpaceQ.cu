/*
*	FiniteElementSpaceQ.cu
*/

#include "FiniteElementSpaceQ.h"

template<MeshType meshType>
FiniteElementSpaceQ<meshType>& FiniteElementSpaceQ<meshType>::operator=(const FiniteElementSpace<meshType> &finiteElementSpace)
{
	this->T = finiteElementSpace.T;
	this->gauss = finiteElementSpace.gauss;
	this->finiteElement = finiteElementSpace.finiteElement;
	this->buildFiniteElementSpace();
	this->buildEdge();
	return *this;
}

template<MeshType meshType>
void FiniteElementSpaceQ<meshType>::buildEdge()
{
	this->edge = {0};
	this->nBT = 1;
	this->notEdge = setdiff(linspace(this->spaceDim),this->edge);

	std::vector<Eigen::Triplet<double>> tE;
	Eigen::SparseMatrix<double> sE(1,this->spaceDim);
	Eigen::Matrix<double,1,Eigen::Dynamic> dE(this->spaceDim);
	for(int n=0;n<this->nodes.T.size();++n)
	{
		for(int j=0;j<this->baseFunction.size();++j)
		{
			BaseFunction b = this->getBaseFunction(j,n);
			double valE = this->T.integrate(project(b.x,0),n);
			assert(b.i < this->spaceDim);
			tE.push_back(Eigen::Triplet<double>(0,b.i,valE));
		}
	}

	sE.setFromTriplets(tE.begin(),tE.end());
	dE = Eigen::Matrix<double,1,Eigen::Dynamic>(sE);
	this->E = dE / (-1*dE(0));
}

#define X(a) template FiniteElementSpaceQ<a>& FiniteElementSpaceQ<a>::operator=(const FiniteElementSpace<a> &finiteElementSpace);
MESH_TYPE_TABLE
#undef X

#define X(a) template void FiniteElementSpaceQ<a>::buildEdge();
MESH_TYPE_TABLE
#undef X

