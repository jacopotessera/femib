/*
*	FiniteElementSpaceV.cu
*/

#include "FiniteElementSpaceV.h"

template<MeshType meshType>
FiniteElementSpaceV<meshType>& FiniteElementSpaceV<meshType>::operator=(const FiniteElementSpace<meshType> &finiteElementSpace)
{
	this->T = finiteElementSpace.T;
	this->gauss = finiteElementSpace.gauss;
	this->finiteElement = finiteElementSpace.finiteElement;
	this->buildFiniteElementSpace();
	this->buildEdge();
	return *this;
}

template<MeshType meshType>
void FiniteElementSpaceV<meshType>::buildEdge()
{
	this->edge = join(this->nodes.E,this->nodes.E+this->spaceDim/this->ambientDim);//TODO 1d? 3d?
	this->nBT = this->edge.size();
	this->notEdge = setdiff(linspace(this->spaceDim),this->edge);
	std::vector<Eigen::Triplet<double>> tE;
	Eigen::SparseMatrix<double> sE(this->nBT,this->spaceDim);
	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> dE(this->nBT,this->spaceDim);
	for(int i=0;i<this->nBT;++i)
	{
		tE.push_back({i,this->edge[i],-1.0});
	}
	sE.setFromTriplets(tE.begin(),tE.end());
	dE = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>(sE);
	this->E = dE;
}

#define X(a) template FiniteElementSpaceV<a>& FiniteElementSpaceV<a>::operator=(const FiniteElementSpace<a> &finiteElementSpace);
MESH_TYPE_TABLE
#undef X

#define X(a) template void FiniteElementSpaceV<a>::buildEdge();
MESH_TYPE_TABLE
#undef X

