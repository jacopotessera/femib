/*
*	FiniteElementSpaceL.h
*/

#include "FiniteElementSpaceL.h"

template<MeshType meshType>
FiniteElementSpaceL<meshType>& FiniteElementSpaceL<meshType>::operator=(const FiniteElementSpace<meshType> &finiteElementSpace)
{
	this->T = finiteElementSpace.T;
	this->gauss = finiteElementSpace.gauss;
	this->finiteElement = finiteElementSpace.finiteElement;
	this->buildFiniteElementSpace();
	this->buildEdge();
	return *this;
}

template<MeshType meshType>
void FiniteElementSpaceL<meshType>::buildEdge()
{
	this->nBT = 0;
	this->notEdge = linspace(this->spaceDim);

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
	
	/*bool thin = true; //TODO
	bool thick = !thin;
	if(thin)
	{
		LOG_INFO("finiteElemenSpaceL: thin.");
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
		LOG_INFO("finiteElemenSpaceL: thick.");
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
	}*/
}

#define X(a) template FiniteElementSpaceL<a>& FiniteElementSpaceL<a>::operator=(const FiniteElementSpace<a> &finiteElementSpace);
MESH_TYPE_TABLE
#undef X

#define X(a) template void FiniteElementSpaceL<a>::buildEdge();
MESH_TYPE_TABLE
#undef X
