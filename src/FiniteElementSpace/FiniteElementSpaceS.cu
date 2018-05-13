/*
*	FiniteElementSpaceS.cu
*/

#include "FiniteElementSpaceS.h"

template<MeshType meshType>
FiniteElementSpaceS<meshType>& FiniteElementSpaceS<meshType>::operator=(const FiniteElementSpace<meshType> &finiteElementSpace)
{
	this->T = finiteElementSpace.T;
	this->gauss = finiteElementSpace.gauss;
	this->finiteElement = finiteElementSpace.finiteElement;
	this->thickness = THIN;
	this->buildFiniteElementSpace();
	this->buildEdge();
	return *this;
}

template<MeshType meshType>
void FiniteElementSpaceS<meshType>::buildEdge()
{
	if(thickness==THIN)
	{
		LOG_INFO("finiteElemenSpaceS: thin.");
		this->edge = join(this->nodes.E,this->nodes.E+(this->spaceDim/this->ambientDim));//TODO 1d? 3d?
		this->nBT = this->edge.size();
		this->notEdge = setdiff(linspace(this->spaceDim),this->edge);
		std::vector<Eigen::Triplet<double>> tE;
		Eigen::SparseMatrix<double> sE(2,this->spaceDim);
		Eigen::Matrix<double,2,Eigen::Dynamic> dE(2,this->spaceDim);

		for(int i=0;i<this->edge.size();++i)
		{
			tE.push_back({i,i*(this->spaceDim/this->ambientDim),1.0});
			tE.push_back({i,this->edge[i],-1.0});
		}
		std::ostringstream tt;
		tt << tE;
		LOG_OK(tt);
		sE.setFromTriplets(tE.begin(),tE.end());
		dE = Eigen::Matrix<double,2,Eigen::Dynamic>(sE);
		this->E = dE;
		std::ostringstream ss;
		ss << this->E;
		LOG_OK(ss);
	}
	else if(thickness==THICK)
	{
		LOG_INFO("finiteElemenSpaceS: thick.");
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
	}
}

#define X(a) template FiniteElementSpaceS<a>& FiniteElementSpaceS<a>::operator=(const FiniteElementSpace<a> &finiteElementSpace);
MESH_TYPE_TABLE
#undef X

/*
#define X(a) template FiniteElementSpaceS<a>::FiniteElementSpaceS();
MESH_TYPE_TABLE
#undef X

#define X(a) template FiniteElementSpaceS<a>::FiniteElementSpaceS(SimplicialMesh<a> t, FiniteElement f, Gauss g, STRUCTURE_THICKNESS thickness);
MESH_TYPE_TABLE
#undef X
*/

#define X(a) template void FiniteElementSpaceS<a>::buildEdge();
MESH_TYPE_TABLE
#undef X

