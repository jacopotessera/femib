/*
*	FiniteElementSpaceV.cu
*/

#ifndef FINITEELEMENTSPACEV_H_INCLUDED_
#define FINITEELEMENTSPACEV_H_INCLUDED_

#include "FiniteElementSpace.h"

template<MeshType meshType>
class FiniteElementSpaceV : public FiniteElementSpace<meshType>
{
	public:
		FiniteElementSpaceV() : FiniteElementSpace<meshType>(){};
		FiniteElementSpaceV(SimplicialMesh<meshType> t, FiniteElement f, Gauss g) : FiniteElementSpace<meshType>(t,f,g){};
		FiniteElementSpaceV& operator=(const FiniteElementSpace<meshType> &finiteElementSpace);
		void buildEdge();
};

#endif

