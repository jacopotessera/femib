/*
*	FiniteElementSpaceQ.h
*/

#ifndef FINITEELEMENTSPACEQ_H_INCLUDED_
#define FINITEELEMENTSPACEQ_H_INCLUDED_

#include "FiniteElementSpace.h"

template<MeshType meshType>
class FiniteElementSpaceQ : public FiniteElementSpace<meshType>
{
	public:
		FiniteElementSpaceQ() : FiniteElementSpace<meshType>(){};
		FiniteElementSpaceQ(SimplicialMesh<meshType> t, FiniteElement f, Gauss g) : FiniteElementSpace<meshType>(t,f,g){};
		FiniteElementSpaceQ<meshType>& operator=(const FiniteElementSpace<meshType> &finiteElementSpace);
		void buildEdge();
};

#endif

