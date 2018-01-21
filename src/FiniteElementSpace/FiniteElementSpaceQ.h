/*
*	FiniteElementSpaceQ.h
*/

#ifndef FINITEELEMENTSPACEQ_H_INCLUDED_
#define FINITEELEMENTSPACEQ_H_INCLUDED_

#include "FiniteElementSpace.h"

class FiniteElementSpaceQ : public FiniteElementSpace
{
	public:
		FiniteElementSpaceQ() : FiniteElementSpace(){};
		FiniteElementSpaceQ(TriangleMesh t, FiniteElement f, Gauss g) : FiniteElementSpace(t,f,g){};
		FiniteElementSpaceQ& operator=(const FiniteElementSpace &finiteElementSpace);
		void buildEdge();
};

#endif

