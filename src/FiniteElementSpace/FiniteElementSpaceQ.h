/*
*	FiniteElementSpaceQ.h
*/

#ifndef FINITEELEMENTSPACEQ_H_INCLUDED_
#define FINITEELEMENTSPACEQ_H_INCLUDED_

#include "FiniteElementSpace.h"

class FiniteElementSpaceQ : public FiniteElementSpace
{
	public:
		FiniteElementSpaceQ();
		FiniteElementSpaceQ(TriangleMesh t, FiniteElement f, Gauss g) : FiniteElementSpace(t,f,g){};
		void buildEdge();
};

#endif

