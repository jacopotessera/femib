/*
*	FiniteElementSpaceS.h
*/

#ifndef FINITEELEMENTSPACES_H_INCLUDED_
#define FINITEELEMENTSPACES_H_INCLUDED_

#include "FiniteElementSpace.h"

class FiniteElementSpaceS : public FiniteElementSpace
{
	public:
		FiniteElementSpaceS();
		FiniteElementSpaceS(TriangleMesh t, FiniteElement f, Gauss g) : FiniteElementSpace(t,f,g){};
		void buildEdge();
};

#endif

