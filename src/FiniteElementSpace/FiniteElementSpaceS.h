/*
*	FiniteElementSpaceS.h
*/

#ifndef FINITEELEMENTSPACES_H_INCLUDED_
#define FINITEELEMENTSPACES_H_INCLUDED_

#include "FiniteElementSpace.h"

class FiniteElementSpaceS : public FiniteElementSpace
{
	public:
		FiniteElementSpaceS() : FiniteElementSpace(){};
		FiniteElementSpaceS(TriangleMesh t, FiniteElement f, Gauss g) : FiniteElementSpace(t,f,g){};
		FiniteElementSpaceS& operator=(const FiniteElementSpace &finiteElementSpace);
		void buildEdge();
};

#endif

