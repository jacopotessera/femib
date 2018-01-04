/*
*	FiniteElementSpaceV.cu
*/

#ifndef FINITEELEMENTSPACEV_H_INCLUDED_
#define FINITEELEMENTSPACEV_H_INCLUDED_

#include "FiniteElementSpace.h"

class FiniteElementSpaceV : public FiniteElementSpace
{
	public:
		FiniteElementSpaceV();
		FiniteElementSpaceV(TriangleMesh t, FiniteElement f, Gauss g) : FiniteElementSpace(t,f,g){};
		void buildEdge();
};

#endif

