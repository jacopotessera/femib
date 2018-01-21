/*
*	FiniteElementSpaceL
*/

#ifndef FINITEELEMENTSPACEL_H_INCLUDED_
#define FINITEELEMENTSPACEL_H_INCLUDED_

#include "FiniteElementSpace.h"

class FiniteElementSpaceL : public FiniteElementSpace
{
	public:
		FiniteElementSpaceL() : FiniteElementSpace(){};
		FiniteElementSpaceL(TriangleMesh t, FiniteElement f, Gauss g) : FiniteElementSpace(t,f,g){};
		FiniteElementSpaceL& operator=(const FiniteElementSpace &finiteElementSpace);
		void buildEdge();
};

#endif

