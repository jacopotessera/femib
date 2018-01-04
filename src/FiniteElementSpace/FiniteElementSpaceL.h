/*
*	FiniteElementSpaceL
*/

#ifndef FINITEELEMENTSPACEL_H_INCLUDED_
#define FINITEELEMENTSPACEL_H_INCLUDED_

#include "FiniteElementSpace.h"

class FiniteElementSpaceL : public FiniteElementSpace
{
	public:
		FiniteElementSpaceL();
		FiniteElementSpaceL(TriangleMesh t, FiniteElement f, Gauss g) : FiniteElementSpace(t,f,g){};
		void buildEdge();
};

#endif

