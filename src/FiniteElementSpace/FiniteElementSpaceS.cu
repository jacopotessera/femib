/*
*	FiniteElementSpaceS.cu
*/

#include "FiniteElementSpaceS.h"

FiniteElementSpaceS::FiniteElementSpaceS(){}

void FiniteElementSpaceS::buildEdge()
{
	nBT = 0;
	notEdge = linspace(spaceDim);
	C = compress(spaceDim,notEdge);
}

