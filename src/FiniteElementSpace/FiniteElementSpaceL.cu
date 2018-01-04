/*
*	FiniteElementSpaceL.h
*/

#include "FiniteElementSpaceL.h"

FiniteElementSpaceL::FiniteElementSpaceL(){}

void FiniteElementSpaceL::buildEdge()
{
	nBT = 0;
	notEdge = linspace(spaceDim);
	C = compress(spaceDim,notEdge);
}

