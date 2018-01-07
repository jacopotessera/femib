/*
*	FiniteElementSpaceV.cu
*/

#include "FiniteElementSpaceV.h"

FiniteElementSpaceV& FiniteElementSpaceV::operator=(const FiniteElementSpace &finiteElementSpace)
{
	T = finiteElementSpace.T;
	gauss = finiteElementSpace.gauss;
	finiteElement = finiteElementSpace.finiteElement;
	buildFiniteElementSpace();
	return *this;
}

void FiniteElementSpaceV::buildEdge()
{
	edge = join(nodes.E,nodes.E+spaceDim/ambientDim); 
	nBT = edge.size();
	notEdge = setdiff(linspace(spaceDim),edge);
	C = compress(spaceDim,notEdge);
}

