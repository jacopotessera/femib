/*
*	FiniteElementSpaceV.cu
*/

#include "FiniteElementSpaceV.h"

FiniteElementSpaceV::FiniteElementSpaceV(){}

void FiniteElementSpaceV::buildEdge()
{
	edge = join(nodes.E,nodes.E+spaceDim/ambientDim); 
	nBT = edge.size();
	notEdge = setdiff(linspace(spaceDim),edge);
	C = compress(spaceDim,notEdge);
}

