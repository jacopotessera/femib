/*
*	FiniteElementSpaceS.cu
*/

#include "FiniteElementSpaceS.h"

FiniteElementSpaceS::FiniteElementSpaceS(){}

void FiniteElementSpaceS::buildEdge()
{
	edge = join(nodes.E,nodes.E+spaceDim/ambientDim);//TODO 1d? 3d?
	nBT = edge.size();
	notEdge = setdiff(linspace(spaceDim),edge);
}

