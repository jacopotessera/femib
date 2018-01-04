/*
*	FiniteElementSpaceQ.cu
*/

#include "FiniteElementSpaceQ.h"

FiniteElementSpaceQ::FiniteElementSpaceQ(){}

void FiniteElementSpaceQ::buildEdge()
{
	edge = {0};
	nBT = 1; //p(0,0)=0
	notEdge = setdiff(linspace(spaceDim),edge);
	C = compress(spaceDim,notEdge);
	Eigen::Matrix<double,Eigen::Dynamic,1> b(nBT);
	b(0) = 0;

	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> B(nBT,spaceDim);
	for(int i=0;i<spaceDim;i++)
		B(0,i) = 1;
}

