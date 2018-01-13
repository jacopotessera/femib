/*
*	FiniteElementSpaceQ.cu
*/

#include "FiniteElementSpaceQ.h"

FiniteElementSpaceQ::FiniteElementSpaceQ(){}

void FiniteElementSpaceQ::buildEdge()
{
	edge = {0};
	nBT = 1;
	notEdge = setdiff(linspace(spaceDim),edge);

	std::vector<Eigen::Triplet<double>> tE;
	Eigen::SparseMatrix<double> sE(1,spaceDim);
	Eigen::Matrix<double,1,Eigen::Dynamic> dE(spaceDim);
	for(int n=0;n<nodes.T.size();++n)
	{
		for(int j=0;j<baseFunction.size();++j)
		{
			BaseFunction b = getBaseFunction(j,n);
			double valE = T.integrate(project(b.x,0),n);
			assert(b.i < spaceDim);
			tE.push_back(Eigen::Triplet<double>(0,b.i,valE));
		}
	}

	sE.setFromTriplets(tE.begin(),tE.end());
	dE = Eigen::Matrix<double,1,Eigen::Dynamic>(sE);
	E = dE / (-1*dE(0));


}

