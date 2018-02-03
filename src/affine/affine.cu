/*
*	affine.cu
*/

#include "affine.h"

dmat affineB(int n, const Mesh &mesh)
{
	dmat B;
	B.rows = mesh.P[0].size;
	if(mesh.T[0].size > 0)
	{
		B.cols = mesh.T[0].size-1;
	}
	else
	{
		LOG_WARNING("dtrian size is 0?");
		B.cols = 0;
	}
	//B.cols = (mesh.T[0].size-1) >=0 ? (mesh.T[0].size-1) : 0;
	for(int i=0;i<B.rows;++i)
	{
		for(int j=0;j<B.cols;++j)
		{
			B(i,j)=mesh.P[mesh.T[n](j+1)](i)-mesh.P[mesh.T[n](0)](i);
		}
	}
	return B;
}

dvec affineb(int n, const Mesh &mesh)
{
	return mesh.P[mesh.T[n](0)];
}

