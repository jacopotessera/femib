/*
*	affine.cu
*/

#include "affine.h"

dmat affineB(int n, const Mesh &mesh)
{
	sLOG_TRACE("mesh.P:");
	for(auto q : mesh.P)
		sLOG_TRACE(q);
	sLOG_TRACE("n: " << n);
	sLOG_TRACE("mesh.T[n]:" << mesh.T[n]);

	dmat B;
	B.rows = mesh.P[0].size;
	if(mesh.T[0].size > 0)
		B.cols = mesh.T[0].size-1;
	else
		throw EXCEPTION("affineB: dtrian size is 0");

	for(int i=0;i<B.rows;++i)
	{
		for(int j=0;j<B.cols;++j)
		{
			B(i,j)=mesh.P[mesh.T[n](j+1)](i)-mesh.P[mesh.T[n](0)](i);
		}
	}
	sLOG_TRACE(B);
	return B;
}

dvec affineb(int n, const Mesh &mesh)
{
	sLOG_TRACE("mesh.P:");
	for(auto q : mesh.P)
		sLOG_TRACE(q);
	sLOG_TRACE("n: " << n);
	sLOG_TRACE("mesh.T[n]:" << mesh.T[n]);

	dvec b = mesh.P[mesh.T[n](0)];
	sLOG_TRACE(b);
	return b;
}

