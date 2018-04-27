/*
*	QuadMesh.h
*/

#include "QuadMesh.h"
#include <stdlib.h>

QuadMesh::QuadMesh(){}
QuadMesh::~QuadMesh(){}

QuadMesh::QuadMesh(Mesh m, Gauss g)
{
	mesh = m;
	gauss = g;

	setDim();
	setMeshDim();
	setTriangleMesh();

}

void QuadMesh::setTriangleMesh()
{
	Mesh<ditrian> mt;
	mt.P = m.P;
	mt.E = m.E;
	
	for(auto q : m.T)
	{
		q2t.append({mt.T.size(),mt.T.size()+1});
		mt.T.append({q(0),q(1),q(3)});
		mt.T.append({q(2),q(3),q(1)});

	}
	triangleMesh = TriangleMesh(mt,g);
}

double QuadMesh::integrate(const std::function<double(dvec)> &f, int n)
{
	__asm__ __volatile__("" :: "m" (f)); //TODO: wat?

	double integral = 0;
	for(int m : q2t[n])
	{
		std::function<double(dvec)> g = [&,f](const dvec &x)
		//std::function<double(dvec)> g = [&f,this,&n](const dvec &x) //TODO: wat?
		{
			__asm__ __volatile__("" :: "m" (f)); //TODO: wat?
			return abs(Bdet[m])*f(B[m]*x+b[m]);
		};
		integral += gauss.integrate(g);
	}
	return integral;
}

double QuadMesh::integrate(const std::function<double(dvec)> &f)
{
	double integral = 0;
	for(int n=0;n<mesh.T.size();++n)
	{
		integral += integrate(f,n);
	}
	return integral;
}
