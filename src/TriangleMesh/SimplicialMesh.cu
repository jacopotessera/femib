/*
*	SimplicialMesh.cu
*/

#include "SimplicialMesh.h"
#include <stdlib.h>

template<MeshType T>
SimplicialMesh<T>::SimplicialMesh(){}

template<MeshType T>
SimplicialMesh<T>::~SimplicialMesh(){}

template<MeshType T>
SimplicialMesh<T>::SimplicialMesh(Mesh m, Gauss g)
{
	mesh = m;
	gauss = g;
	setTriangleMesh();
}

template<MeshType T>
double SimplicialMesh<T>::integrate(const std::function<double(dvec)> &f, int n)
{
	__asm__ __volatile__("" :: "m" (f));

	double integral = 0;
	for(int mm=0;mm<q2t[n].size();++mm)
	{
		int mmm = q2t[n][mm];
		std::function<double(dvec)> g = [&,this,f](const dvec &x)
		//std::function<double(dvec)> g = [&f,this,&n](const dvec &x) //TODO: wat?
		{
			__asm__ __volatile__("" :: "m" (f));
			return abs(triangleMesh.Bdet[mmm])*f(triangleMesh.B[mmm]*x+triangleMesh.b[mmm]);
		};
		integral += gauss.integrate(g);
	}
	return integral;
}

template<MeshType T>
double SimplicialMesh<T>::integrate(const std::function<double(dvec)> &f)
{
	double integral = 0;
	for(int n=0;n<mesh.T.size();++n)
	{
		integral += integrate(f,n);
	}
	return integral;
}

#define X(a) template SimplicialMesh<a>::SimplicialMesh();
MESH_TYPE_TABLE
#undef X

#define X(a) template SimplicialMesh<a>::~SimplicialMesh();
MESH_TYPE_TABLE
#undef X

#define X(a) template SimplicialMesh<a>::SimplicialMesh(Mesh m, Gauss g);
MESH_TYPE_TABLE
#undef X

#define X(a) template double SimplicialMesh<a>::integrate(const std::function<double(dvec)> &f, int n);
MESH_TYPE_TABLE
#undef X

#define X(a) template double SimplicialMesh<a>::integrate(const std::function<double(dvec)> &f);
MESH_TYPE_TABLE
#undef X

template<MeshType T>
void SimplicialMesh<T>::setTriangleMesh(){}

/*template<MeshType T>
dvec SimplicialMesh<T>::toMesh0x(const dvec& x, int n){}

template<MeshType T>
dmat SimplicialMesh<T>::toMesh0dx(const dvec& x, int n){}*/

template<>
void SimplicialMesh<oneDim>::setTriangleMesh()
{
	mesh0.P = {{0.0},{1.0}};
	mesh0.T = {{0,1}};

	for(int mm=0;mm<mesh0.T.size();++mm)
	{
		B0.push_back(affineB(mm,mesh0));
		b0.push_back(affineb(mm,mesh0));
sLOG_OK("oneDim: " << B0.size() << " " << b0.size() );
	}

	int n=0;
	for(auto q : mesh.T)
	{
		q2t.push_back({n++});
	}
	triangleMesh = TriangleMesh(mesh,gauss);
}

template<>
void SimplicialMesh<Triangular>::setTriangleMesh()
{
	mesh0.P = {{0.0,0.0},{1.0,0.0},{0.0,1.0}};
	mesh0.T = {{0,1,2}};

	for(int mm=0;mm<mesh0.T.size();++mm)
	{
		B0.push_back(affineB(mm,mesh0));
		b0.push_back(affineb(mm,mesh0));
sLOG_OK("Triangular: " << B0.size() << " " << b0.size() );
	}

	int n=0;	
	for(auto q : mesh.T)
	{
		q2t.push_back({n++});
	}
	triangleMesh = TriangleMesh(mesh,gauss);
}

template<>
void SimplicialMesh<Parallelogram>::setTriangleMesh()
{
	mesh0.P = {{0.0,0.0},{1.0,0.0},{1.0,1.0},{0.0,1.0}};
	mesh0.T = {{0,1,3},{2,3,1}};

	for(int mm=0;mm<mesh0.T.size();++mm)
	{
		B0.push_back(affineB(mm,mesh0));
		b0.push_back(affineb(mm,mesh0));
sLOG_OK("Parallelogram: " << B0.size() << " " << b0.size() );
	}

	Mesh mt;
	mt.P = mesh.P;
	mt.E = mesh.E;
	
	for(auto q : mesh.T)
	{
		//sLOG_DEBUG(q);
		q2t.push_back({(int)mt.T.size(),(int)mt.T.size()+1});
		mt.T.push_back({q(0),q(1),q(3)});
		mt.T.push_back({q(2),q(3),q(1)});
	}

	for(auto q : mt.T)
	{
		//sLOG_DEBUG(q);
	}

	triangleMesh = TriangleMesh(mt,gauss);
}

#define X(a) template void SimplicialMesh<a>::setTriangleMesh();
MESH_TYPE_TABLE
#undef X

template<MeshType meshType>
dvec SimplicialMesh<meshType>::toMesh0x(const dvec& x, int n)
{
	for(int mm=0;mm<q2t[n].size();++mm)
	{
		int m = q2t[n][mm];
		dvec X = triangleMesh.Binv[m]*(x-triangleMesh.b[m]);
		if(in_std(X))
		{
//sLOG_OK(B0[mm]*X+b0[mm]);
			return B0[mm]*X+b0[mm];
			//return affineB(mm,mesh0)*X+affineb(mm,mesh0);
		}
	}
	throw EXCEPTION("x is not in T[n]");
}

#define X(a) template dvec SimplicialMesh<a>::toMesh0x(const dvec& x, int n);
MESH_TYPE_TABLE
#undef X

template<MeshType meshType>
dmat SimplicialMesh<meshType>::toMesh0dx(const dvec& x, int n)
{

	for(int mm=0;mm<q2t[n].size();++mm)
	{
		int m = q2t[n][mm];
		dvec X = triangleMesh.Binv[m]*(x-triangleMesh.b[m]);
		if(in_std(X))
		{
//sLOG_OK(B0[mm]*triangleMesh.Binv[m]);
			return B0[mm]*triangleMesh.Binv[m];
			//return affineB(mm,mesh0)*triangleMesh.Binv[m];
		}
	}
	throw EXCEPTION("x is not in T[n]");
}

#define X(a) template dmat SimplicialMesh<a>::toMesh0dx(const dvec& x, int n);
MESH_TYPE_TABLE
#undef X

template<MeshType meshType>
bool SimplicialMesh<meshType>::xInN(const dvec& x, int n){
	//sLOG_DEBUG("x: " << x );
	bool e = false;
	for(auto m : q2t[n])
	{
		//sLOG_DEBUG("T: " << ditrian2dtrian(triangleMesh.mesh.T[m],triangleMesh.mesh.P.data()));
		e = e || accurate(x,ditrian2dtrian(triangleMesh.mesh.T[m],triangleMesh.mesh.P.data()));
	}
	return e;
}

#define X(a) template bool SimplicialMesh<a>::xInN(const dvec& x, int n);
MESH_TYPE_TABLE
#undef X
