/*
*	TriangleMesh.h
*/

#include "TriangleMesh.h"
#include <stdlib.h>

TriangleMesh::TriangleMesh(){}
TriangleMesh::~TriangleMesh(){}

TriangleMesh::TriangleMesh(Mesh m, Gauss g)
{
	mesh = m;
	gauss = g;

	setDim();
	setMeshDim();
	setAffineTransformation();	
}

void TriangleMesh::setDim()
{
	this->dim = mesh.P[0].size;
}

void TriangleMesh::setMeshDim()
{
	this->meshDim = mesh.T[0].size;
}

double TriangleMesh::getMeshRadius()
{
	double h = 0;
	for(ditrian t : mesh.T)
	{
		for(ditrian e : getEdges(t))
		{
			dvec v = mesh.P[e(0)]-mesh.P[e(1)];
			h = h > ddot(v,v) ? h : ddot(v,v);
		}
	}
	return h;
}

std::vector<std::vector<double>> TriangleMesh::getBox()
{
	std::vector<double> min = {0,0,0};
	std::vector<double> max = {0,0,0};

	for(dvec p : mesh.P)
	{
		for(int i=0;i<p.size;++i)
		{
			min[i] = p(i) < min[i] ? p(i) : min[i];
			max[i] = p(i) > max[i] ? p(i) : max[i];
		}
	}
	return {min,max};
}


void TriangleMesh::setAffineTransformation()
{
	for(int n=0;n<mesh.T.size();++n)
	{
		B.push_back(affineB(n,mesh));
		Binv.push_back(pinv(B[n]));
		Bdet.push_back(ddet(B[n]));

		b.push_back(affineb(n,mesh));

		std::vector<double> d_row;
		std::vector<dvec> p_row;
		d.push_back(d_row);
		p.push_back(p_row);
		for(int k=0;k<gauss.n;++k)
		{
			d[n].push_back(abs(Bdet[n])*gauss.weights[k]);
			p[n].push_back(B[n]*gauss.nodes[k]+b[n]);
		}
	}
}

void TriangleMesh::loadOnGPU()
{
	int q = mesh.T.size()/MAX_BLOCKS;
	int mod = mesh.T.size()%MAX_BLOCKS;

	dvec gP[mesh.P.size()];
	ditrian gT[q][MAX_BLOCKS];
	ditrian gTq[mod];

	for(int j=0;j<mesh.P.size();++j)
	{
		gP[j] = mesh.P[j];
	}

	for(int j=0;j<q*MAX_BLOCKS;++j)
	{
		gT[j/MAX_BLOCKS][j%MAX_BLOCKS] = mesh.T[j];
	}

	for(int j=0;j<mod;++j)
	{
		gTq[j] = mesh.T[MAX_BLOCKS*q+j];
	}

	HANDLE_ERROR(cudaMalloc((void**)&devP,mesh.P.size()*sizeof(dvec)));

	for(int i=0;i<q;++i)
	{
		devT.push_back(new ditrian());
		HANDLE_ERROR(cudaMalloc((void**)&(devT[i]),MAX_BLOCKS*sizeof(ditrian)));
	}

	if(mod>0)
	{
		HANDLE_ERROR(cudaMalloc((void**)&devTq,mod*sizeof(ditrian)));
	}

	HANDLE_ERROR(cudaMemcpy(devP,gP,mesh.P.size()*sizeof(dvec),cudaMemcpyHostToDevice));

	for(int i=0;i<q;++i)
	{
		HANDLE_ERROR(cudaMemcpy(devT[i],gT[i],MAX_BLOCKS*sizeof(ditrian),cudaMemcpyHostToDevice));
	}

	if(mod>0)
	{
		HANDLE_ERROR(cudaMemcpy(devTq,gTq,mod*sizeof(ditrian),cudaMemcpyHostToDevice));
	}
}

double TriangleMesh::integrate(const std::function<double(dvec)> &f, int n)
{
	__asm__ __volatile__("" :: "m" (f)); //TODO: wat?
	std::function<double(dvec)> g = [&,f](const dvec &x)
	//std::function<double(dvec)> g = [&f,this,&n](const dvec &x) //TODO: wat?
	{
		__asm__ __volatile__("" :: "m" (f)); //TODO: wat?
		return abs(Bdet[n])*f(B[n]*x+b[n]);
	};
	return gauss.integrate(g);
}

double TriangleMesh::integrate(const std::function<double(dvec)> &f)
{
	double integral = 0;
	for(int n=0;n<mesh.T.size();++n)
	{
		integral += integrate(f,n);
	}
	return integral;
}

std::vector<ditrian> getEdges(const ditrian &t)
{
	std::vector<ditrian> e;
	for(int i=0;i<t.size;++i)
	{
		for(int j=i+1;j<t.size;++j)
		{
			e.push_back(ditrian{t(i),t(j)});
		}	
	}
	return e;
}

