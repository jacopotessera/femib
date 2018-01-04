/*
*	Mesh.h
*/

#ifndef MESH_H_INCLUDED_
#define MESH_H_INCLUDED_

#include <functional>
#include <vector>

#include "../dmat/dmat.h"

struct Mesh
{
	std::vector<dvec> P;
	std::vector<ditrian> T;
	std::vector<int> E;
};

struct Nodes
{
	std::vector<dvec> P;
	std::vector<std::vector<int> > T;
	std::vector<int> E;
};

struct xDx
{
	dvec x;
	dmat dx;
};

struct F
{
	std::function<dvec(dvec)> x;
	std::function<dmat(dvec)> dx;
	xDx operator()(const dvec &x)
	{
		return {this->x(x),this->dx(x)};
	};
};

struct BaseFunction
{
	F f;
	std::function<dvec(dvec)> x;
	std::function<dmat(dvec)> dx;
	int i;
	int mini_i;
};

#endif

