/*
*	dmat.h
*/

#ifndef DMAT_H_INCLUDED_
#define DMAT_H_INCLUDED_

#define M_DVEC 3
#define M_DTRIAN 4

#include <initializer_list>

#include "../../lib/Log.h"

struct dvec
{
	int size = 0;
	double v[M_DVEC] = {0,0,0};
	double& operator()(int row);
	double operator()(int row) const;
	dvec();
	//dvec(int size);
	dvec(const std::initializer_list<double> &list);
	dvec& operator+=(const dvec &a);
};

struct divec
{
	int size = 0;
	int v[M_DVEC] = {-1,-1,-1};
	int& operator()(int row);
	int operator()(int row) const;
	divec();
	divec(const std::initializer_list<int> &list);
};

struct dmat
{
	int rows = 0;
	int cols = 0;
	double m[M_DVEC][M_DVEC] = {{0,0,0},{0,0,0},{0,0,0}};
	double& operator()(int row, int col);
	double operator()(int row, int col) const;
	dmat();
	//dmat(int rows, int cols);
	dmat(const std::initializer_list<std::initializer_list<double> > &list);
	dmat& operator*=(double a);
};

struct dtrian
{
	int size = 0;
	dvec p[M_DTRIAN] = {{},{},{},{}};
	dvec& operator()(int row);
	dvec operator()(int row) const;
	dtrian();
	dtrian(const std::initializer_list<dvec> &list);
};

struct ditrian
{
	int size = 0;	
	int p[M_DTRIAN] = {-1,-1,-1,-1};
	int& operator()(int row);
	int operator()(int row) const;
	ditrian();
	ditrian(const std::initializer_list<int> &list);
};

namespace std
{
	template<>
	struct hash<dvec>
	{
		std::size_t operator()(const dvec &f) const
		{
			return
				std::hash<int>{}(f.size)
				^ std::hash<double>{}(f.v[0]) << 4
				^ std::hash<double>{}(f.v[1]) << 8
				^ std::hash<double>{}(f.v[2]) << 12;
		}
	};
}

#endif

