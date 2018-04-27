/*
*	dquad.h
*/

#ifndef DQUAD_H_INCLUDED_
#define DQUAD_H_INCLUDED_

#define M_DVEC 3
#define M_DQUAD 8

#include <initializer_list>

#include "dmat.h"
#include "dmat_impl.h"
#include "../../lib/Log.h"

struct dquad
{
	int size = 0;
	dvec p[M_DQUAD] = {{},{},{},{},{},{},{},{}};
	dvec& operator()(int row);
	dvec operator()(int row) const;
	dquad();
	dquad(const std::initializer_list<dvec> &list);
};

struct diquad
{
	int size = 0;	
	int p[M_DQUAD] = {-1,-1,-1,-1,-1,-1,-1,-1};
	int& operator()(int row);
	int operator()(int row) const;
	diquad();
	diquad(const std::initializer_list<int> &list);
};

#endif

