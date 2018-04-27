/*
*	dquad_impl.h
*/

#ifndef DQUAD_IMPL_H_INCLUDED_
#define DQUAD_IMPL_H_INCLUDED_

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dquad.h"
#include "../utils/Exception.h"

#define M_PI           3.14159265358979323846
#define M_EPS          0.000001
#define M_EPS2         M_EPS*M_EPS

std::ostream& operator<<(std::ostream& out, const dquad &s);
std::ostream& operator<<(std::ostream& out, const diquad &t);

__host__ __device__ bool operator==(const dquad lhs, dquad rhs);
__host__ __device__ bool operator==(const diquad lhs, diquad rhs);

__host__ __device__ double dmax(const dquad &A, int j);
__host__ __device__ double dmin(const dquad &A, int j);

__host__ __device__ bool in_box(const dvec &P, const dquad &T);
__host__ __device__ bool in_std(const dvec &P);
__host__ __device__ bool in_triangle(const dvec &P, const dquad &T);

__host__ __device__ double distancePointSegment(const dvec &P, const dquad &T);
__host__ __device__ double distancePointTriangle(const dvec &P, const dquad &T);

__host__ __device__ bool accurate(const dvec &P, const dquad &T);
__global__ void parallel_accurate(dvec *P,diquad *T,dvec *X,bool *N);
__host__ bool serial_accurate(dvec *P,diquad T,dvec X);

__host__ __device__ dquad diquad2dquad(const diquad &T, const dvec *P);

__host__ __device__ dtrian[M_QUAD] dquad2dtrian(const dquad &T);
//std::vector<dtrian> dquad2dtrian(const dquad &Q);

#endif

