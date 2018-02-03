/*
*	dmat_impl.h
*/

#ifndef DMAT_IMPL_H_INCLUDED_
#define DMAT_IMPL_H_INCLUDED_

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "dmat.h"
#include "../../lib/Log.h"

#define M_PI           3.14159265358979323846
#define M_EPS          0.000001
#define M_EPS2         M_EPS*M_EPS

std::ostream& operator<<(std::ostream& out, const dvec &v);
std::ostream& operator<<(std::ostream& out, const divec &v);
std::ostream& operator<<(std::ostream& out, const dmat &A);
std::ostream& operator<<(std::ostream& out, const dtrian &s);
std::ostream& operator<<(std::ostream& out, const ditrian &t);

__host__ __device__ double _abs(double x);
__host__ __device__ bool operator==(const dvec &lhs, const dvec &rhs);
__host__ __device__ bool operator<(const dvec &lhs, const dvec &rhs);
__host__ __device__ bool operator==(const divec &lhs, const divec &rhs);
__host__ __device__ bool operator==(const dmat &lhs, const dmat &rhs);
__host__ __device__ bool operator==(const dtrian lhs, dtrian rhs);
__host__ __device__ bool operator==(const ditrian lhs, ditrian rhs);
__host__ __device__ dvec operator*(double a, const dvec &v);
__host__ __device__ dmat operator*(double a, const dmat &B);
__host__ __device__ dvec operator+(const dvec &a, const dvec &b);
__host__ __device__ dmat operator+(const dmat &A, const dmat &B);
__host__ __device__ dvec operator-(const dvec &a, const dvec &b);
__host__ __device__ dmat operator-(const dmat &A, const dmat &B);
__host__ __device__ dmat operator*(const dmat &A, const dmat &B);
__host__ __device__ dvec operator*(const dmat &A, const dvec &v);

__host__ __device__ double ddot(const dvec &a, const dvec &b);
__host__ __device__ dvec dhat(const dvec &a, const dvec &b);
__host__ __device__ double ddet(const dmat &A);
__host__ __device__ double dtrace(const dmat &A);
__host__ __device__ double dnorm(const dmat &A);
__host__ __device__ dmat inv(const dmat &A);
__host__ __device__ dmat pinv(const dmat &A);
__host__ __device__ dmat trans(const dmat &A);

__host__ __device__ double dmax(const dtrian &A, int j);
__host__ __device__ double dmin(const dtrian &A, int j);

__host__ __device__ bool in_box(const dvec &P, const dtrian &T);
__host__ __device__ bool in_std(const dvec &P);
__host__ __device__ bool in_triangle(const dvec &P, const dtrian &T);

__host__ __device__ double distancePointPoint(const dvec &P, const dvec &Q);
__host__ __device__ double distancePointSegment(const dvec &P, const dtrian &T);
__host__ __device__ double distancePointTriangle(const dvec &P, const dtrian &T);

__host__ __device__ bool accurate(const dvec &P, const dtrian &T);
__global__ void parallel_accurate(dvec *P,ditrian *T,dvec *X,bool *N);
__host__ bool serial_accurate(dvec *P,ditrian T,dvec X);

__host__ __device__ dtrian ditrian2dtrian(const ditrian &T, const dvec *P);
__host__ __device__ dmat dvec2dmat(dvec *P);

#endif

