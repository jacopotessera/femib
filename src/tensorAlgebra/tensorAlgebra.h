/*
*	tensorAlgebra.h
*/

#ifndef TENSORALGEBRA_H_INCLUDED_
#define TENSORALGEBRA_H_INCLUDED_

#include <vector>
#include <functional>
#include <stdexcept>

#include "../dmat/dmat_impl.h"
#include "../utils/Mesh.h"

// divergenza
double div(const dmat &A);

// doppio prodotto interno
double dpi(const dmat &A, const dmat &B);

// prodotto scalare
double pf(const dmat &A, const dvec &B);
double pf(const dmat &A, const dmat &B);

// gradiente simmetrico
dmat symm(const dmat &A);

// dot-divergenza
dvec dotdiv(const dvec &b, const dmat &B);

dmat vec2mat(const dvec &b);

std::function<double(dvec)> div(const std::function<dmat(dvec)> &A);
std::function<double(dvec)> dpi(const std::function<dmat(dvec)> &A, const std::function<dmat(dvec)> &B);
std::function<double(dvec)> pf(const std::function<dmat(dvec)> &A, const std::function<dvec(dvec)> &B);
std::function<double(dvec)> pf(const std::function<dmat(dvec)> &A, const std::function<dmat(dvec)> &B);
std::function<dmat(dvec)> symm(const std::function<dmat(dvec)> &A);
std::function<dvec(dvec)> dotdiv(const std::function<dvec(dvec)> &b, const std::function<dmat(dvec)> &A);
std::function<double(dvec)> ddot(const std::function<dvec(dvec)> &a, const std::function<dvec(dvec)> &b);
std::function<double(dvec)> operator*(const std::function<double(dvec)> &a, const std::function<double(dvec)> &b);
std::function<double(dvec)> operator*(double a, const std::function<double(dvec)> &b);
std::function<dmat(dvec)> operator*(const std::function<dmat(dvec)> &a, const dmat &b);
std::function<dmat(dvec)> operator*(const std::function<dmat(dvec)> &a, const std::function<dmat(dvec)> &b);
std::function<double(dvec)> project(const std::function<dvec(dvec)> &a, int i);
std::function<double(dvec)> operator+(const std::function<double(dvec)> &a, const std::function<double(dvec)> &b);
std::function<double(dvec)> operator-(const std::function<double(dvec)> &a, const std::function<double(dvec)> &b);

template<typename T>
std::function<T(dvec)> constant(const T &c);

template<typename T, typename U, typename V>
std::function<T(V)> compose(const std::function<T(U)> &a, const std::function<U(V)> &b);

F compose(const F &a, const F &b);

#endif

