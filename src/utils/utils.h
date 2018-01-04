/*
*	utils.h
*/

#ifndef UTILS_H_INCLUDED_
#define UTILS_H_INCLUDED_

#include <algorithm>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

#include "../dmat/dmat.h"
#include "../dmat/dmat_impl.h"

std::vector<int> setdiff(std::vector<int> x, const std::vector<int> &y);
std::vector<int> operator+(std::vector<int> x, int y);
std::vector<int> join(std::vector<int> x, const std::vector<int> &y);
std::vector<int> linspace(int a);
int find(const std::vector<int> &x, int i);
int find(const std::vector<dvec> &x, const dvec &v);
Eigen::SparseMatrix<double> compress(int dim, const std::vector<int> &x);
std::vector<double> dvec2vector(const dvec &v);
std::vector<int> dtrian2vector(const dtrian &t);

std::vector<double> operator+(const std::vector<double> &A, const std::vector<double> &B);
std::vector<double> operator*(double a, const std::vector<double> &B);

std::vector<double> eigen2vector(const Eigen::Matrix<double,Eigen::Dynamic,1> &v);
Eigen::Matrix<double,Eigen::Dynamic,1> vector2eigen(const std::vector<double> &v);

#endif

