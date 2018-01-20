/*
*	utils.h
*/

#ifndef UTILS_H_INCLUDED_
#define UTILS_H_INCLUDED_

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>

#include "../dmat/dmat.h"
#include "../dmat/dmat_impl.h"

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> edmat;
typedef std::vector<Eigen::Triplet<double>> etmat;
typedef Eigen::SparseMatrix<double> esmat;
typedef Eigen::Matrix<double,Eigen::Dynamic,1> evec;

std::vector<int> setdiff(std::vector<int> x, const std::vector<int> &y);
std::vector<int> operator+(std::vector<int> x, int y);
std::vector<int> join(std::vector<int> x, const std::vector<int> &y);
std::vector<int> linspace(int a);
int find(const std::vector<int> &x, int i);
int find(const std::vector<dvec> &x, const dvec &v);
std::vector<double> dvec2vector(const dvec &v);
std::vector<int> dtrian2vector(const dtrian &t);

std::vector<double> operator+(const std::vector<double> &A, const std::vector<double> &B);
std::vector<double> operator*(double a, const std::vector<double> &B);

std::vector<double> eigen2vector(const Eigen::Matrix<double,Eigen::Dynamic,1> &v);
Eigen::Matrix<double,Eigen::Dynamic,1> vector2eigen(const std::vector<double> &v);

Eigen::SparseMatrix<double> getColumns(Eigen::SparseMatrix<double> S, const std::vector<int> &x);
Eigen::SparseMatrix<double> getRows(Eigen::SparseMatrix<double> S, const std::vector<int> &x);
evec getRows(evec S, const std::vector<int> &x);
etmat esmat2etmat(const esmat& A);
etmat esmat2etmat(const esmat& A, int rowDrift, int colDrift);
esmat etmat2esmat(const etmat &A, int rows, int cols);
etmat transpose(const etmat &A);
etmat& operator+=(etmat& lhs, const etmat &rhs);
std::ostream& operator<<(std::ostream& out, const etmat &t);

std::string getTimestamp();
std::vector<double> join(evec a, evec b, std::vector<int> ne, std::vector<int> e);
std::ostream& operator<<(std::ostream& out, const std::vector<double> &T);

#endif

