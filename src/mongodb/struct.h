/*
*	struct.h
*/

#ifndef STRUCT_H_INCLUDED_
#define STRUCT_H_INCLUDED_

struct Parameters
{
	double rho;
	double eta;
	double deltarho;
	double kappa;
	double deltat;
	int steps;
	double TMAX;
};

struct miniFE
{
	std::string finiteElement;
	std::string gauss;
	Mesh mesh;
};

struct miniSim
{
	std::string id;
	std::string date;
	bool full;
	Parameters parameters;
	miniFE V;
	miniFE Q;
	miniFE S;
	miniFE L;
};

struct timestep
{
	std::string id;
	std::string date;
	int time;
	std::vector<double> u;
	std::vector<double> q;
	std::vector<double> x;
	std::vector<double> l;
};

struct plotData
{
	std::string id;
	std::string date;
	int time;
	std::vector<std::vector<std::vector<double>>> u;
	std::vector<std::vector<std::vector<double>>> q;
	std::vector<std::vector<std::vector<double>>> x;
};

#endif

