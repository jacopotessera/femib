/*
*	createGauss.h
*/

#ifndef CREATEGAUSS_H_INCLUDED_
#define CREATEGAUSS_H_INCLUDED_

#include "Gauss.h"

Gauss createGauss1_1d()
{
	dvec v1 = {0.5};
	double p1 = {1.0};

	std::vector<dvec> v = {v1};
	std::vector<double> p = {p1};

	return Gauss("gauss1_1d",1,v,p,1.0);
}

Gauss createGauss2_1d()
{
	dvec v1 = {0.5+0.5*(-sqrt(1.0/3.0))};
	dvec v2 = {0.5+0.5*( sqrt(1.0/3.0))};
	double p1 = 0.5;
	double p2 = 0.5;

	std::vector<dvec> v = {v1,v2};
	std::vector<double> p = {p1,p2};

	return Gauss("gauss2_1d",2,v,p,1.0);
}

Gauss createGauss3_1d()
{
	dvec v1 = {0.5};
	dvec v2 = {0.5+0.5*(-sqrt(3.0/5.0))};
	dvec v3 = {0.5+0.5*( sqrt(3.0/5.0))};
	double p1 = 8.0/9.0;
	double p2 = 5.0/9.0;
	double p3 = 5.0/9.0;

	std::vector<dvec> v = {v1,v2,v3};
	std::vector<double> p = {p1,p2,p3};

	return Gauss("gauss3_1d",3,v,p,1.0);
}

Gauss createGauss4_1d()
{
	dvec v1 = {0.5+0.5*(-sqrt(3.0/7.0-2.0/7.0*sqrt(6.0/5.0)))};
	dvec v2 = {0.5+0.5*( sqrt(3.0/7.0-2.0/7.0*sqrt(6.0/5.0)))};
	dvec v3 = {0.5+0.5*(-sqrt(3.0/7.0+2.0/7.0*sqrt(6.0/5.0)))};
	dvec v4 = {0.5+0.5*( sqrt(3.0/7.0+2.0/7.0*sqrt(6.0/5.0)))};
	double p1 = (18.0+sqrt(30.0))/36.0;
	double p2 = (18.0+sqrt(30.0))/36.0;
	double p3 = (18.0-sqrt(30.0))/36.0;
	double p4 = (18.0-sqrt(30.0))/36.0;

	std::vector<dvec> v = {v1,v2,v3,v4};
	std::vector<double> p = {p1,p2,p3,p4};

	return Gauss("gauss4_1d",4,v,p,1.0);
}

Gauss createGauss5_1d()
{
	dvec v1 = {0.5};
	dvec v2 = {0.5+0.5*(1.0/3.0)*(-sqrt(5.0-2.0*sqrt(10.0/7.0)))};
	dvec v3 = {0.5+0.5*(1.0/3.0)*( sqrt(5.0-2.0*sqrt(10.0/7.0)))};
	dvec v4 = {0.5+0.5*(1.0/3.0)*(-sqrt(5.0+2.0*sqrt(10.0/7.0)))};
	dvec v5 = {0.5+0.5*(1.0/3.0)*( sqrt(5.0+2.0*sqrt(10.0/7.0)))};
	double p1 = (128.0/225.0);
	double p2 = (322.0+13.0*sqrt(70.0))/900.0;
	double p3 = (322.0+13.0*sqrt(70.0))/900.0;
	double p4 = (322.0-13.0*sqrt(70.0))/900.0;
	double p5 = (322.0-13.0*sqrt(70.0))/900.0;

	std::vector<dvec> v = {v1,v2,v3,v4,v5};
	std::vector<double> p = {p1,p2,p3,p4,p5};

	return Gauss("gauss5_1d",5,v,p,1.0);
}

Gauss createGauss1_2d()
{
	dvec v1 = {{1.0/3.0},{1.0/3.0}};
	double p1 = 1.0;

	std::vector<dvec> v = {v1};
	std::vector<double> p = {p1};

	return Gauss("gauss1_2d",1,v,p,0.5);
}

Gauss createGauss2_2d()
{
	dvec v1 = {{1.0/6.0},{1.0/6.0}};
	dvec v2 = {{1.0/6.0},{2.0/3.0}};
	dvec v3 = {{2.0/3.0},{1.0/6.0}};
	double p1 = 1.0/6.0;
	double p2 = 1.0/6.0;
	double p3 = 1.0/6.0;

	std::vector<dvec> v = {v1,v2,v3};
	std::vector<double> p = {p1,p2,p3};

	return Gauss("gauss2_2d",3,v,p,0.5);
}

Gauss createGauss3_2d()
{
	dvec v1 = {{1.0/3.0},{1.0/3.0}};
	dvec v2 = {{1.0/5.0},{1.0/5.0}};
	dvec v3 = {{1.0/5.0},{6.0/10.0}};
	dvec v4 = {{6.0/10.0},{1.0/5.0}};
	double p1 = -0.5*9.0/16.0;
	double p2 = 0.5*25.0/48.0;
	double p3 = 0.5*25.0/48.0;
	double p4 = 0.5*25.0/48.0;

	std::vector<dvec> v = {v1,v2,v3,v4};
	std::vector<double> p = {p1,p2,p3,p4};

	return Gauss("gauss3_2d",4,v,p,0.5);
}

Gauss createGauss5_2d()
{
	dvec v1 = {{1.0/3.0},{1.0/3.0}};
	dvec v2 = {{0.47014206410511},{0.47014206410511}};
	dvec v3 = {{0.47014206410511},{0.05971587178977}};
	dvec v4 = {{0.05971587178977},{0.47014206410511}};
	dvec v5 = {{0.10128650732346},{0.10128650732346}};
	dvec v6 = {{0.10128650732346},{0.79742698535309}};
	dvec v7 = {{0.79742698535309},{0.10128650732346}};
	double p1 = 0.225/2;
	double p2 = 0.13239415278851/2;
	double p3 = 0.13239415278851/2;
	double p4 = 0.13239415278851/2;
	double p5 = 0.12593918054483/2;
	double p6 = 0.12593918054483/2;
	double p7 = 0.12593918054483/2;

	std::vector<dvec> v = {v1,v2,v3,v4,v5,v6,v7};
	std::vector<double> p = {p1,p2,p3,p4,p5,p6,p7};

	return Gauss("gauss5_2d",7,v,p,0.5);
}

#endif

