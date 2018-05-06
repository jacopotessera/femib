/*
*	createFiniteELement.h
*/

#ifndef CREATEFINITEELEMENT_H_INCLUDED_
#define CREATEFINITEELEMENT_H_INCLUDED_

#include "../utils/utils.h"
#include "FiniteElement.h"

dvec findCenterOf(const Mesh &mesh, int n)
{
	dvec c;
	c.size = mesh.P[0].size;
	double k = 1.0/mesh.T[n].size;
	for(int j=0;j<mesh.T[n].size;++j)
	{
		c = c + k*mesh.P[mesh.T[n](j)];
	}
	return c;
}

FiniteElement createFiniteElementP0_2d2d()
{
	F f;
	FiniteElement P0_2d2d;
	P0_2d2d.finiteElementName = "P0_2d2d";
	P0_2d2d.size = 2;
	P0_2d2d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes;
		nodes.P = mesh.P;
		for(int n=0;n<mesh.T.size();++n)
		{
			std::vector<int> row;
			nodes.T.push_back(row);
			for(int i=0;i<mesh.T[n].size;++i)
			{
				nodes.T[n].push_back(mesh.T[n](i));
			}
		}
		nodes.E = mesh.E;
		return nodes;
	};

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({1,0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		return dmat({{0,0},{0,0}});
	};

	P0_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,1});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		return dmat({{0,0},{0,0}});
	};

	P0_2d2d.baseFunctions.push_back(f);

	return P0_2d2d;
}

FiniteElement createFiniteElementP1_2d2d()
{
	F f;
	FiniteElement P1_2d2d;
	P1_2d2d.finiteElementName = "P1_2d2d";
	P1_2d2d.size = 6;
	P1_2d2d.ambientDim = 2;
	P1_2d2d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes;
		nodes.P = mesh.P;
		for(int n=0;n<mesh.T.size();++n)
		{
			std::vector<int> row;
			nodes.T.push_back(row);
			for(int i=0;i<mesh.T[n].size;++i)
			{
				nodes.T[n].push_back(mesh.T[n](i));
			}
		}
		nodes.E = mesh.E;
		return nodes;
	};

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({1-x(0)-x(1),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{-1,-1},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P1_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(0),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{1,0},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P1_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(1),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,1},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P1_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,1-x(0)-x(1)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{-1,-1}});
		else
			return dmat({{0,0},{0,0}});
	};

	P1_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,x(0)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{1,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P1_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,x(1)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{0,1}});
		else
			return dmat({{0,0},{0,0}});
	};

	P1_2d2d.baseFunctions.push_back(f);

	return P1_2d2d;
}

FiniteElement createFiniteElementP0_2d1d()
{
	F f;
	FiniteElement P0_2d1d;
	P0_2d1d.finiteElementName = "P0_2d1d";
	P0_2d1d.size = 1;
	P0_2d1d.ambientDim = 1;
	P0_2d1d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes;
		for(int n=0;n<mesh.T.size();++n)
		{
			std::vector<int> row;
			nodes.T.push_back(row);
			nodes.P.push_back(findCenterOf(mesh,n));
			nodes.T[n].push_back(n);
		}
		nodes.E = mesh.E;
		return nodes;
	};

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({1});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		return dmat({{0,0}});
	};

	P0_2d1d.baseFunctions.push_back(f);

	return P0_2d1d;
}

FiniteElement createFiniteElementP1_2d1d()
{
	F f;
	FiniteElement P1_2d1d;
	P1_2d1d.finiteElementName = "P1_2d1d";
	P1_2d1d.size = 3;
	P1_2d1d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes;
		nodes.P = mesh.P;
		for(int n=0;n<mesh.T.size();++n)
		{
			std::vector<int> row;
			nodes.T.push_back(row);
			for(int i=0;i<mesh.T[n].size;++i)
			{
				nodes.T[n].push_back(mesh.T[n](i));
			}
		}
		nodes.E = mesh.E;
		return nodes;
	};

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({1-x(0)-x(1)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{-1,-1}});
		else
			return dmat({{0,0}});
	};

	P1_2d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(0)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{1,0}});
		else
			return dmat({{0,0}});
	};

	P1_2d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(1)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,1}});
		else
			return dmat({{0,0}});
	};

	P1_2d1d.baseFunctions.push_back(f);

	return P1_2d1d;
}

Nodes mesh2nodes(const Mesh &mesh)
{
	Nodes nodes;
	nodes.P = mesh.P;
	for(int n=0;n<mesh.T.size();++n)
	{
		std::vector<int> row;
		nodes.T.push_back(row);
		for(int i=0;i<mesh.T[n].size;++i)
		{
			nodes.T[n].push_back(mesh.T[n](i));
		}
	}
	nodes.E = mesh.E;
	return nodes;
}

FiniteElement createFiniteElementP1P0_2d1d()
{
	F f;
	FiniteElement P1P0_2d1d;
	P1P0_2d1d.finiteElementName = "P1P0_2d1d";
	P1P0_2d1d.size = 4;
	P1P0_2d1d.ambientDim = 1;
	P1P0_2d1d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes = mesh2nodes(mesh);
		for(int n=0;n<mesh.T.size();++n)
		{
			nodes.P.push_back(findCenterOf(mesh,n));
			nodes.T[n].push_back(nodes.P.size()-1);
		}
		return nodes;
	};
	
	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({1-x(0)-x(1)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{-1,-1}});
		else
			return dmat({{0,0}});
	};

	P1P0_2d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(0)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{1,0}});
		else
			return dmat({{0,0}});
	};

	P1P0_2d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(1)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,1}});
		else
			return dmat({{0,0}});
	};

	P1P0_2d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{

		if(in_std(x))
			return dvec({1.0});
		
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		return dmat({{0,0}});
	};

	P1P0_2d1d.baseFunctions.push_back(f);
	
	return P1P0_2d1d;
}

FiniteElement createFiniteElementP2_2d2d()
{
	F f;
	FiniteElement P2_2d2d;
	P2_2d2d.finiteElementName = "P2_2d2d";
	P2_2d2d.size = 12;
	P2_2d2d.ambientDim = 2;
	P2_2d2d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes = mesh2nodes(mesh);
		
		for(int n=0;n<mesh.T.size();n++)
		{
			std::vector<std::vector<int>> lines = {{mesh.T[n](0),mesh.T[n](1)},{mesh.T[n](1),mesh.T[n](2)},{mesh.T[n](2),mesh.T[n](0)}};
			for(int i=0;i<lines.size();i++)
			{
				dvec puntoMedio = 0.5*(mesh.P[lines[i][0]]+mesh.P[lines[i][1]]);
				int k = find(nodes.P,puntoMedio);
				if(k==-1)
				{
					nodes.P.push_back(puntoMedio);
					nodes.T[n].push_back(nodes.P.size()-1);
					if(find(mesh.E,lines[i][0])!=-1 && find(mesh.E,lines[i][1])!=-1)
					{
						nodes.E.push_back(nodes.P.size()-1);
					}
				}
				else
				{
					nodes.T[n].push_back(k);
				}
			}
		}
		return nodes;
	};

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({2*pow(x(0),2.0)+2*pow(x(1),2.0)+4*x(0)*x(1)-3*x(0)-3*x(1)+1,0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{4*x(0)+4*x(1)-3,4*x(0)+4*x(1)-3},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({2*pow(x(0),2.0)-x(0),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{4*x(0)-1,0},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({2*pow(x(1),2.0)-x(1),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,4*x(1)-1},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};
	
	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({-4*pow(x(0),2.0)+4*x(0)-4*x(0)*x(1),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{-8*x(0)-4*x(1)+4,-4*x(0)},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({4*x(0)*x(1),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{4*x(1),4*x(0)},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);
	
		f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({-4*pow(x(1),2.0)+4*x(1)-4*x(0)*x(1),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{-4*x(1),-8*x(1)-4*x(0)+4},{0,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,2*pow(x(0),2.0)+2*pow(x(1),2.0)+4*x(0)*x(1)-3*x(0)-3*x(1)+1});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{4*x(0)+4*x(1)-3,4*x(0)+4*x(1)-3}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,2*pow(x(0),2.0)-x(0)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{4*x(0)-1,0}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,2*pow(x(1),2.0)-x(1)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{0,4*x(1)-1}});
		else
			return dmat({{0,0},{0,0}});
	};
	
	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,-4*pow(x(0),2.0)+4*x(0)-4*x(0)*x(1)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{-8*x(0)-4*x(1)+4,-4*x(0)}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,4*x(0)*x(1)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{4*x(1),4*x(0)}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);
	
		f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,-4*pow(x(1),2.0)+4*x(1)-4*x(0)*x(1)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0,0},{-4*x(1),-8*x(1)-4*x(0)+4}});
		else
			return dmat({{0,0},{0,0}});
	};

	P2_2d2d.baseFunctions.push_back(f);
	
	P2_2d2d.stdNodes = {{0,0},{1,0},{0,1},{0.5,0},{0.5,0.5},{0,0.5}};

	return P2_2d2d;
}

FiniteElement createFiniteElementP1_1d1d()
{
	F f;
	FiniteElement P1_1d1d;
	P1_1d1d.finiteElementName = "P1_1d1d";
	P1_1d1d.size = 2;
	P1_1d1d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes;
		nodes.P = mesh.P;
		for(int n=0;n<mesh.T.size();++n)
		{
			std::vector<int> row;
			nodes.T.push_back(row);
			for(int i=0;i<mesh.T[n].size;++i)
			{
				nodes.T[n].push_back(mesh.T[n](i));
			}
		}
		nodes.E = mesh.E;
		return nodes;
	};

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({1-x(0)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{-1}});
		else
			return dmat({{0.0}});
	};

	P1_1d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(0)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{1}});
		else
			return dmat({{0.0}});
	};

	P1_1d1d.baseFunctions.push_back(f);

	return P1_1d1d;
}

FiniteElement createFiniteElementP1_1d2d()
{
	F f;
	FiniteElement P1_1d2d;
	P1_1d2d.finiteElementName = "P1_1d2d";
	P1_1d2d.size = 4;
	P1_1d2d.ambientDim = 2;
	P1_1d2d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes;
		nodes.P = mesh.P;
		for(int n=0;n<mesh.T.size();++n)
		{
			std::vector<int> row;
			nodes.T.push_back(row);
			for(int i=0;i<mesh.T[n].size;++i)
			{
				nodes.T[n].push_back(mesh.T[n](i));
			}
		}
		nodes.E = mesh.E;
		return nodes;
	};

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({1-x(0),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{-1},{0.0}});
		else
			return dmat({{0.0},{0.0}});
	};

	P1_1d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({x(0),0});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{1},{0.0}});
		else
			return dmat({{0.0},{0.0}});
	};

	P1_1d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,1-x(0)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0.0},{-1}});
		else
			return dmat({{0.0},{0.0}});
	};

	P1_1d2d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std(x))
			return dvec({0,x(0)});
		else
			return dvec({0,0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std(x))
			return dmat({{0.0},{1}});
		else
			return dmat({{0.0},{0.0}});
	};

	P1_1d2d.baseFunctions.push_back(f);

	return P1_1d2d;
}

bool in_std_q(const dvec& x)
{
	Mesh m0;
	m0.P = {{0.0,0.0},{1.0,0.0},{1.0,1.0},{0.0,1.0}};
	m0.T = {{0,1,3},{2,3,1}};

	dmat A = inv(affineB(1,m0));
	dvec a = affineb(1,m0);
	return in_std(x) || in_std(A*(x-a));
}

FiniteElement createFiniteElementQ1_2d1d()
{
	F f;
	FiniteElement Q1_2d1d;
	Q1_2d1d.finiteElementName = "Q1_2d1d";
	Q1_2d1d.size = 4;
	Q1_2d1d.ambientDim = 1;
	Q1_2d1d.buildNodes = [](const Mesh &mesh)
	{
		Nodes nodes;
		nodes.P = mesh.P;
		for(int n=0;n<mesh.T.size();++n)
		{
			std::vector<int> row;
			nodes.T.push_back(row);
			for(int i=0;i<mesh.T[n].size;++i)
			{
				nodes.T[n].push_back(mesh.T[n](i));
			}
		}
		nodes.E = mesh.E;
		return nodes;
	};

	//inv({
	//{0,0,0,1},
	//{1,0,0,1},
	//{1,1,1,1},
	//{0, 1,0,1}
	//})*{0,1,0,0}

	f.x = [](const dvec &x)
	{
		if(in_std_q(x))
			return dvec({-x(0)-x(1)+x(0)*x(1)+1});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std_q(x))
			return dmat({{-1+x(1),-1+x(0)}});
		else
			return dmat({{0,0}});
	};

	Q1_2d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std_q(x))
			return dvec({x(0)-x(0)*x(1)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std_q(x))
			return dmat({{1-x(1),-x(0)}});
		else
			return dmat({{0,0}});
	};

	Q1_2d1d.baseFunctions.push_back(f);

	f.x = [](const dvec &x)
	{
		if(in_std_q(x))
			return dvec({x(0)*x(1)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std_q(x))
			return dmat({{x(1),x(0)}});
		else
			return dmat({{0,0}});
	};

	Q1_2d1d.baseFunctions.push_back(f);


	f.x = [](const dvec &x)
	{
		if(in_std_q(x))
			return dvec({x(1)-x(0)*x(1)});
		else
			return dvec({0.0});
	};

	f.dx = [](const dvec &x)
	{
		if(in_std_q(x))
			return dmat({{-x(1),1-x(0)}});
		else
			return dmat({{0,0}});
	};

	Q1_2d1d.baseFunctions.push_back(f);

	return Q1_2d1d;
}

#endif

