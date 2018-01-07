/*
*	FiniteElement.cu
*/

#include "FiniteElement.h"

FiniteElement::FiniteElement(){}
FiniteElement::~FiniteElement(){}

[[deprecated]]
std::vector<F> buildFunctions(const Mesh &mesh, const Nodes &nodes,const std::vector<std::vector<int> > &support, const FiniteElement &f)
{
	std::vector<F> functions;
	for(int d=0;d<nodes.P[0].size;++d){
		for(int i=0;i<nodes.P.size();++i) //TODO: mancano una fila di funzioni
		{
			F function;

			function.x = [d,i,&mesh,&nodes,&support,&f](const dvec &x)
			{
				for(int j=0;j<support[i].size();++j)
				{
					dtrian t = ditrian2dtrian(mesh.T[j],mesh.P.data());
					if(in_triangle(x,t))
					{
						return f.baseFunctions[find(nodes.T[support[i][j]],i)].x(pinv(affineB(support[i][j],mesh))*(x-affineb(support[i][j],mesh)));
					}
				}
				dvec v; v.size=mesh.P[0].size;
				return v;
				return x;
			};

			function.dx = [d,i,&mesh,&nodes,&support,&f](const dvec &x)
			{
				for(int j=0;j<support[i].size();++j)
				{
					dtrian t = ditrian2dtrian(mesh.T[j],mesh.P.data());
					if(in_triangle(x,t))
					{
						return f.baseFunctions[find(nodes.T[support[i][j]],i)].dx(pinv(affineB(support[i][j],mesh))*(x-affineb(support[i][j],mesh)))*pinv(affineB(support[i][j],mesh)); //*affine^(-1);
					}
				}
				dmat v; v.rows=mesh.P[0].size; v.cols=mesh.P[0].size;
				return v;
			};

			functions.emplace_back(function);
		}
	}
	return functions;
}

bool FiniteElement::check()
{
	bool e = true;
	for(int i=0;e && i<stdNodes.size();i++)
	{
		for(int j=0;e && j<baseFunctions.size();j++)
		{
			e = e && ( project(baseFunctions[j].x,j/stdNodes[0].size)(stdNodes[i]) == (i==j ? 1 : 0) );
		}
	}
	return e;
}


