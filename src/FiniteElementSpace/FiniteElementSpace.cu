/*
*	FiniteElementSpace.cu
*/

#include "FiniteElementSpace.h"

FiniteElementSpace::FiniteElementSpace(){}
FiniteElementSpace::~FiniteElementSpace(){}

FiniteElementSpace::FiniteElementSpace(TriangleMesh t, FiniteElement f, Gauss g)
{
	T = t;
	finiteElement = f;
	gauss = g;
	buildFiniteElementSpace();
}

void FiniteElementSpace::buildEdge(){}

BaseFunction FiniteElementSpace::getBaseFunction(int i, int n)
{
	BaseFunction b;
	b.x = [i,n,this](const dvec &x){
		for(int m : T.q2t(n))
		{
			if(serial_accurate(x,T.mt.T[m]))
				return baseFunction[i].x(T.Binv[m]*(x-T.b[m]));
		}
	};
	b.dx = [i,n,this](const dvec &x){return baseFunction[i].dx(T.Binv[n]*(x-T.b[n]))*T.Binv[n];};
	b.f = {b.x,b.dx};
	b.i = getIndex(i, n);
	b.mini_i = getMiniIndex(i, n);
	return b;
}

int FiniteElementSpace::getIndex(int i, int n)
{
	int ii=i%(elementDim/ambientDim);
	int D=0;
  
	for(int j=0;j<ambientDim;++j)
	{
		if(i<(j+1)*elementDim/ambientDim){D=j*spaceDim/ambientDim;break;}
	}

	return nodes.T[n][ii]+D;
}

int FiniteElementSpace::getMiniIndex(int i, int n)
{
	int ii=i%(elementDim/ambientDim);
	int D=0;
  
	for(int j=0;j<ambientDim;++j)
	{
		if(i<(j+1)*elementDim/ambientDim){D=j*spaceDim/ambientDim;break;}
	}

	return find(notEdge,nodes.T[n][ii]+D);
}

void FiniteElementSpace::buildFiniteElementSpace()
{
	nodes = finiteElement.buildNodes(T.mesh);
	baseFunction = finiteElement.baseFunctions;
	for(int i=0;i<nodes.P.size();i++)
	{
		std::vector<int> temp;
		support.emplace_back(temp);
		for(int j=0;j<nodes.T.size();j++)
		{
			if(find(nodes.T[j],i)!=-1){support[i].push_back(j);}
		}
	}
	functions = buildFunctions(T.mesh,nodes,support,finiteElement);
	for(int n=0;n<baseFunction.size();n++)
	{
		std::vector<dvec> val_row;
		std::vector<dmat> Dval_row;
		val.push_back(val_row);
		Dval.push_back(Dval_row);
		for(int k=0;k<gauss.n;k++)
		{
			(val[n]).push_back(baseFunction[n].x(gauss.nodes[k]));
			(Dval[n]).push_back(baseFunction[n].dx(gauss.nodes[k]));
		}
	}
	this->setElementDim();
	this->setSpaceDim();
}

void FiniteElementSpace::setElementDim()
{
	this->elementDim = this->baseFunction.size();
}

void FiniteElementSpace::setSpaceDim()
{
	this->ambientDim = finiteElement.ambientDim;
	this->theOtherDim = this->nodes.P[0].size;
	this->spaceDim = this->nodes.P.size()*ambientDim;
}

F FiniteElementSpace::operator()(const std::vector<double> &v)
{
	LOG_WARNING("FiniteElementSpace::operator() not yet implemented!");
	std::function<dvec(dvec)> a = [&](const dvec &x)
	{
		return dvec({});
	};
	std::function<dmat(dvec)> Da = [&](const dvec &x)
	{
		return dmat({{}});
	};
	F Fa = {a,Da};
	return Fa;
}

F FiniteElementSpace::operator()(const std::vector<double> &v, int n)
{
	std::function<dvec(dvec)>  a = [&v,this,n](const dvec &x)
	{
		dvec y; y.size = ambientDim;
		for(int i=0;i<baseFunction.size();++i)
		{
			y += v[getIndex(i,n)]*baseFunction[i].x(T.Binv[n]*(x-T.b[n]));
		}
		return y;
	};
	std::function<dmat(dvec)>  Da = [&v,this,n](const dvec &x)
	{
		dmat y; y.rows = ambientDim; y.cols = theOtherDim;
		for(int i=0;i<baseFunction.size();++i)
		{
			y = y + v[getIndex(i,n)]*baseFunction[i].dx(T.Binv[n]*(x-T.b[n]))*T.Binv[n];
		}
		return y;
	};
	F Fa = {a,Da};
	return Fa;
}

void FiniteElementSpace::calc(const std::vector<double> &v)
{
	preCalc.clear();
	for(int n=0;n<nodes.T.size();++n)
	{
		std::map<dvec,xDx> temp;
		preCalc.push_back(temp);
		for(int k=0;k<gauss.n;k++)
		{
			dvec y; y.size = ambientDim;
			dmat z; z.rows = ambientDim; z.cols = theOtherDim;
	
			for(int i=0;i<baseFunction.size();++i)
			{
				y += v[getIndex(i,n)]*baseFunction[i].x(gauss.nodes[k]);
				z = z + v[getIndex(i,n)]*baseFunction[i].dx(gauss.nodes[k])*T.Binv[n];
			}
			dvec x = T.B[n]*gauss.nodes[k]+T.b[n];
			xDx w = {y,z};
			preCalc[n][x] = w;
		}
	}
}

F FiniteElementSpace::getPreCalc(int n)
{
	std::function<dvec(dvec)>  a = [&](const dvec &x)
	{
		std::map<dvec, xDx>::iterator i = preCalc[n].find(x);

		if(i!=preCalc[n].end())
		{
			return preCalc[n][x].x;
		}
		else
		{
			dvec v; v.size = x.size; //TODO: non è vero
			return  v;
		}
	};

	std::function<dmat(dvec)>  Da = [&](const dvec &x)
	{
		std::map<dvec, xDx>::iterator i = preCalc[n].find(x);
		if(i!=preCalc[n].end())
		{
			return preCalc[n][x].dx;
		}
		else
		{
			dmat v; v.rows = x.size; v.cols = x.size; //TODO: non è vero
			return v;
		}
	};
	return {a,Da};
}

std::vector<std::vector<int>> FiniteElementSpace::collisionDetection(const std::vector<std::vector<dvec>> &X)
{
	std::vector<std::vector<int>> MM;
	int size = X.size()*X[0].size();

	int q = nodes.T.size()/MAX_BLOCKS;
	int mod = nodes.T.size()%MAX_BLOCKS;

	bool N[q][X.size()][X[0].size()][MAX_BLOCKS];
	bool Nq[X.size()][X[0].size()][mod];
	std::vector<dvec> XX;
	for(int i=0;i<X.size();++i)
	{
		for(int j=0;j<X[i].size();++j)
		{
			XX.push_back(X[i][j]);
		}
	}
	dvec *devX;
	bool *devN[q];
	bool *devNq;

	HANDLE_ERROR(cudaMalloc((void**)&devX,size*sizeof(dvec)));
	HANDLE_ERROR(cudaMemcpy(devX,XX.data(),size*sizeof(dvec),cudaMemcpyHostToDevice));

	for(int i=0;i<q;++i)
	{
		HANDLE_ERROR(cudaMalloc((void**)&devN[i],size*MAX_BLOCKS*sizeof(bool)));
	}
	if(mod>0)
	{
		HANDLE_ERROR(cudaMalloc((void**)&devNq,size*mod*sizeof(bool)));
	}

	for(int i=0;i<q;++i)
	{
		parallel_accurate<<<size,MAX_BLOCKS>>>(T.devP,T.devT[i],devX,devN[i]);
		HANDLE_ERROR(cudaMemcpy(N[i],devN[i],size*MAX_BLOCKS*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	if(mod>0)
	{
		parallel_accurate<<<size,mod>>>(T.devP,T.devTq,devX,devNq);
		HANDLE_ERROR(cudaMemcpy(Nq,devNq,size*mod*sizeof(bool),cudaMemcpyDeviceToHost));
	}

	//std::cout << "serial " << serial_accurate(T.mesh.P.data(),T.mesh.T[i],X[0]) << std::endl;

	cudaFree(devX);
	for(int i=0;i<q;++i)
	{
		cudaFree(devN[i]);
	}
	if(mod>0)
	{
		cudaFree(devNq);
	}

	for(int n=0;n<X.size();++n)
	{
		std::vector<int> temp;
		MM.push_back(temp);
		for(int k=0;k<X[n].size();++k)
		{
			for(int i=0;i<q;++i)
			{
				for(int m=0;m<MAX_BLOCKS;++m)
				{
					if(N[i][n][k][m]==1)
					{
						if(find(MM[n],m+i*MAX_BLOCKS)==-1)
							MM[n].push_back(m+i*MAX_BLOCKS);
						break;
					}
				}
			}
		}
	}

	for(int n=0;n<X.size();++n)
	{
		std::vector<int> temp;
		MM.push_back(temp);
		for(int k=0;k<X[n].size();++k)
		{
			for(int m=0;m<mod;++m)
			{
				if(Nq[n][k][m]==1)
				{
					if(find(MM[n],m+q*MAX_BLOCKS)==-1)					
						MM[n].push_back(m+q*MAX_BLOCKS);
					break;
				}
			}
		}
	}
	return MM;
}

std::vector<int> FiniteElementSpace::collisionDetection(const std::vector<dvec> &X)
{
	std::vector<int> MM;
	int size = X.size();

	int q = nodes.T.size()/MAX_BLOCKS;
	int mod = nodes.T.size()%MAX_BLOCKS;

	bool N[q][X.size()][MAX_BLOCKS];
	bool Nq[X.size()][mod];

	dvec *devX;
	bool *devN[q];
	bool *devNq;

	HANDLE_ERROR(cudaMalloc((void**)&devX,size*sizeof(dvec)));
	HANDLE_ERROR(cudaMemcpy(devX,X.data(),size*sizeof(dvec),cudaMemcpyHostToDevice));

	for(int i=0;i<q;++i)
	{
		HANDLE_ERROR(cudaMalloc((void**)&devN[i],size*MAX_BLOCKS*sizeof(bool)));
	}
	if(mod>0)
	{
		HANDLE_ERROR(cudaMalloc((void**)&devNq,size*mod*sizeof(bool)));
	}

	for(int i=0;i<q;++i)
	{
		parallel_accurate<<<size,MAX_BLOCKS>>>(T.devP,T.devT[i],devX,devN[i]);
		HANDLE_ERROR(cudaMemcpy(N[i],devN[i],size*MAX_BLOCKS*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	if(mod>0)
	{
		parallel_accurate<<<size,mod>>>(T.devP,T.devTq,devX,devNq);
		HANDLE_ERROR(cudaMemcpy(Nq,devNq,size*mod*sizeof(bool),cudaMemcpyDeviceToHost));
	}

	//std::cout << "serial " << serial_accurate(T.mesh.P.data(),T.mesh.T[i],X[0]) << std::endl;

	cudaFree(devX);
	for(int i=0;i<q;++i)
	{
		cudaFree(devN[i]);
	}
	if(mod>0)
	{
		cudaFree(devNq);
	}
	for(int n=0;n<X.size();++n)
	{
		for(int i=0;i<q;++i)
		{
			for(int m=0;m<MAX_BLOCKS;++m)
			{
				if(N[i][n][m]==1)
				{
					MM.push_back(m+i*MAX_BLOCKS);
					break;
				}
			}
		}
	}

	for(int n=0;n<X.size();++n)
	{
		for(int m=0;m<mod;++m)
		{
			if(Nq[n][m]==1)
			{
				MM.push_back(m+q*MAX_BLOCKS);
				break;
			}
		}
	}
	return MM;
}

std::vector<dvec> FiniteElementSpace::getValuesInMeshNodes(const std::vector<double> &a)
{
	std::vector<dvec> x;
	x.reserve(nodes.P.size());
	x.resize(nodes.P.size());
	for(int n=0;n<nodes.T.size();++n)
	{
		for(int i=0;i<nodes.T[n].size();++i)
		{
			x[nodes.T[n][i]] = ((*this)(a,n).x(nodes.P[nodes.T[n][i]])); //TODO
		}
	}
	return x;

}

std::vector<std::vector<dvec>> FiniteElementSpace::getValuesInGaussNodes(const std::vector<double> &a)
{
	std::vector<std::vector<dvec>> x;
	for(int n=0;n<T.mesh.T.size();++n)
	{
		std::vector<dvec> xx;
		x.push_back(xx);
		for(int k=0;k<T.gauss.n;++k)
		{
			x[n].push_back((*this)(a,n).x(T.B[n]*T.gauss.nodes[k]+T.b[n]));
		}
	}
	return x;
}

Eigen::SparseMatrix<double> FiniteElementSpace::gC(const Eigen::SparseMatrix<double>& S)
{
	return getColumns(S,edge);
}

Eigen::SparseMatrix<double> FiniteElementSpace::gR(const Eigen::SparseMatrix<double>& S)
{
	return getRows(S,edge);
}

miniFE finiteElementSpace2miniFE(const FiniteElementSpace &finiteElementSpace)
{
	miniFE m;
	m.finiteElement = finiteElementSpace.finiteElement.finiteElementName;
	m.gauss = finiteElementSpace.gauss.gaussName;
	m.mesh.P = finiteElementSpace.T.mesh.P;
	m.mesh.T = finiteElementSpace.T.mesh.T;
	m.mesh.E = finiteElementSpace.T.mesh.E;
	return m;
}

FiniteElementSpace miniFE2FiniteElementSpace(const miniFE &mini, GaussService &gaussService, FiniteElementService &finiteElementService)
{
	FiniteElementSpace finiteElementSpace;
	TriangleMesh triangleMesh = TriangleMesh(mini.mesh,gaussService.getGauss(mini.gauss));
	triangleMesh.loadOnGPU();
	finiteElementSpace = FiniteElementSpace(triangleMesh,finiteElementService.getFiniteElement(mini.finiteElement),gaussService.getGauss(mini.gauss));
	finiteElementSpace.buildFiniteElementSpace();
	finiteElementSpace.buildEdge();
	return finiteElementSpace;
}

Eigen::SparseMatrix<double> FiniteElementSpace::applyEdgeCondition(const Eigen::SparseMatrix<double>& S)
{
	if(edge.size()>0)
		return S + gC(S)*E;
	else
		return S;
}

Eigen::SparseMatrix<double> compress(const Eigen::SparseMatrix<double> &S, const FiniteElementSpace &E, const FiniteElementSpace &F)
{
	return getRows(getColumns(S,E.notEdge),F.notEdge);
	//C*A*Eigen::SparseMatrix<double>(C.transpose())
	//C*a
}

