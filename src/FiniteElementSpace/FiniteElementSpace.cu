/*
*	FiniteElementSpace.cu
*/

#include "FiniteElementSpace.h"

template<MeshType meshType>
FiniteElementSpace<meshType>::FiniteElementSpace(){}

template<MeshType meshType>
FiniteElementSpace<meshType>::~FiniteElementSpace(){}

template<MeshType meshType>
FiniteElementSpace<meshType>::FiniteElementSpace(SimplicialMesh<meshType> t, FiniteElement f, Gauss g)
{
	T = t;
	finiteElement = f;
	gauss = g;
	buildFiniteElementSpace();
}

template<MeshType meshType>
void FiniteElementSpace<meshType>::buildEdge(){}

template<MeshType meshType>
BaseFunction FiniteElementSpace<meshType>::getBaseFunction(int i, int n)
{
	BaseFunction b;
	b.x = [i,n,this](const dvec &x){
		//sLOG_DEBUG("x: " << x);
		if(T.xInN(x,n))
		{
			dvec y = T.toMesh0x(x,n);
			//sLOG_DEBUG("y: " << y);
			return baseFunction[i].x(y);
		}
		else
		{
			dvec zero; zero.size = ambientDim;
			//sLOG_DEBUG("x not in n!");
			return zero;
		}
	};
	b.dx = [i,n,this](const dvec &x){
		if(T.xInN(x,n))
			return baseFunction[i].dx( T.toMesh0x(x,n) )* T.toMesh0dx(x,n);
		else
		{
			dmat zero; zero.rows = ambientDim; zero.cols = theOtherDim;
			return zero;
		}
	};
	b.f = {b.x,b.dx};
	b.i = getIndex(i, n);
	b.mini_i = getMiniIndex(i, n);
	return b;
}

template<MeshType meshType>
int FiniteElementSpace<meshType>::getIndex(int i, int n)
{
	int ii=i%(elementDim/ambientDim);
	int D=0;
  
	for(int j=0;j<ambientDim;++j)
	{
		if(i<(j+1)*elementDim/ambientDim){D=j*spaceDim/ambientDim;break;}
	}

	return nodes.T[n][ii]+D;
}

template<MeshType meshType>
int FiniteElementSpace<meshType>::getMiniIndex(int i, int n)
{
	int ii=i%(elementDim/ambientDim);
	int D=0;
  
	for(int j=0;j<ambientDim;++j)
	{
		if(i<(j+1)*elementDim/ambientDim){D=j*spaceDim/ambientDim;break;}
	}

	return find(notEdge,nodes.T[n][ii]+D);
}

template<MeshType meshType>
void FiniteElementSpace<meshType>::buildFiniteElementSpace()
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

template<MeshType meshType>
void FiniteElementSpace<meshType>::setElementDim()
{
	this->elementDim = this->baseFunction.size();
}

template<MeshType meshType>
void FiniteElementSpace<meshType>::setSpaceDim()
{
	this->ambientDim = finiteElement.ambientDim;
	this->theOtherDim = this->nodes.P[0].size;
	this->spaceDim = this->nodes.P.size()*ambientDim;
}

template<MeshType meshType>
F FiniteElementSpace<meshType>::operator()(const std::vector<double> &v)
{
	LOG_WARNING("FiniteElementSpace<meshType>::operator() not yet implemented!");
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

template<MeshType meshType>
F FiniteElementSpace<meshType>::operator()(const std::vector<double> &v, int n)
{
	std::function<dvec(dvec)>  a = [&v,this,n](const dvec &x)
	{
		dvec y; y.size = ambientDim;
		for(int i=0;i<baseFunction.size();++i)
		{
			if(T.xInN(x,n))
			{
				sLOG_DEBUG("x: " << x);
				dvec q = T.toMesh0x(x,n);
				sLOG_DEBUG("q: " << q);
				y += v[getIndex(i,n)]*baseFunction[i].x(q);
			}
		}
		return y;
	};
	std::function<dmat(dvec)>  Da = [&v,this,n](const dvec &x)
	{
		dmat y; y.rows = ambientDim; y.cols = theOtherDim;
		for(int i=0;i<baseFunction.size();++i)
		{
			if(T.xInN(x,n))
				y = y + v[getIndex(i,n)]*baseFunction[i].dx( T.toMesh0x(x,n) )*T.toMesh0dx(x,n);
		}
		return y;
	};
	F Fa = {a,Da};
	return Fa;
}

template<MeshType meshType>
void FiniteElementSpace<meshType>::calc(const std::vector<double> &v)
{
	preCalc.clear();
	for(int n=0;n<nodes.T.size();++n)
	{
		std::unordered_map<dvec,xDx> temp;
		preCalc.push_back(temp);
		for(int mm=0;mm<T.q2t[n].size();++mm)
		{
			int m = T.q2t[n][mm];
			for(int k=0;k<gauss.n;k++)
			{
				dvec y; y.size = ambientDim;
				dmat z; z.rows = ambientDim; z.cols = theOtherDim;

				for(int i=0;i<baseFunction.size();++i)
				{
					y += v[getIndex(i,n)]*baseFunction[i].x( affineB(mm,T.mesh0)*gauss.nodes[k]+affineb(mm,T.mesh0) );
					z = z + v[getIndex(i,n)]*baseFunction[i].dx( affineB(mm,T.mesh0)*gauss.nodes[k]+affineb(mm,T.mesh0) )* affineB(mm,T.mesh0);
				}
				dvec x = T.triangleMesh.B[m]* ( gauss.nodes[k] ) + T.triangleMesh.b[m];
				xDx w = {y,z};
				preCalc[n][x] = w;
			}
		}
	}
}

template<MeshType meshType>
F FiniteElementSpace<meshType>::getPreCalc(int n)
{
	std::function<dvec(dvec)>  a = [&,n](const dvec &x)
	{
		std::unordered_map<dvec, xDx>::iterator i = preCalc[n].find(x);

		if(i!=preCalc[n].end())
		{
			//sLOG_DEBUG("n: " << n);
			//sLOG_DEBUG("x: " << x);
			//sLOG_DEBUG("x: " << i->first);
			//sLOG_DEBUG("preCalc(n).x(x) = ");
			//sLOG_DEBUG("y: " << i->second.x);
			return i->second.x;
		}
		else
		{
			dvec v; v.size = x.size; //TODO: non è vero
			return  v;
		}
	};

	std::function<dmat(dvec)>  Da = [&,n](const dvec &x)
	{
		std::unordered_map<dvec, xDx>::iterator i = preCalc[n].find(x);
		if(i!=preCalc[n].end())
		{
			//sLOG_DEBUG("n: " << n);
			//sLOG_DEBUG("x: " << x);
			//sLOG_DEBUG("x: " << i->first);
			//sLOG_DEBUG("preCalc(n).dx(x) = ");
			//sLOG_DEBUG("dy: " << i->second.dx);
			return i->second.dx;
		}
		else
		{
			dmat v; v.rows = x.size; v.cols = x.size; //TODO: non è vero
			return v;
		}
	};
	return {a,Da};
}

template<MeshType meshType>
std::vector<std::vector<int>> FiniteElementSpace<meshType>::collisionDetection(const std::vector<std::vector<dvec>> &X)
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
		parallel_accurate<<<size,MAX_BLOCKS>>>(T.triangleMesh.devP,T.triangleMesh.devT[i],devX,devN[i]);
		HANDLE_ERROR(cudaMemcpy(N[i],devN[i],size*MAX_BLOCKS*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	if(mod>0)
	{
		parallel_accurate<<<size,mod>>>(T.triangleMesh.devP,T.triangleMesh.devTq,devX,devNq);
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

template<MeshType meshType>
std::vector<int> FiniteElementSpace<meshType>::collisionDetection(const std::vector<dvec> &X)
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
		parallel_accurate<<<size,MAX_BLOCKS>>>(T.triangleMesh.devP,T.triangleMesh.devT[i],devX,devN[i]);
		HANDLE_ERROR(cudaMemcpy(N[i],devN[i],size*MAX_BLOCKS*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	if(mod>0)
	{
		parallel_accurate<<<size,mod>>>(T.triangleMesh.devP,T.triangleMesh.devTq,devX,devNq);
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

template<MeshType meshType>
std::vector<dvec> FiniteElementSpace<meshType>::getValuesInMeshNodes(const std::vector<double> &a)
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

template<MeshType meshType>
std::vector<std::vector<dvec>> FiniteElementSpace<meshType>::getValuesInGaussNodes(const std::vector<double> &a)
{
	std::vector<std::vector<dvec>> x;
	for(int n=0;n<T.mesh.T.size();++n)
	{
		for(int mm=0;mm<T.q2t[n].size();++mm)
		{
			int m = T.q2t[n][mm];
			std::vector<dvec> xx;
			x.push_back(xx);
			for(int k=0;k<T.gauss.n;++k)
			{
//sLOG_OK("n: " << n );
//sLOG_OK("mm: " << mm );
//sLOG_OK("m: " << m );
//sLOG_OK(T.triangleMesh.B[m] << T.gauss.nodes[k]);
//sLOG_OK(T.triangleMesh.B[m]*T.gauss.nodes[k]+T.triangleMesh.b[m]);
				x[n].push_back((*this)(a,n).x(T.triangleMesh.B[m]*T.gauss.nodes[k]+T.triangleMesh.b[m]));
			}
		}
	}
	return x;
}

template<MeshType meshType>
Eigen::SparseMatrix<double> FiniteElementSpace<meshType>::gC(const Eigen::SparseMatrix<double>& S)
{
	return getColumns(S,edge);
}

template<MeshType meshType>
Eigen::SparseMatrix<double> FiniteElementSpace<meshType>::gR(const Eigen::SparseMatrix<double>& S)
{
	return getRows(S,edge);
}

template<MeshType meshType>
miniFE finiteElementSpace2miniFE(const FiniteElementSpace<meshType> &finiteElementSpace)
{
	miniFE m;
	m.finiteElement = finiteElementSpace.finiteElement.finiteElementName;
	m.gauss = finiteElementSpace.gauss.gaussName;
	m.mesh.P = finiteElementSpace.T.mesh.P;
	m.mesh.T = finiteElementSpace.T.mesh.T;
	m.mesh.E = finiteElementSpace.T.mesh.E;
	return m;
}

template<MeshType meshType>
FiniteElementSpace<meshType> miniFE2FiniteElementSpace(const miniFE &mini, GaussService &gaussService, FiniteElementService &finiteElementService)
{
	FiniteElementSpace<meshType> finiteElementSpace;
	SimplicialMesh<meshType> triangleMesh = SimplicialMesh<meshType>(mini.mesh,gaussService.getGauss(mini.gauss));
	triangleMesh.triangleMesh.loadOnGPU();
	finiteElementSpace = FiniteElementSpace<meshType>(triangleMesh,finiteElementService.getFiniteElement(mini.finiteElement),gaussService.getGauss(mini.gauss));
	finiteElementSpace.buildFiniteElementSpace();
	finiteElementSpace.buildEdge();
	return finiteElementSpace;
}

template<MeshType meshType>
Eigen::SparseMatrix<double> FiniteElementSpace<meshType>::applyEdgeCondition(const Eigen::SparseMatrix<double>& S)
{
	if(edge.size()>0)
		return S + gC(S)*E;
	else
		return S;
}

template<MeshType meshTypeA, MeshType meshTypeB>
Eigen::SparseMatrix<double> compress(const Eigen::SparseMatrix<double> &S, const FiniteElementSpace<meshTypeA> &E, const FiniteElementSpace<meshTypeB> &F)
{
	return getRows(getColumns(S,E.notEdge),F.notEdge);
	//C*A*Eigen::SparseMatrix<double>(C.transpose())
	//C*a
}

#define X(a) template FiniteElementSpace<a>::FiniteElementSpace();
MESH_TYPE_TABLE
#undef X

#define X(a) template FiniteElementSpace<a>::~FiniteElementSpace();
MESH_TYPE_TABLE
#undef X

#define X(a) template FiniteElementSpace<a>::FiniteElementSpace(SimplicialMesh<a> t, FiniteElement f, Gauss g);
MESH_TYPE_TABLE
#undef X

#define X(a) template F FiniteElementSpace<a>::operator()(const std::vector<double> &v, int n);
MESH_TYPE_TABLE
#undef X

#define X(a) template void FiniteElementSpace<a>::buildFiniteElementSpace();
MESH_TYPE_TABLE
#undef X

#define X(a) template void FiniteElementSpace<a>::calc(const std::vector<double> &v);
MESH_TYPE_TABLE
#undef X

#define X(a) template F FiniteElementSpace<a>::getPreCalc(int n);
MESH_TYPE_TABLE
#undef X

#define X(a) template FiniteElementSpace<a> miniFE2FiniteElementSpace(const miniFE &mini, GaussService &gaussService, FiniteElementService &finiteElementService);
MESH_TYPE_TABLE
#undef X

#define X(a) template miniFE finiteElementSpace2miniFE(const FiniteElementSpace<a> &finiteElementSpace);
MESH_TYPE_TABLE
#undef X

#define X(a) template std::vector<dvec> FiniteElementSpace<a>::getValuesInMeshNodes(const std::vector<double> &v);
MESH_TYPE_TABLE
#undef X

#define X(a) template BaseFunction FiniteElementSpace<a>::getBaseFunction(int i, int n);
MESH_TYPE_TABLE
#undef X

#define X(a) template std::vector<int> FiniteElementSpace<a>::collisionDetection(const std::vector<dvec> &X);
MESH_TYPE_TABLE
#undef X

#define X(a) template std::vector<std::vector<int>> FiniteElementSpace<a>::collisionDetection(const std::vector<std::vector<dvec>> &X);
MESH_TYPE_TABLE
#undef X

#define X(a) template std::vector<std::vector<dvec>> FiniteElementSpace<a>::getValuesInGaussNodes(const std::vector<double> &v);
MESH_TYPE_TABLE
#undef X

#define X(a) template Eigen::SparseMatrix<double> compress(const Eigen::SparseMatrix<double> &S, const FiniteElementSpace<a> &E, const FiniteElementSpace<a> &F);
MESH_TYPE_TABLE
#undef X
template Eigen::SparseMatrix<double> compress(const Eigen::SparseMatrix<double> &S, const FiniteElementSpace<oneDim> &E, const FiniteElementSpace<Triangular> &F);

#define X(a) template Eigen::SparseMatrix<double> FiniteElementSpace<a>::applyEdgeCondition(const Eigen::SparseMatrix<double> &S);
MESH_TYPE_TABLE
#undef X

