/*
*	Simulation.cu
*/

#include "Simulation.h"
#define M_THREAD 16 //TODO

Simulation::Simulation(){}

Simulation::Simulation(std::string id,dbconfig db,Parameters parameters,
	FiniteElementSpaceV_ V,FiniteElementSpaceQ_ Q,FiniteElementSpaceS_ S,FiniteElementSpaceL_ L,
	timestep t0,timestep t1,bool full)
{
	this->id = id;
	this->db = db;
	this->parameters = parameters;
	this->V = V;
	this->Q = Q;
	this->S = S;
	this->L = L;
	this->full = full;
	setInitialValues(t0,t1);
	saveSimulation();
}

Simulation::~Simulation(){}

double Simulation::getEnergy(timestep t)
{
	std::function<double(dvec)> isInS = [](dvec x){
		if(false)
			return 1;
		else
			return 0;
	};

	double Ekin = 0.0;
	for(int n=0;n<V.nodes.T.size();++n)
		Ekin += 0.5*V.T.integrate((constant(parameters.rho)+parameters.deltarho*isInS)*ddot(V(t.u,n).x,V(t.u,n).x),n);

	double Epot = 0.0;
	for(int n=0;n<S.nodes.T.size();++n)
		Epot += parameters.kappa*S.T.integrate(pf(S(t.x,n).dx,S(t.x,n).dx),n);

	return Ekin+Epot;
}

void Simulation::setInitialValues(timestep t0, timestep t1)
{
	t0.time = 0;
	t0.id = this->id;
	timesteps.push_back(t0);
	t1.time = 1;
	t1.id = this->id;
	timesteps.push_back(t1);
}

void Simulation::clear()
{
	Ct.clear();
	Lf.clear();
	MB.clear();
	M.clear();
	MM.clear(); //TODO
}

miniSim Simulation::sim2miniSim()
{
	miniSim mini;
	mini.id = this->id;
	mini.parameters = this->parameters;
	mini.full = this->full;

	mini.V = finiteElementSpace2miniFE(V);
	mini.Q = finiteElementSpace2miniFE(Q);
	mini.S = finiteElementSpace2miniFE(S);
	mini.L = finiteElementSpace2miniFE(L);

	return mini;
}

void Simulation::saveSimulation()
{
	save_sim(db, sim2miniSim());
	saveTimestep(0);
	//savePlotData(0); //TODO
	saveTimestep(1);
	//savePlotData(1);
}

void Simulation::getSimulation(dbconfig db,std::string id)
{
	miniSim mini = get_sim(db,id);
	this->id = id;
	this->db = db;
	this->parameters = mini.parameters;
	this->full = mini.full;

	V = miniFE2FiniteElementSpace<Triangular>(mini.V,gaussService,finiteElementService);
	Q = miniFE2FiniteElementSpace<Triangular>(mini.Q,gaussService,finiteElementService);
	S = miniFE2FiniteElementSpace<Triangular>(mini.S,gaussService,finiteElementService);
	L = miniFE2FiniteElementSpace<Triangular>(mini.L,gaussService,finiteElementService);

	int time = get_time(db,id);
	for(int i=0;i<time;++i)
	{
		timesteps.push_back(getTimestep(i));
	}
}

int Simulation::getTime()
{
	return timesteps.size()-1;
}

void Simulation::saveTimestep(int time)
{
	save_timestep(this->db,timesteps[time]);
}

void Simulation::saveTimestep()
{
	saveTimestep(getTime());
}

timestep Simulation::getTimestep(int time)
{
	return get_timestep(db,id,time);
}

plotData Simulation::timestep2plotData(timestep t)
{
	plotData p;
	p.id = t.id;
	p.time = t.time;

	std::vector<std::vector<double>> box = V.T.triangleMesh.getBox();
	double step = 0.1;

	std::vector<dvec> G;
	for(double x=box[0][0];x<box[1][0]+M_EPS;x+=step)
	{
		G.push_back({x});
	}
	for(int d=1;d<V.T.triangleMesh.dim;++d)
	{
		std::vector<dvec> slice = G;
		G.clear();
		for(double x=box[0][d];x<box[1][d]+M_EPS;x+=step)
		{
			for(dvec v : slice)
			{
				v(v.size++) = x;
				G.push_back(v);
			}
		}
	}

//	for(dvec v : G)
//		std::cout << v << std::endl;

	std::vector<int> M = V.collisionDetection(G);

	for(int i=0;i<G.size();++i)
	{
//sLOG_OK("i: " << i);
//sLOG_OK("G[i]: " << G[i]);
//sLOG_OK("M[i]: " << M[i]);
//sLOG_OK(ditrian2dtrian(V.T.mesh.T[M[i]],V.T.mesh.P.data()));
//assert(accurate(G[i],ditrian2dtrian(V.T.mesh.T[M[i]],V.T.mesh.P.data())));
		assert(M[i]!=-1);
		dvec vv = V(timesteps[p.time].u,M[i]).x(G[i]);
		dvec qq = Q(timesteps[p.time].q,M[i]).x(G[i]);
		p.u.push_back({dvec2vector(G[i]),dvec2vector(vv)});
		p.q.push_back({dvec2vector(G[i]),dvec2vector(qq)});
	}

	/*std::vector<std::vector<double>> boxS = S.T.getBox();
	double stepS = 1;

	std::vector<dvec> H;
	for(double x=boxS[0][0];x<boxS[1][0]+M_EPS;x+=stepS)
	{
		H.push_back({x});
	}
	for(int d=1;d<S.T.dim;++d)
	{
		std::vector<dvec> slice = H;
		H.clear();
		for(double x=box[0][d];x<boxS[1][d]+M_EPS;x+=stepS)
		{
			for(dvec v : slice)
			{
				v(v.size++) = x;
				H.push_back(v);
			}
		}
	}
	std::vector<int> N = S.collisionDetection(H);

	for(int i=0;i<H.size();++i)
	{
		assert(N[i]!=-1);
		dvec ss = S(timesteps[p.time].x,N[i]).x(H[i]);
		p.x.push_back({dvec2vector(H[i]),dvec2vector(ss)});
	}*/

	for(dvec pp : S.nodes.P)
	{
		int m = S.collisionDetection({pp})[0];
		dvec s = S(timesteps[p.time].x,m).x(pp);
std::ostringstream ss,tt;
ss << "x    = [ " << pp(0) << " ]";//<< " , " << pp(1) << " ]";
tt << "S(x) = [ " << s(0) << " , " << s(1) << " ]";
LOG_TRACE(ss);
LOG_TRACE(tt);
		if(s(0)!=0 || s(1)!=0){
			p.x.push_back({dvec2vector(pp),dvec2vector(s)});
		}
		else{
			LOG_WARNING("S(x) == 0 !");
			p.x.push_back({dvec2vector(pp),p.x[p.x.size()-1][1]});
		}
	}

	return p;
}

void Simulation::savePlotData(int time)
{
	save_plotData(this->db,timestep2plotData(timesteps[time]));
}

void Simulation::savePlotData()
{
	savePlotData(getTime());
}

void Simulation::buildEdge()
{
	if(full)
	{
		edge = V.edge;
		edge = join(edge,Q.edge+V.spaceDim);
		edge = join(edge,S.edge+(V.spaceDim+Q.spaceDim));
		notEdge = setdiff(linspace(V.spaceDim+Q.spaceDim+S.spaceDim+L.spaceDim),V.edge);
		notEdge = setdiff(notEdge,Q.edge+V.spaceDim);
		notEdge = setdiff(notEdge,S.edge+(V.spaceDim+Q.spaceDim));
	}
	else
	{
		edge = V.edge;
		edge = join(edge,Q.edge+V.spaceDim);
		notEdge = setdiff(linspace(V.spaceDim+Q.spaceDim),V.edge);
		notEdge = setdiff(notEdge,Q.edge+V.spaceDim);
	}
}

void Simulation::prepare()
{
	LOG_INFO("buildEdge..."); buildEdge();
	LOG_INFO("buildFluidMatrices...");buildFluidMatrices();
	if(full)
	{
		LOG_INFO("buildStructureMatrices...");buildStructureMatrices();
		LOG_INFO("buildMultiplierMatrices...");buildMultiplierMatrices();
	}
	//TODO creare matrice sparse qui
}

void Simulation::advance()
{
	auto t0 = std::chrono::high_resolution_clock::now();
	LOG_INFO("clear...");clear();
	LOG_INFO("buildK2f...");buildK2f();
	if(full)
	{
		LOG_INFO("buildLf...");buildLf();
	}
	else
	{
		LOG_INFO("buildF...");buildF();
	}
	LOG_INFO("triplet2sparse...");triplet2sparse();
	LOG_INFO("buildb...");buildb();
	LOG_INFO("solve...");solve();
	LOG_INFO("save...");save();
	LOG_INFO("savePlotData...");savePlotData();
	LOG_INFO("Done.");
	auto t1 = std::chrono::high_resolution_clock::now();
	double duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

	std::ostringstream c0; c0 << "Energy: " << getEnergy(timesteps[getTime()]);
	LOG_OK(c0);
	std::ostringstream d0; d0 << "Simulation " << id << " timestep " << getTime() << " took " << duration << " microseconds.";
	LOG_OK(d0);
}

void Simulation::advance(int steps)
{
	for(int i=0;i<steps;++i)
	{
		this->advance();
	}
}

void Simulation::save()
{
	saveTimestep();
}

void Simulation::buildFluidMatrices()
{
	etmat B;
	for(int n=0;n<V.nodes.T.size();++n)
	{
		for(int i=0;i<V.baseFunction.size();++i)
		{
			BaseFunction a = V.getBaseFunction(i,n);
			for(int j=0;j<V.baseFunction.size();++j)
			{
				BaseFunction b = V.getBaseFunction(j,n);
				double valMf = V.T.integrate(ddot(a.x,b.x),n);
				double valK1f = V.T.integrate(pf(symm(a.dx),symm(b.dx)),n);
				assert(a.i < V.spaceDim && b.i < V.spaceDim);
				assert(a.mini_i < (V.spaceDim-V.nBT) && b.mini_i < (V.spaceDim-V.nBT));
				if(valMf!=0)
				{
					Mf.push_back(Eigen::Triplet<double>(a.i,b.i,valMf));
					if(a.mini_i!=-1 && b.mini_i!=-1)
					{
						C.push_back(Eigen::Triplet<double>(a.mini_i,b.mini_i,(parameters.rho/parameters.deltat)*valMf));
					}
				}
				if(valK1f!=0)
				{
					K1f.push_back(Eigen::Triplet<double>(a.i,b.i,valK1f));
					if(a.mini_i!=-1 && b.mini_i!=-1)
					{
						C.push_back(Eigen::Triplet<double>(a.mini_i,b.mini_i,(parameters.eta)*valK1f));
					}
				}
			}
			for(int j=0;j<Q.baseFunction.size();++j)
			{
				BaseFunction b = Q.getBaseFunction(j,n);
				double valB = -V.T.integrate(project(b.x,0)*div(a.dx),n);
				assert(a.i < V.spaceDim && b.i < Q.spaceDim);
				assert(a.mini_i < (V.spaceDim-V.nBT) && b.mini_i < (Q.spaceDim-Q.nBT));
				if(valB!=0)
				{
					B.push_back(Eigen::Triplet<double>(a.i,b.i,valB));
				}
			}
		}
	}
	Eigen::SparseMatrix<double> sB = Eigen::SparseMatrix<double>(V.spaceDim,Q.spaceDim);
	sB.setFromTriplets(B.begin(),B.end());
	etmat temp = esmat2etmat(compress(Q.applyEdgeCondition(sB),Q,V),0,(V.spaceDim-V.nBT));
	C+=temp;
	C+=transpose(temp);
}

void Simulation::buildStructureMatrices()
{
	etmat Ks;
	etmat Bs;
	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int i=0;i<S.baseFunction.size();++i)
		{
			BaseFunction a = S.getBaseFunction(i,n);
			for(int j=0;j<S.baseFunction.size();++j)
			{
				BaseFunction b = S.getBaseFunction(j,n);
				double valMs = S.T.integrate(ddot(a.x,b.x),n);
				double valKs = parameters.kappa*S.T.integrate(pf(a.dx,b.dx),n);
				Ms.push_back(Eigen::Triplet<double>(a.i,b.i,valMs));
				Ks.push_back(Eigen::Triplet<double>(a.i,b.i,valKs));
				Bs.push_back(Eigen::Triplet<double>(a.i,b.i,parameters.deltarho/(parameters.deltat*parameters.deltat)*valMs+valKs));
			}
		}
	}
	Eigen::SparseMatrix<double> sB = Eigen::SparseMatrix<double>(S.spaceDim,S.spaceDim);
	sB.setFromTriplets(Bs.begin(),Bs.end());
	etmat temp = esmat2etmat(compress(S.applyEdgeCondition(sB),S,S),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT));
	C+=temp;
}

void Simulation::buildMultiplierMatrices()
{
	etmat Bs;
	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int i=0;i<L.baseFunction.size();++i)
		{
			BaseFunction a = L.getBaseFunction(i,n);
			for(int j=0;j<S.baseFunction.size();++j)
			{
				BaseFunction b = S.getBaseFunction(j,n);
				double valLs = S.T.integrate(ddot(a.x,b.x),n);
				Ls.push_back(Eigen::Triplet<double>(a.i,b.i,valLs));
				Bs.push_back(Eigen::Triplet<double>(a.i,b.i,-(1.0/parameters.deltat)*valLs));
			}
		}
	}
	Eigen::SparseMatrix<double> sB = Eigen::SparseMatrix<double>(L.spaceDim,S.spaceDim);
	sB.setFromTriplets(Bs.begin(),Bs.end());
	etmat temp = esmat2etmat(compress(transpose(L.applyEdgeCondition(transpose(S.applyEdgeCondition(sB)))),S,L),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT));
	C+=temp;
	C+=transpose(temp);
}

void Simulation::buildK2f()
{
logx::Logger::getInstance()->setLogLevel("src/TriangleMesh/SimplicialMesh.cu",LOG_LEVEL_DEBUG);
logx::Logger::getInstance()->setLogLevel("src/TriangleMesh/SimplicialMesh.cu",LOG_LEVEL_INFO);
	int time = timesteps.size();
	for(int n=0;n<V.nodes.T.size();++n)
	{
		F v = V(timesteps[time-1].u,n);
		for(int i=0;i<V.elementDim;++i)
		{
			BaseFunction a = V.getBaseFunction(i,n);
			for(int j=0;j<V.elementDim;++j)
			{
				BaseFunction b = V.getBaseFunction(j,n);
				std::function<double(dvec)> g = ddot(dotdiv(v.x,a.dx),a.x)-ddot(dotdiv(v.x,b.dx),b.x);
//sLOG_OK("n: " << n << "/" << V.nodes.T.size() << std::endl << "i: " << i << "/" << V.elementDim << std::endl << "j: " << j << "/" << V.elementDim);
				double valK2f = V.T.integrate(g,n);
//sLOG_OK("valK2f: " << valK2f);
				if(valK2f!=0)
				{
					if(a.mini_i!=-1 && b.mini_i!=-1)
					{
						Ct.push_back(Eigen::Triplet<double>(a.mini_i,b.mini_i,(parameters.rho/2.0)*valK2f));
					}
				}
			}
		}
	}
}

void Simulation::buildLf()
{
	int time = timesteps.size();
	std::vector<std::vector<dvec>> yyy = S.getValuesInGaussNodes(timesteps[time-1].x);
	MM = V.collisionDetection(yyy);
	S.calc(timesteps[time-1].x);
	for(int n=0;n<S.nodes.T.size();++n)
	{
		F preS = S.getPreCalc(n);
		assert(MM[n].size()!=0);
		for(int k=0;k<MM[n].size();++k)
		{
			int m=MM[n][k];
			for(int i=0;i<L.baseFunction.size();++i)
			{
				BaseFunction a = L.getBaseFunction(i,n);
				for(int j=0;j<V.baseFunction.size();++j)
				{
					BaseFunction b = V.getBaseFunction(j,m);
					double valLf = S.T.integrate(ddot(a.x,compose(b.f,preS).x),n);
					if(valLf!=0)
					{
						Lf.push_back(Eigen::Triplet<double>(a.i,b.i,valLf));
						/*if(a.mini_i!=-1 && b.mini_i!=-1)
						{
							Ct.push_back(Eigen::Triplet<double>(a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),b.mini_i,valLf));
							Ct.push_back(Eigen::Triplet<double>(b.mini_i,a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),valLf));
						}*/
					}
				}
			}
		}
	}

	Eigen::SparseMatrix<double> sB = Eigen::SparseMatrix<double>(L.spaceDim,V.spaceDim);
	sB.setFromTriplets(Lf.begin(),Lf.end());
	etmat temp = esmat2etmat(compress(transpose(L.applyEdgeCondition(transpose(V.applyEdgeCondition(sB)))),V,L),
		(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),0);

	/*std::ostringstream ss0;
	ss0 << "0. Lf: " << std::endl << edmat(sB);	
	LOG_TRACE(ss0);

	std::ostringstream ss1;
	//ss << "1. Lf: " << std::endl << edmat(etmat2esmat(esmat2etmat(compress(transpose(L.applyEdgeCondition(transpose(V.applyEdgeCondition(sB)))),V,L),0,0),8,10));	
	ss1 << "1. Lf: " << std::endl << edmat(transpose(V.applyEdgeCondition(sB)));
	LOG_TRACE(ss1);

	std::ostringstream ss;
	//ss << "1. Lf: " << std::endl << edmat(etmat2esmat(esmat2etmat(compress(transpose(L.applyEdgeCondition(transpose(V.applyEdgeCondition(sB)))),V,L),0,0),8,10));	
	ss << "1. Lf: " << std::endl << edmat(L.applyEdgeCondition(transpose(V.applyEdgeCondition(sB))));
	LOG_TRACE(ss);

	std::ostringstream ss2;
	ss2 << "2. Lf: " << std::endl << edmat(etmat2esmat(esmat2etmat(compress(transpose(L.applyEdgeCondition(transpose(V.applyEdgeCondition(sB)))),V,L),0,0),8,10));	
	LOG_TRACE(ss2);*/

	Ct+=temp;
	Ct+=transpose(temp);
}

void Simulation::buildF()
{
	int time = timesteps.size();
  	FF = evec::Zero(V.spaceDim);
logx::Logger::getInstance()->setLogLevel("src/TriangleMesh/SimplicialMesh.cu",LOG_LEVEL_DEBUG);
	std::vector<std::vector<dvec>> yyy = S.getValuesInGaussNodes(timesteps[time-1].x);
	MM = V.collisionDetection(yyy);

	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int k=0;k<MM[n].size();++k)
		{
			int m=MM[n][k];
			assert(m>-1);
			for(int i=0;i<V.baseFunction.size();++i)
			{
				BaseFunction a = V.getBaseFunction(i,m);
				double valF = S.T.integrate(pf(S(timesteps[time-1].x,n).dx,compose(a.f,S(timesteps[time-1].x,n)).dx),n);
				FF(a.i) += (-parameters.kappa)*valF; //WAT2
				for(int j=0;j<V.baseFunction.size();++j)
				{
					BaseFunction b = V.getBaseFunction(j,m);
					double valMB = S.T.integrate(ddot(compose(a.f,S(timesteps[time-1].x,n)).x,compose(b.f,S(timesteps[time-1].x,n)).x),n);
					if(valMB!=0)
					{
						MB.push_back(Eigen::Triplet<double>(a.i,b.i,valMB));
						if(a.mini_i!=-1 && b.mini_i!=-1)
						{
							Ct.push_back(Eigen::Triplet<double>(a.mini_i,b.mini_i,(parameters.deltarho/parameters.deltat)*valMB));
						}
					}
				}
			}
		}
	}
	std::ostringstream ss;
	ss << "|FF| = " << FF.norm();
	LOG_INFO(ss);
}

void Simulation::triplet2sparse()
{
	if(full)
	{
		sC = esmat((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+(L.spaceDim-L.nBT),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+(L.spaceDim-L.nBT));
		sCt = esmat((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+(L.spaceDim-L.nBT),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+(L.spaceDim-L.nBT));
		sLs = esmat(L.spaceDim,L.spaceDim);
		sLf = esmat(L.spaceDim,V.spaceDim);

		sLs.setFromTriplets(Ls.begin(),Ls.end());
		sLf.setFromTriplets(Lf.begin(),Lf.end());
	}
	else
	{
		sC = esmat((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT));
		sCt = esmat((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT));
	}

	sC.setFromTriplets(C.begin(),C.end());
	sCt.setFromTriplets(Ct.begin(),Ct.end());
	sCt = sC+sCt;

	sMf = esmat(V.spaceDim,V.spaceDim);
	sMs = esmat(S.spaceDim,S.spaceDim);
	sMB = esmat(V.spaceDim,V.spaceDim);

	sMf.setFromTriplets(Mf.begin(),Mf.end());
	sMs.setFromTriplets(Ms.begin(),Ms.end());
	sMB.setFromTriplets(MB.begin(),MB.end());
}

void Simulation::buildb()
{
	int time = timesteps.size();
	evec u_1 = vector2eigen(timesteps[time-1].u);
	evec x_1 = vector2eigen(timesteps[time-1].x);
	evec x_2 = vector2eigen(timesteps[time-2].x);

	if(full)
	{
		b = evec::Zero((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+L.spaceDim);
		evec f = (parameters.rho/parameters.deltat)*sMf*u_1;
		evec o = evec::Zero(Q.spaceDim);
		evec g = (parameters.deltarho/(parameters.deltat*parameters.deltat))*sMs*(2.0*x_1-x_2);
		evec d = (-1.0/parameters.deltat)*sLs*(x_1);

		b << getRows(f,V.notEdge)
		,getRows(o,Q.notEdge)
		,getRows(g,S.notEdge)
		,getRows(d,L.notEdge);

		/*for(int i=0;i<V.spaceDim+Q.spaceDim+S.spaceDim+L.spaceDim;++i)
		{
			int j=find(notEdge,i);
			if(j!=-1)
			{
				if(i<V.spaceDim)
				{
					b(j)=f(i);
				}
				else if(i<V.spaceDim+Q.spaceDim)
				{
					b(j)=0;
				}
				else if(i<V.spaceDim+Q.spaceDim+S.spaceDim)
				{
					b(j)=g(i-V.spaceDim-Q.spaceDim);
				}
				else
				{
					b(j)=d(i-V.spaceDim-Q.spaceDim-S.spaceDim);
				}
//std::cout << i << " " << j << " " << b(j) << std::endl;
			}
		}*/
	}
	else
	{
		b = evec::Zero((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT));
		evec f = (parameters.rho/parameters.deltat)*sMf*u_1+(parameters.deltarho/parameters.deltat)*sMB*u_1+FF;
		evec o = evec::Zero(Q.spaceDim);
		for(int i=0;i<V.spaceDim+Q.spaceDim;++i)
		{
			int j=find(notEdge,i);
			if(j!=-1)
			{
				if(i<V.spaceDim)
				{
					b(j)=f(i);
				}
				else if(i<V.spaceDim+Q.spaceDim)
				{
					b(j)=0;
				}
			}
		}
	}
}

void Simulation::updateX()
{
	int time = timesteps.size()-1;

	std::vector<dvec> Xt = S.getValuesInMeshNodes(timesteps[time-1].x);

	for(dvec v : Xt)
	{
		std::ostringstream ss;
		ss << v;
		LOG_TRACE(ss);
	}

	std::vector<int> MMM = V.collisionDetection(Xt);
	std::ostringstream ss;
	ss << "CollisionDetection done. Found " << MMM.size() << " / " <<  Xt.size() << " points.";
	LOG_DEBUG(ss);

	std::vector<dvec> u;
	u.reserve(S.nodes.P.size());
	u.resize(S.nodes.P.size());

	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int k=0;k<S.nodes.T[n].size();++k)
		{
			int i = S.nodes.T[n][k];
			int m=MMM[i];
			assert(m>-1);
			u[i]= compose(V(timesteps[time].u,m),S(timesteps[time-1].x,n)).x(S.nodes.P[i]);
		}
	}
	std::vector<double> uu(S.spaceDim);
	for(int i=0;i<S.spaceDim/S.ambientDim;++i)
	{
		for(int j=0;j<S.ambientDim;++j)
		{

			uu[i+j*S.spaceDim/S.ambientDim]=u[i](j);
		}
	}
	timesteps[time].x = timesteps[time-1].x+parameters.deltat*uu;
}

timestep Simulation::eigen2timestep(evec a)
{
	LOG_INFO("eigen2timestep...");
	timestep t;

	evec bV = a.block(0,0,V.spaceDim-V.nBT,1);
	evec aV = getColumns(V.E.sparseView(),V.notEdge)*bV;
	std::vector<double> tV = join(bV,aV,V.notEdge,V.edge);

	evec bQ = a.block(V.spaceDim-V.nBT,0,Q.spaceDim-Q.nBT,1);
	evec aQ = getColumns(Q.E.sparseView(),Q.notEdge)*bQ;
	std::vector<double> tQ = join(bQ,aQ,Q.notEdge,Q.edge);

	t.u = tV;
	t.q = tQ;

if(full){
	evec bS = a.block(V.spaceDim-V.nBT+Q.spaceDim-Q.nBT,0,S.spaceDim-S.nBT,1);
	evec aS = getColumns(S.E.sparseView(),S.notEdge)*bS;
	std::vector<double> tS = join(bS,aS,S.notEdge,S.edge);

//std::cout << S.E*vector2eigen(tS) << std::endl;
//std::cout << "[ " << tS[0] << " , " << tS[65] << " ]" << std::endl;
//std::cout << "[ " << tS[64] << " , " << tS[129] << " ]" << std::endl;

	evec bL = a.block(V.spaceDim-V.nBT+Q.spaceDim-Q.nBT+S.spaceDim-S.nBT,0,L.spaceDim-L.nBT,1);
	evec aL = getColumns(L.E.sparseView(),L.notEdge)*bL;
	std::vector<double> tL = join(bL,aL,L.notEdge,L.edge);
	//t.x = eigen2vector(bS);
	//t.l = eigen2vector(bL);
	t.x = tS;
	t.l = tL;
}
	//std::cout << tV << std::endl;
	//std::cout << tQ << std::endl;
	//std::cout << tS << std::endl;
	//std::cout << eigen2vector(bL) << std::endl;


	/*int j=0;
	for(int i=0;i<V.spaceDim && j<a.size();++i)
	{
		if(find(edge,i)==-1)
			t.u.push_back(a(j++));
		else
			t.u.push_back(0);
	}
	for(int i=0;i<Q.spaceDim && j<a.size();++i)
	{
		if(find(edge,i+V.spaceDim)==-1)
			t.q.push_back(a(j++));
		else
		{
			double b = 0;
			evec A = a.block(V.spaceDim-V.nBT,0,Q.spaceDim-Q.nBT,1);
			b = (getColumns(Q.E.sparseView(),Q.notEdge)*A)(0,0);
			t.q.push_back(b);
		}
	}
		bool first = true; //TODO
	for(int i=0;i<S.spaceDim && j<a.size();++i)
	{

		if(find(edge,i+V.spaceDim+Q.spaceDim)==-1)
		{
			t.x.push_back(a(j++));
		}
		else
		{
			double b0,b1 = 0;
			evec A = a.block(V.spaceDim-V.nBT+Q.spaceDim-Q.nBT,0,S.spaceDim-S.nBT,1);
			b0 = (getColumns(S.E.sparseView(),S.notEdge)*A)(0,0);
			b1 = (getColumns(S.E.sparseView(),S.notEdge)*A)(1,0);
			if(first)
			{	
				first = false;
				t.x.push_back(b0);
			}
			else{
std::cout << "A!" << std::endl;
				t.x.push_back(b1);
			}
		}
	}
	for(int i=0;i<L.spaceDim && j<a.size();++i)
	{
		if(find(edge,i+V.spaceDim+Q.spaceDim+S.spaceDim)==-1)
			t.l.push_back(a(j++));
		else
			t.l.push_back(0);
	}*/
	t.id = id;
	t.time = timesteps.size();
	return t;
}

void Simulation::solve()
{
//std::cout << (Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>)sCt << std::endl;
//std::cout << b << std::endl;

	if(full)
	{
		//Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,Eigen::IncompleteLUT<double>> solver;
		Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> solver;
		solver.compute(sCt);
		evec v = solver.solve(b);
		timesteps.push_back(eigen2timestep(v));
	}
	else
	{
		//Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> solver;
		//Eigen::SparseLU<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> solver;
		//Eigen::HouseholderQR<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> solver;
		//Eigen::JacobiSVD<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> solver;
		//Eigen::SPQR<Eigen::SparseMatrix<double>> solver;
		//Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,Eigen::IncompleteLUT<double>> solver;
		//Eigen::SuperLU<Eigen::SparseMatrix<double>> solver;
		Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,Eigen::Lower|Eigen::Upper> solver;
		//std::cout << Eigen::nbThreads() << std::endl;
		solver.compute(sCt);
		evec v = solver.solve(b);
		timesteps.push_back(eigen2timestep(v));
		updateX();
	}
}

