/*
*	Simulation.cu
*/

#include "Simulation.h"
#define M_THREAD 16 //TODO

Simulation::Simulation(){}

Simulation::Simulation(std::string id,dbconfig db,Parameters parameters,
	FiniteElementSpaceV V,FiniteElementSpaceQ Q,FiniteElementSpaceS S,FiniteElementSpaceL L,
	timestep t0,timestep t1)
{
	this->id = id;
	this->db = db;
	this->parameters = parameters;
	this->V = V;
	this->Q = Q;
	this->S = S;
	this->L = L;
	this->time = 2; //TODO
	this->full = true;
	setInitialValues(t0,t1);
	saveSimulation();
}

Simulation::Simulation(std::string id,dbconfig db,Parameters parameters,
	FiniteElementSpaceV V,FiniteElementSpaceQ Q,FiniteElementSpaceS S,
	timestep t0,timestep t1)
{
	this->id = id;
	this->db = db;
	this->parameters = parameters;
	this->V = V;
	this->Q = Q;
	this->S = S;
	this->L = FiniteElementSpaceL(S.T,S.finiteElement,S.gauss); //TODO
	this->time = 2; //TODO
	this->full = false;
	setInitialValues(t0,t1);
	saveSimulation();
}

Simulation::~Simulation(){}

void Simulation::setInitialValues(timestep t0, timestep t1)
{
	timesteps.push_back(t0);
	timesteps.push_back(t1);
}

void Simulation::clear()
{
	Ct.clear();
	Lf.clear();
	MB.clear();
	M.clear();	
	MM.clear();
}

miniSim Simulation::sim2miniSim()
{
	miniSim mini;
	mini.id = this->id;
	mini.parameters = this->parameters;
	mini.full = this->full;

	mini.V = finiteElementSpace2miniFE(V);
	//mini.V.finiteElement = this->V.finiteElement.finiteElementName;
	//mini.V.gauss = this->V.gauss.gaussName;
	//mini.V.mesh.P = this->V.T.mesh.P;
	//mini.V.mesh.T = this->V.T.mesh.T;
	//mini.V.mesh.E = this->V.T.mesh.E;

	mini.Q.finiteElement = this->Q.finiteElement.finiteElementName;
	mini.Q.gauss = this->Q.gauss.gaussName;
	mini.Q.mesh.P = this->Q.T.mesh.P;
	mini.Q.mesh.T = this->Q.T.mesh.T;
	mini.Q.mesh.E = this->Q.T.mesh.E;

	mini.S.finiteElement = this->S.finiteElement.finiteElementName;
	mini.S.gauss = this->S.gauss.gaussName;
	mini.S.mesh.P = this->S.T.mesh.P;
	mini.S.mesh.T = this->S.T.mesh.T;
	mini.S.mesh.E = this->S.T.mesh.E;

	mini.L.finiteElement = this->L.finiteElement.finiteElementName;
	mini.L.gauss = this->L.gauss.gaussName;
	mini.L.mesh.P = this->L.T.mesh.P;
	mini.L.mesh.T = this->L.T.mesh.T;
	mini.L.mesh.E = this->L.T.mesh.E;

	return mini;
}

void Simulation::saveSimulation()
{
	save_sim(db, sim2miniSim());
	saveTimestep(0);
	saveTimestep(1);
}

void Simulation::getSimulation(dbconfig db,std::string id)
{
	miniSim mini = get_sim(db,id);
	this->id = id;
	this->db = db;
	this->full = mini.full;
	this->parameters = mini.parameters;

	V = miniFE2FiniteElementSpace(mini.V,gaussService,finiteElementService);
	//TriangleMesh TV = TriangleMesh(mini.V.mesh,gaussService.getGauss(mini.V.gauss));
	//TV.loadOnGPU();
	//V = FiniteElementSpaceV(TV,finiteElementService.getFiniteElement(mini.V.finiteElement),gaussService.getGauss(mini.V.gauss));
	//V.buildFiniteElementSpace();
	//V.buildEdge();

	TriangleMesh TQ = TriangleMesh(mini.Q.mesh,gaussService.getGauss(mini.Q.gauss));
	TQ.loadOnGPU();
	Q = FiniteElementSpaceQ(TQ,finiteElementService.getFiniteElement(mini.Q.finiteElement),gaussService.getGauss(mini.Q.gauss));
	Q.buildFiniteElementSpace();
	Q.buildEdge();

	TriangleMesh TS = TriangleMesh(mini.S.mesh,gaussService.getGauss(mini.S.gauss));
	TS.loadOnGPU();
	S = FiniteElementSpaceS(TS,finiteElementService.getFiniteElement(mini.S.finiteElement),gaussService.getGauss(mini.S.gauss));
	S.buildFiniteElementSpace();
	S.buildEdge();

	TriangleMesh TL = TriangleMesh(mini.L.mesh,gaussService.getGauss(mini.L.gauss));
	TL.loadOnGPU();
	L = FiniteElementSpaceL(TL,finiteElementService.getFiniteElement(mini.L.finiteElement),gaussService.getGauss(mini.L.gauss));
	L.buildFiniteElementSpace();
	L.buildEdge();

//	int 
	time = get_time(db,id);

	for(int i=0;i<time;++i)
	{
		timesteps.push_back(getTimestep(i));
	}
}

void Simulation::saveTimestep(int time)
{
	save_timestep(this->db,timesteps[time]);
}

void Simulation::saveTimestep()
{
	//int time = timesteps.size();
	saveTimestep(this->time-1); //TODO
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

	std::vector<std::vector<double>> box = V.T.getBox();
	double step = 0.1;

	std::vector<dvec> G;
	for(double x=box[0][0];x<box[1][0]+M_EPS;x+=step)
	{
		G.push_back({x});
	}
	for(int d=1;d<V.T.dim;++d)
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
		assert(M[i]!=-1);
		dvec vv = V(timesteps[p.time].u,M[i]).x(G[i]);
		dvec qq = Q(timesteps[p.time].q,M[i]).x(G[i]);
		p.u.push_back({dvec2vector(G[i]),dvec2vector(vv)});
		p.q.push_back({dvec2vector(G[i]),dvec2vector(qq)});
	}

	std::vector<std::vector<double>> boxS = S.T.getBox();
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
	}

	return p;
}

void Simulation::savePlotData(int time)
{
	save_plotData(this->db,timestep2plotData(timesteps[time]));
}

void Simulation::savePlotData()
{
	//int time = timesteps.size();
	savePlotData(this->time-1); //TODO
}

void Simulation::buildEdge()
{
	if(full){
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
	buildEdge();
	buildFluidMatrices();
	if(full)
	{
		buildStructureMatrices();
		buildMultiplierMatrices();
	}
	//TODO creare matrice sparse qui
}

void Simulation::advance()
{
	this->advance(1);
}

void Simulation::advance(int steps)
{
	for(int i=0;i<steps;++i)
	{
		std::cout << "clear..." << std::endl;clear();
		std::cout << "buildK2f..." << std::endl;buildK2f();
		if(full)
		{
			std::cout << "buildLf..." << std::endl;buildLf();		
		}
		else
		{
			std::cout << "buildF..." << std::endl;buildF();
		}
		std::cout << "triplet2sparse..." << std::endl;triplet2sparse();
		std::cout << "buildb..." << std::endl;buildb();
		std::cout << "solve..." << std::endl;solve();
		std::cout << "save..." << std::endl;save();
		std::cout << "savePlotData..." << std::endl;savePlotData();
		std::cout << "Done." << std::endl;
	}
}

void Simulation::save()
{
	saveTimestep();
}

void Simulation::buildFluidMatrices()
{
	std::vector<Eigen::Triplet<double>> B;

	for(int n=0;n<V.nodes.T.size();++n)
	{
		for(int i=0;i<V.baseFunction.size();++i)
		{
			BaseFunction a = V.getBaseFunction(i,n);
			for(int j=0;j<V.baseFunction.size();++j)
			{
				BaseFunction b = V.getBaseFunction(j,n);

				//double valMf = V.T.integrate(ddot(a,b),n);
				//double valK1f = V.T.integrate(pf(symm(Da),symm(Db)),n);
				//double valP = V.T.integrate(pf(Da,Db),n); //poisson
				double valMf = V.T.integrate(ddot(a.x,b.x),n);
				double valK1f = V.T.integrate(pf(symm(a.dx),symm(b.dx)),n);
				//TODO: aggiungere assert intelligenti qua e la
				assert(a.i < V.spaceDim && b.i < V.spaceDim);
				assert(a.mini_i <= (V.spaceDim-V.nBT) && b.mini_i <= (V.spaceDim-V.nBT));

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
	std::vector<Eigen::Triplet<double>> H;
	Eigen::SparseMatrix<double> sH(1,Q.spaceDim);
	Eigen::Matrix<double,1,Eigen::Dynamic> dH(Q.spaceDim);
	for(int n=0;n<Q.nodes.T.size();++n)
	{
		for(int j=0;j<Q.baseFunction.size();++j)
		{
			BaseFunction b = Q.getBaseFunction(j,n);
			double valH = Q.T.integrate(project(b.x,0),n);
			assert(b.i < Q.spaceDim);
			H.push_back(Eigen::Triplet<double>(0,b.i,valH));
		}
	}

	sH.setFromTriplets(H.begin(),H.end());
	dH = Eigen::Matrix<double,1,Eigen::Dynamic>(sH);
	dH = dH / dH(0);

	Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> sBc0(V.spaceDim,1);
	sBc0=sB.col(0);
	sB = sB - sBc0*dH;
	for(int n=0;n<sB.outerSize();++n)
	{
		for(Eigen::SparseMatrix<double>::InnerIterator it(sB,n);it;++it)
		{
			double valB=it.value();
			if(valB!=0)
			{ 
				int i0 = find(V.notEdge,it.row());
				int j0 = find(Q.notEdge,it.col());
				if(i0!=-1 && j0!=-1)
				{
					C.push_back(Eigen::Triplet<double>(i0,j0+(V.spaceDim-V.nBT),valB));
					C.push_back(Eigen::Triplet<double>(j0+(V.spaceDim-V.nBT),i0,valB));
				}
			}
		}
	}
}

void Simulation::buildStructureMatrices()
{
	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int i=0;i<S.baseFunction.size();++i)
		{
			std::function<dvec(dvec)> a = [&](const dvec &x){return S.baseFunction[i].x(pinv(affineB(n,S.T.mesh))*(x-affineb(n,S.T.mesh)));};
			std::function<dmat(dvec)> Da = [&](const dvec &x){return S.baseFunction[i].dx(pinv(affineB(n,S.T.mesh))*(x-affineb(n,S.T.mesh)))*S.T.Binv[n];};
			int i_=S.getIndex(i,n);
			int i0=S.getMiniIndex(i,n);
			for(int j=0;j<S.baseFunction.size();++j)
			{
				std::function<dvec(dvec)> b = [&](const dvec &x){return S.baseFunction[j].x(pinv(affineB(n,S.T.mesh))*(x-affineb(n,S.T.mesh)));};
				std::function<dmat(dvec)> Db = [&](const dvec &x){return S.baseFunction[j].dx(pinv(affineB(n,S.T.mesh))*(x-affineb(n,S.T.mesh)))*S.T.Binv[n];};
				int j_=S.getIndex(j,n);
				int j0=S.getMiniIndex(j,n);			
	
				double valMs = S.T.integrate(ddot(a,b),n);
				double valKs = S.T.integrate(pf(Da,Db),n);

				Ms.push_back(Eigen::Triplet<double>(i_,j_,valMs));

				if(i0!=-1 && j0!=-1)
				{
					if(valMs!=0)
					{
						C.push_back(Eigen::Triplet<double>(i0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),parameters.deltarho/(parameters.deltat*parameters.deltat)*valMs));
					}
					if(valKs!=0)
					{
						C.push_back(Eigen::Triplet<double>(i0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),parameters.kappa*valKs));
					}
				}
			}
		}
	}
}

void Simulation::buildMultiplierMatrices()
{
	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int i=0;i<S.baseFunction.size();++i)
		{
			std::function<dvec(dvec)> a = [&](const dvec &x){return S.baseFunction[i].x(pinv(affineB(n,S.T.mesh))*(x-affineb(n,S.T.mesh)));};
			std::function<dmat(dvec)> Da = [&](const dvec &x){return S.baseFunction[i].dx(pinv(affineB(n,S.T.mesh))*(x-affineb(n,S.T.mesh)))*S.T.Binv[n];};
			int i_=S.getIndex(i,n);
			int i0=S.getMiniIndex(i,n);
			for(int j=0;j<L.baseFunction.size();++j)
			{
				std::function<dvec(dvec)> b = [&](const dvec &x){return L.baseFunction[j].x(pinv(affineB(n,L.T.mesh))*(x-affineb(n,L.T.mesh)));};
				std::function<dmat(dvec)> Db = [&](const dvec &x){return L.baseFunction[j].dx(pinv(affineB(n,L.T.mesh))*(x-affineb(n,L.T.mesh)))*L.T.Binv[n];};
				int j_=L.getIndex(j,n);
				int j0=L.getMiniIndex(j,n);			
	
				double valLs = -S.T.integrate(ddot(a,b),n);

				Ls.push_back(Eigen::Triplet<double>(i_,j_,valLs));

				if(i0!=-1 && j0!=-1)
				{
					if(valLs!=0)
					{
						C.push_back(Eigen::Triplet<double>(i0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),valLs));
						C.push_back(Eigen::Triplet<double>(j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),i0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),valLs));
					}
				}
			}
		}
	}
}

void Simulation::buildK2f()
{
	//int time = timesteps.size();// - 1;
	for(int n=0;n<V.nodes.T.size();++n)
	{
		F v = V(timesteps[time-1].u,n);
		for(int i=0;i<V.elementDim;++i)
		{
			std::function<dvec(dvec)> a = [&](const dvec &x){return V.baseFunction[i].x(V.T.Binv[n]*(x-V.T.b[n]));};
			std::function<dmat(dvec)> Da = [&](const dvec &x){return V.baseFunction[i].dx(V.T.Binv[n]*(x-V.T.b[n]))*V.T.Binv[n];};
			int i_=V.getIndex(i,n);
			int i0=V.getMiniIndex(i,n);
			for(int j=0;j<V.elementDim;++j)
			{
				std::function<dvec(dvec)> b = [&](const dvec &x){return V.baseFunction[j].x(V.T.Binv[n]*(x-V.T.b[n]));};
				std::function<dmat(dvec)> Db = [&](const dvec &x){return V.baseFunction[j].dx(V.T.Binv[n]*(x-V.T.b[n]))*V.T.Binv[n];};
				int j_=V.getIndex(j,n);
				int j0=V.getMiniIndex(j,n);
				std::function<double(dvec)> g = ddot(dotdiv(v.x,Da),a)-ddot(dotdiv(v.x,Db),b);
				double valK2f = V.T.integrate(g,n);
				if(valK2f!=0)
				{          
					if(i0!=-1 && j0!=-1)
					{
						Ct.push_back(Eigen::Triplet<double>(i0,j0,parameters.rho/2.0*valK2f));
					}
				}  
			}  
		}
	}
}

void Simulation::buildLf()
{
	//int time = timesteps.size();// - 1;
	std::vector<std::vector<dvec>> yyy = S.getValuesInGaussNodes(timesteps[time-1].x);
	MM = V.collisionDetection(yyy);
	S.calc(timesteps[time-1].x);
	for(int n=0;n<S.nodes.T.size();++n)
	{
		F preS = S.getPreCalc(n);
		for(int k=0;k<MM[n].size();++k)
		{
			int m=MM[n][k];
			for(int i=0;i<L.baseFunction.size();++i)
			{
				std::function<dvec(dvec)> a = [&](const dvec &x){return S.baseFunction[i].x(S.T.Binv[n]*(x-S.T.b[n]));};
				std::function<dmat(dvec)> Da = [&](const dvec &x){return S.baseFunction[i].dx(S.T.Binv[n]*(x-S.T.b[n]))*S.T.Binv[n];};
				int i_ = S.getIndex(i,n);
				int i0 = S.getMiniIndex(i,n);

				for(int j=0;j<V.baseFunction.size();++j)
				{
					std::function<dvec(dvec)> b = [&](const dvec &x){return V.baseFunction[j].x(V.T.Binv[m]*(x-V.T.b[m]));};
					std::function<dmat(dvec)> Db = [&](const dvec &x){return V.baseFunction[j].dx(V.T.Binv[m]*(x-V.T.b[m]))*V.T.Binv[m];};
					int j_ = V.getIndex(j,m);
					int j0 = V.getMiniIndex(j,m);

					F dB = {b,Db};
					double valLf = S.T.integrate(pf(Da,compose(dB,preS).dx)+ddot(a,compose(dB,preS).x),n);
					if(valLf!=0)
					{
						Lf.push_back(Eigen::Triplet<double>(i_,j_,valLf));
						if(i0!=-1 && j0!=-1)
						{
						  Ct.push_back(Eigen::Triplet<double>(i0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),j0,valLf));
						  Ct.push_back(Eigen::Triplet<double>(j0,i0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),valLf));
						}
					}
				}    
			}
		}
	}
}

void Simulation::buildF()
{
	int time = timesteps.size();
  	FF = evec::Zero(V.spaceDim);

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
				/*std::function<dvec(dvec)> a = [&](const dvec &x){return V.baseFunction[i].x(V.T.Binv[m]*(x-V.T.b[m]));};
				std::function<dmat(dvec)> Da = [&](const dvec &x){return V.baseFunction[i].dx(V.T.Binv[m]*(x-V.T.b[m]))*V.T.Binv[m];};
				int i_ = V.getIndex(i,m);
				int i0 = V.getMiniIndex(i,m);
				F dA = {a,Da};*/

				BaseFunction a = V.getBaseFunction(i,m);
				double valF = S.T.integrate(pf(S(timesteps[time-1].x,n).dx,compose(a.f,S(timesteps[time-1].x,n)).dx),n); //WAT?

				FF(a.i) += (-parameters.kappa)*valF; //WAT2

				for(int j=0;j<V.baseFunction.size();++j)
				{
					/*std::function<dvec(dvec)> b = [&](const dvec &x){return V.baseFunction[j].x(V.T.Binv[m]*(x-V.T.b[m]));};
					std::function<dmat(dvec)> Db = [&](const dvec &x){return V.baseFunction[j].dx(V.T.Binv[m]*(x-V.T.b[m]))*V.T.Binv[m];};
					int j_ = V.getIndex(j,m);
					int j0 = V.getMiniIndex(j,m);
					F dB = {b,Db};*/

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
	std::cout << "|FF| = " << FF.norm() << std::endl;
}

void Simulation::triplet2sparse()
{
	if(full)
	{
		sC = esmat((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+L.spaceDim,(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+L.spaceDim);
		sCt = esmat((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+L.spaceDim,(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+L.spaceDim);
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
	//int time = timesteps.size();// - 1;
	evec u_1 = vector2eigen(timesteps[time-1].u);
	evec x_1 = vector2eigen(timesteps[time-1].x);
	evec x_2 = vector2eigen(timesteps[time-2].x);

	if(full)
	{
		b = evec::Zero((V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT)+L.spaceDim);
		evec f = parameters.rho/parameters.deltat*sMf*u_1;
		evec o = evec::Zero(Q.spaceDim);
		evec g = parameters.deltarho/(parameters.deltat*parameters.deltat)*sMs*(2.0*x_1-x_2);
		evec d = (-1.0/parameters.deltat)*sLs*(x_1);

		for(int i=0;i<V.spaceDim+Q.spaceDim+S.spaceDim+L.spaceDim;++i)
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
			}
		}
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
	std::vector<int> MMM = V.collisionDetection(Xt);

	std::vector<dvec> u;
	u.reserve(S.nodes.P.size());
	u.resize(S.nodes.P.size());

	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int k=0;k<S.nodes.T[n].size();++k)
		{
			int i = S.nodes.T[n][k];
			int m=MMM[i];
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

timestep Simulation::eigen2timestep(evec a) //TODO: implementare gestione bordi
{
	timestep t;
	int j=0;
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
			t.q.push_back(0);
	}
	for(int i=0;i<S.spaceDim && j<a.size();++i)
	{
		if(find(edge,i+V.spaceDim+Q.spaceDim)==-1)
		{
			t.x.push_back(a(j++));
		}
		else
			t.x.push_back(0);
	}
	for(int i=0;i<L.spaceDim && j<a.size();++i)
	{
		if(find(edge,i+V.spaceDim+Q.spaceDim+S.spaceDim)==-1)
			t.l.push_back(a(j++));
		else
			t.l.push_back(0);
	}
	t.id = id;
	t.time = time; //timesteps.size()
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
		time++; //TODO
	}
	else
	{
		//Eigen::SparseQR<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> solver;
		//Eigen::SparseLU<Eigen::SparseMatrix<double>,Eigen::COLAMDOrdering<int>> solver;
		//Eigen::HouseholderQR<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> solver;
		//Eigen::JacobiSVD<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>> solver;
		//Eigen::SPQR<Eigen::SparseMatrix<double>> solver;
		Eigen::BiCGSTAB<Eigen::SparseMatrix<double>,Eigen::IncompleteLUT<double>> solver;
		//Eigen::SuperLU<Eigen::SparseMatrix<double>> solver;
		//Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,Eigen::Lower|Eigen::Upper> solver;
		std::cout << Eigen::nbThreads() << std::endl;
		solver.compute(sCt);
		evec v = solver.solve(b);
		timesteps.push_back(eigen2timestep(v));
		updateX();
		time++; //TODO
	}
}

