/*
*	parallelSimulation.cu
*/

void Simulation::buildFluidMatricesParallel()
{
	static std::mutex mutex;
	int N = this->V.nodes.T.size()/M_THREAD;

	std::vector<Eigen::Triplet<double>> H;

	auto func = [&](int I)
	{
		int min = I*N;
		int max = (I==(M_THREAD-1)) ? V.nodes.T.size() : ((I+1)*N);
	
		std::vector<Eigen::Triplet<double>> pC;
		std::vector<Eigen::Triplet<double>> pMf;
		std::vector<Eigen::Triplet<double>> pK1f;
		std::vector<Eigen::Triplet<double>> pB;
		std::vector<Eigen::Triplet<double>> pH;

		//mutex.lock();
		//std::cout << "Thread (" << I << "): "<< min << " - " << max << std::endl;
		//mutex.unlock();

		for(int n=min;n<max;++n)
		{
			for(int i=0;i<this->V.baseFunction.size();++i)
			{
				BaseFunction a = V.getBaseFunction(i,n);
				for(int j=0;j<this->V.baseFunction.size();++j)
				{
					BaseFunction b = V.getBaseFunction(j,n);
					double valMf = V.T.integrate(ddot(a.x,b.x),n);
					double valK1f = V.T.integrate(pf(symm(a.dx),symm(b.dx)),n);
					assert(a.i < V.spaceDim && b.i < V.spaceDim);
					assert(a.mini_i <= (V.spaceDim-V.nBT) && b.mini_i <= (V.spaceDim-V.nBT));
					if(valMf!=0)
					{
						pMf.push_back(Eigen::Triplet<double>(a.i,b.i,valMf));
						if(a.mini_i!=-1 && b.mini_i!=-1)
						{
							pC.push_back(Eigen::Triplet<double>(a.mini_i,b.mini_i,(parameters.rho/parameters.deltat)*valMf));
						}
					}
					if(valK1f!=0)
					{
						pK1f.push_back(Eigen::Triplet<double>(a.i,b.i,valK1f));
						if(a.mini_i!=-1 && b.mini_i!=-1)
						{
							pC.push_back(Eigen::Triplet<double>(a.mini_i,b.mini_i,(parameters.eta)*valK1f));
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
						pB.push_back(Eigen::Triplet<double>(a.i,b.i,valB));
					}
//					if(i==0)
//					{
//						double valH = Q.T.integrate(project(b.x,0),n);
//						pH.push_back(Eigen::Triplet<double>(0,b.i,valH));
//					}
				}
			}
			for(int j=0;j<Q.baseFunction.size();++j)
			{
				BaseFunction b = Q.getBaseFunction(j,n);
				double valH = Q.T.integrate(project(b.x,0),n);
				assert(b.i < Q.spaceDim);
				pH.push_back(Eigen::Triplet<double>(0,b.i,valH));
			}
		}
		mutex.lock();
		C.insert(std::end(C), std::begin(pC), std::end(pC));
		B.insert(std::end(B), std::begin(pB), std::end(pB));
		Mf.insert(std::end(Mf), std::begin(pMf), std::end(pMf));
		K1f.insert(std::end(K1f), std::begin(pK1f), std::end(pK1f));
		H.insert(std::end(H), std::begin(pH), std::end(pH));
		//std::cout << "(" << I << ") End." << std::endl;		
		//std::cout << "(" << I << "): C.size(): " << this->C.size() << std::endl;
		mutex.unlock();
	};

	std::vector<std::thread> th;

	for(int i=0;i<M_THREAD;i++)
	{
		th.push_back(std::thread(func,i));
	}

	for(auto &t : th){
		t.join();
	}
	Eigen::SparseMatrix<double> sB = Eigen::SparseMatrix<double>(V.spaceDim,Q.spaceDim);
	sB.setFromTriplets(B.begin(),B.end());

	Eigen::SparseMatrix<double> sH(1,Q.spaceDim);
	Eigen::Matrix<double,1,Eigen::Dynamic> dH(Q.spaceDim);

	sH.setFromTriplets(H.begin(),H.end());
	dH = Eigen::Matrix<double,1,Eigen::Dynamic>(sH);
	dH = dH / dH(0);

	Eigen::Matrix<double,Eigen::Dynamic,1> sBc0(V.spaceDim,1);
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

void Simulation::buildStructureMatricesParallel()
{
	static std::mutex mutex;
	int N = this->S.nodes.T.size()/M_THREAD;

	auto func = [&](int I)
	{
		int min = I*N;
		int max = (I==(M_THREAD-1)) ? S.nodes.T.size() : ((I+1)*N);
	
		std::vector<Eigen::Triplet<double>> pC;
		std::vector<Eigen::Triplet<double>> pMs;
		std::vector<Eigen::Triplet<double>> pKs;

		//mutex.lock();
		//std::cout << "Thread (" << I << "): "<< min << " - " << max << std::endl;
		//mutex.unlock();

		for(int n=min;n<max;++n)
		{
			for(int i=0;i<S.baseFunction.size();++i)
			{
				BaseFunction a = S.getBaseFunction(i,n);
				for(int j=0;j<S.baseFunction.size();++j)
				{
					BaseFunction b = S.getBaseFunction(j,n);		
					double valMs = S.T.integrate(ddot(a.x,b.x),n);
					double valKs = S.T.integrate(pf(a.dx,b.dx),n);
					pMs.push_back(Eigen::Triplet<double>(a.i,b.i,valMs));
					if(a.mini_i!=-1 && b.mini_i!=-1)
					{
						if(valMs!=0)
						{
							pC.push_back(Eigen::Triplet<double>(a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),b.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),parameters.deltarho/(parameters.deltat*parameters.deltat)*valMs));
						}
						if(valKs!=0)
						{
							pC.push_back(Eigen::Triplet<double>(a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),b.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),parameters.kappa*valKs));
						}
					}
				}
			}
			mutex.lock();
			C.insert(std::end(C), std::begin(pC), std::end(pC));
			Ms.insert(std::end(Ms), std::begin(pMs), std::end(pMs));
			Ks.insert(std::end(Ks), std::begin(pKs), std::end(pKs));
			//std::cout << "(" << I << ") End." << std::endl;		
			//std::cout << "(" << I << "): C.size(): " << this->C.size() << std::endl;
			mutex.unlock();		
		}
	};
	std::vector<std::thread> th;
	for(int i=0;i<M_THREAD;i++)
	{
		th.push_back(std::thread(func,i));
	}
	for(auto &t : th){
		t.join();
	}
}

void Simulation::buildMultiplierMatricesParallel()
{
	static std::mutex mutex;
	int N = this->S.nodes.T.size()/M_THREAD;

	auto func = [&](int I)
	{
		int min = I*N;
		int max = (I==(M_THREAD-1)) ? S.nodes.T.size() : ((I+1)*N);
	
		std::vector<Eigen::Triplet<double>> pC;
		std::vector<Eigen::Triplet<double>> pLs;

		//mutex.lock();
		//std::cout << "Thread (" << I << "): "<< min << " - " << max << std::endl;
		//mutex.unlock();

		for(int n=min;n<max;++n)
		{
			for(int i=0;i<S.baseFunction.size();++i)
			{
				BaseFunction a = S.getBaseFunction(i,n);
				for(int j=0;j<L.baseFunction.size();++j)
				{
					BaseFunction b = S.getBaseFunction(j,n);
					double valLs = -S.T.integrate(ddot(a.x,b.x),n);
					pLs.push_back(Eigen::Triplet<double>(a.i,b.i,valLs));
					if(a.mini_i!=-1 && b.mini_i!=-1)
					{
						if(valLs!=0)
						{
							pC.push_back(Eigen::Triplet<double>(a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),b.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),valLs));
							pC.push_back(Eigen::Triplet<double>(b.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),valLs));
						}
					}
				}
			}
		}
		mutex.lock();
		C.insert(std::end(C), std::begin(pC), std::end(pC));
		Ls.insert(std::end(pLs), std::begin(pLs), std::end(pLs));
		//std::cout << "(" << I << ") End." << std::endl;		
		//std::cout << "(" << I << "): C.size(): " << this->C.size() << std::endl;
		mutex.unlock();		
	};
	std::vector<std::thread> th;
	for(int i=0;i<M_THREAD;i++)
	{
		th.push_back(std::thread(func,i));
	}
	for(auto &t : th){
		t.join();
	}
}

void Simulation::buildK2fParallel()
{
	static std::mutex mutex;
	int N = this->V.nodes.T.size()/M_THREAD;

	auto func = [&](int I)
	{
		int min = I*N;
		int max = (I==(M_THREAD-1)) ? V.nodes.T.size() : ((I+1)*N);
	
		std::vector<Eigen::Triplet<double>> pCt;

		for(int n=min;n<max;++n)
		{
			F v = V(timesteps[time-1].u,n);
			for(int i=0;i<V.elementDim;++i)
			{
				BaseFunction a = V.getBaseFunction(i,n);
				for(int j=0;j<V.elementDim;++j)
				{
					BaseFunction b = S.getBaseFunction(j,n);
					std::function<double(dvec)> g = ddot(dotdiv(v.x,a.dx),a.x)-ddot(dotdiv(v.x,b.dx),b.x);
					double valK2f = V.T.integrate(g,n);
					if(valK2f!=0)
					{          
						if(a.mini_i!=-1 && b.mini_i!=-1)
						{
							pCt.push_back(Eigen::Triplet<double>(a.mini_i,b.mini_i,(parameters.rho/2.0)*valK2f));
						}
					}  
				}  
			}
		}
		mutex.lock();
		Ct.insert(std::end(Ct), std::begin(pCt), std::end(pCt));
		mutex.unlock();		
	};
	std::vector<std::thread> th;
	for(int i=0;i<M_THREAD;i++)
	{
		th.push_back(std::thread(func,i));
	}
	for(auto &t : th){
		t.join();
	}
}

void Simulation::buildLf()
{
	//int time = timesteps.size();// - 1;
	std::vector<std::vector<dvec>> yyy = S.getValuesInGaussNodes(timesteps[time-1].x);
	MM = V.collisionDetection(yyy);
	S.calc(timesteps[time-1].x);
		
		
	static std::mutex mutex;
	int N = this->S.nodes.T.size()/M_THREAD;

	auto func = [&](int I)
	{
		int min = I*N;
		int max = (I==(M_THREAD-1)) ? S.nodes.T.size() : ((I+1)*N);
		std::vector<Eigen::Triplet<double>> pCt;
		for(int n=min;n<max;++n)
		{
			F preS = S.getPreCalc(n);
			for(int k=0;k<MM[n].size();++k)
			{
				int m=MM[n][k];
				for(int i=0;i<L.baseFunction.size();++i)
				{
					BaseFunction a = S.getBaseFunction(i,n);
					//std::function<dvec(dvec)> a = [&](const dvec &x){return S.baseFunction[i].x(S.T.Binv[n]*(x-S.T.b[n]));};
					//std::function<dmat(dvec)> Da = [&](const dvec &x){return S.baseFunction[i].dx(S.T.Binv[n]*(x-S.T.b[n]))*S.T.Binv[n];};
					//int i_ = S.getIndex(i,n);
					//int i0 = S.getMiniIndex(i,n);
					for(int j=0;j<V.baseFunction.size();++j)
					{
						BaseFunction b = V.getBaseFunction(j,m);
						//std::function<dvec(dvec)> b = [&](const dvec &x){return V.baseFunction[j].x(V.T.Binv[m]*(x-V.T.b[m]));};
						//std::function<dmat(dvec)> Db = [&](const dvec &x){return V.baseFunction[j].dx(V.T.Binv[m]*(x-V.T.b[m]))*V.T.Binv[m];};
						//int j_ = V.getIndex(j,m);
						//int j0 = V.getMiniIndex(j,m);
						//F dB = {b,Db};
						double valLf = S.T.integrate(pf(a.dx,compose(b.f,preS).dx)+ddot(a.x,compose(b.f,preS).x),n);
						if(valLf!=0)
						{
							pLf.push_back(Eigen::Triplet<double>(a.i,b.i,valLf));
							if(a.mini_i!=-1 && b.mini_i!=-1)
							{
							  pCt.push_back(Eigen::Triplet<double>(a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),b.mini_i,valLf));
							  pCt.push_back(Eigen::Triplet<double>(b.mini_i,a.mini_i+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),valLf));
							}
						}
					}    
				}
			}
		}
		mutex.lock();
		Ct.insert(std::end(Ct), std::begin(pCt), std::end(pCt));
		mutex.unlock();		
	};
	std::vector<std::thread> th;
	for(int i=0;i<M_THREAD;i++)
	{
		th.push_back(std::thread(func,i));
	}
	for(auto &t : th){
		t.join();
	}
}

void Simulation::buildFParallel()
{
	int time = timesteps.size();
  	FF = evec::Zero(V.spaceDim);

	std::vector<std::vector<dvec>> yyy = S.getValuesInGaussNodes(timesteps[time-1].x);
	MM = V.collisionDetection(yyy);

	static std::mutex mutex;
	int N = this->S.nodes.T.size()/M_THREAD;

	auto func = [&](int I)
	{
		int min = I*N;
		int max = (I==(M_THREAD-1)) ? S.nodes.T.size() : ((I+1)*N);
		std::vector<Eigen::Triplet<double>> pCt;
		for(int n=min;n<max;++n)
		//for(int n=0;n<S.nodes.T.size();++n)
		{
			for(int k=0;k<MM[n].size();++k)
			{
				int m=MM[n][k];
				assert(m>0);
				for(int i=0;i<V.baseFunction.size();++i)
				{
					BaseFunction a = V.getBaseFunction(i,m);
					double valF = S.T.integrate(pf(S(timesteps[time-1].x,n).dx,compose(a.f,S(timesteps[time-1].x,n)).dx),n);

					pFF(a.i) += (-parameters.kappa)*valF;

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
		mutex.lock();
		Ct.insert(std::end(Ct), std::begin(pCt), std::end(pCt));
		FF += pFF;
		mutex.unlock();		
	};
	std::vector<std::thread> th;
	for(int i=0;i<M_THREAD;i++)
	{
		th.push_back(std::thread(func,i));
	}
	for(auto &t : th){
		t.join();
	}
	std::cout << "|FF| = " << FF.norm() << std::endl;
}


