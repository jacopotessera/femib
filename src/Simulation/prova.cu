void Simulation::diag(FiniteElementSpace f, int position, std::function<std::function<double(dvec)>(F,F)> g, double coeff)
{
	for(int n=0;n<f.nodes.T.size();++n)
	{
		for(int i=0;i<f.baseFunction.size();++i)
		{
			for(int j=0;j<f.baseFunction.size();++j)
			{
				std::function<dvec(dvec)> a = [&](const dvec &x){return f.baseFunction[i].x(pinv(affineB(n,f.T.mesh))*(x-affineb(n,f.T.mesh)));};
				std::function<dmat(dvec)> Da = [&](const dvec &x){return f.baseFunction[i].dx(pinv(affineB(n,f.T.mesh))*(x-affineb(n,f.T.mesh)))*f.T.Binv[n];};
				std::function<dvec(dvec)> b = [&](const dvec &x){return f.baseFunction[j].x(pinv(affineB(n,f.T.mesh))*(x-affineb(n,f.T.mesh)));};
				std::function<dmat(dvec)> Db = [&](const dvec &x){return f.baseFunction[j].dx(pinv(affineB(n,f.T.mesh))*(x-affineb(n,f.T.mesh)))*f.T.Binv[n];};
				
				F Fa = {a,Da};
				F Fb = {b,Db}; 

				double valM = f.T.integrate(g(Fa,Fb),n);

				int i_=f.getIndex(i,n);
				int j_=f.getIndex(j,n);
				int i0=f.getMiniIndex(i,n);
				int j0=f.getMiniIndex(j,n);
				M.push_back(EigenTriplet(i_,j_,valM));
				if(valM!=0)
				{
					if(i0!=-1 && j0!=-1)
					{
						M_.push_back(EigenTriplet(i0+position,j0+position,coeff*valM));
					}
				}    
			}
		}
	}
}

void Simulation::nonDiag(FiniteElementSpace f1, FiniteElementSpace f2, int position1, int position2, std::function<std::function<double(dvec)>(F,F)> g, double coeff)
{
	for(int n=0;n<f1.nodes.T.size();++n)
	{
		for(int i=0;i<f1.baseFunction.size();++i)
		{
			for(int j=0;j<f2.baseFunction.size();++j)
			{
				std::function<dvec(dvec)> a = [&](const dvec &x){return f1.baseFunction[i].x(pinv(affineB(n,f1.T.mesh))*(x-affineb(n,f.1T.mesh)));};
				std::function<dmat(dvec)> Da = [&](const dvec &x){return f1.baseFunction[i].dx(pinv(affineB(n,f1.T.mesh))*(x-affineb(n,f1.T.mesh)))*f1.T.Binv[n];};
				std::function<dvec(dvec)> b = [&](const dvec &x){return f2.baseFunction[j].x(pinv(affineB(n,f2.T.mesh))*(x-affineb(n,f2.T.mesh)));};
				std::function<dmat(dvec)> Db = [&](const dvec &x){return f2.baseFunction[j].dx(pinv(affineB(n,f2.T.mesh))*(x-affineb(n,f2.T.mesh)))*f2.T.Binv[n];};
				
				F Fa = {a,Da};
				F Fb = {b,Db}; 

				double valM = f1.T.integrate(g(Fa,Fb),n);

				int i_=f.getIndex(i,n);
				int j_=f.getIndex(j,n);
				int i0=f.getMiniIndex(i,n);
				int j0=f.getMiniIndex(j,n);
				M.push_back(EigenTriplet(i_,j_,valM));
				if(valM!=0)
				{
					if(i0!=-1 && j0!=-1)
					{
						M_.push_back(EigenTriplet(i0+position1,j0+position2,coeff*valM));
						M_.push_back(EigenTriplet(j0+position2,i0+position1,coeff*valM));
					}
				}    
			}
		}
	}
}
