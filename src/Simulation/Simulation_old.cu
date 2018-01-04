/*void Simulation::buildFF() //poisson
{
	EigenVector temp = EigenVector::Zero(V.spaceDim);
	EigenVector minitemp = EigenVector::Zero((V.spaceDim-V.nBT)/2);
	for(int n=0;n<this->V.nodes.T.size();n++)
	{
		for(int k=0;k<gauss.n;k++)
		{
			for(int i=0;i<this->V.elementDim;i++)
			{
				std::function<dvec(dvec)> forzante = [](dvec v){dvec w; w.size=2;w(0)=v(0)*v(0)-v(1)*v(1);w(1)=0; return w;};
				double val=(this->V.T.d)[n][k]*ddot(this->V.val[i][k],forzante(V.T.B[n]*(gauss.nodes[k])+V.T.b[n]));
				if(val!=0)
				{
					divec iD=index(i,V.elementDim,V.spaceDim,V.T.dim);
					int i0=V.nodes.T[n][iD(0)]+iD(1);
					int j0=find(V.notEdge,i0);
					temp(i0)+=val;
					if(!(j0==-1))
					{
						minitemp(j0)+=val;
					}
				}   
			}
		}
	}
	FF = temp;
	miniFF = minitemp;
}*/

[[deprecated]]
void Simulation::buildFluidMatrices_old(){
	for(int n=0;n<this->V.nodes.T.size();n++){
		for(int k=0;k<V.gauss.n;k++){
			for(int i=0;i<this->V.elementDim;i++){
				for(int j=0;j<this->V.elementDim;j++){
					double valMf=(this->V.T.d)[n][k]*ddot(this->V.val[i][k],this->V.val[j][k]);
					double valK1f=(this->V.T.d)[n][k]*pf(symm(this->V.Dval[i][k]*this->V.T.Binv[n]),symm(this->V.Dval[j][k]*this->V.T.Binv[n]));
					//double valK1f=(this->V.T.d)[n][k]*(dot(V.Dval[i][k].submat(0,0,0,1)*this->V.T.Binv[n],V.Dval[j][k].submat(0,0,0,1)*this->V.T.Binv[n]));//poisson
					if(valMf!=0){
						divec iD=index(i,V.elementDim,V.spaceDim,V.T.dim);
						divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
						int i0=V.nodes.T[n][iD(0)]+iD(1);
						int i1=V.nodes.T[n][jD(0)]+jD(1);
						int j0=find(V.notEdge,i0);
						int j1=find(V.notEdge,i1);
						this->Mf.push_back(EigenTriplet(i0,i1,valMf));
						if(!(j0==-1 || j1==-1)){
							this->C.push_back(EigenTriplet(j0,j1,this->parameters.rho/this->parameters.deltat*valMf));
						}
					}    
					if(valK1f!=0){
						divec iD=index(i,V.elementDim,V.spaceDim,V.T.dim);
						divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
						int i0=V.nodes.T[n][iD(0)]+iD(1);
						int i1=V.nodes.T[n][jD(0)]+jD(1);
						int j0=find(V.notEdge,i0);
						int j1=find(V.notEdge,i1);
						this->Mf.push_back(EigenTriplet(i0,i1,valK1f)); //TODO: ???
						if(!(j0==-1 || j1==-1)){
							this->C.push_back(EigenTriplet(j0,j1,this->parameters.eta*valK1f));
							//this->CC.push_back(EigenTriplet(j0,j1,valK1f));						
						}
					}    
				}
			}
			for(int i=0;i<this->Q.elementDim;i++){
				for(int j=0;j<this->V.elementDim;j++){
					double valB=-(V.T.d)[n][k]*Q.val[i][k](0)*div(V.Dval[j][k]*V.T.Binv[n]);
					if(valB!=0){
						divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
						int i0=Q.nodes.T[n][i];
						int i1=V.nodes.T[n][jD(0)]+jD(1);
						int j0=find(Q.notEdge,i0);
						int j1=find(V.notEdge,i1);
						if(!(j0==-1 || j1==-1)){
							this->C.push_back(EigenTriplet(j0+(V.spaceDim-V.nBT),j1,valB));
							this->C.push_back(EigenTriplet(j1,j0+(V.spaceDim-V.nBT),valB));
						}
					}
				}
			}
		}
	}
}

void Simulation::buildFluidMatrices_old2(){
	for(int n=0;n<this->V.nodes.T.size();n++){
		for(int k=0;k<V.gauss.n;k++){
			for(int i=0;i<this->V.elementDim;i++){
				for(int j=0;j<this->V.elementDim;j++){
					double valMf=(this->V.T.d)[n][k]*ddot(this->V.val[i][k],this->V.val[j][k]);
					double valK1f=(this->V.T.d)[n][k]*pf(symm(this->V.Dval[i][k]*this->V.T.Binv[n]),symm(this->V.Dval[j][k]*this->V.T.Binv[n]));
					//double valK1f=(this->V.T.d)[n][k]*(dot(V.Dval[i][k].submat(0,0,0,1)*this->V.T.Binv[n],V.Dval[j][k].submat(0,0,0,1)*this->V.T.Binv[n]));//poisson
					if(valMf!=0){
						divec iD=index(i,V.elementDim,V.spaceDim,V.T.dim);
						divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
						int i0=V.nodes.T[n][iD(0)]+iD(1);
						int i1=V.nodes.T[n][jD(0)]+jD(1);
						int j0=find(V.notEdge,i0);
						int j1=find(V.notEdge,i1);
						this->Mf.push_back(EigenTriplet(i0,i1,valMf));
						if(!(j0==-1 || j1==-1)){
							//std::cout << "(" << j0 <<"," << j1 <<") = " << this->parameters.rho/this->parameters.deltat*valMf << std::endl;							
							this->C.push_back(EigenTriplet(j0,j1,this->parameters.rho/this->parameters.deltat*valMf));
						}
					}
				}
			}
		}
	}
}

void Simulation::buildStructureMatrices_old()
{
	for(int n=0;n<S.nodes.T.size();n++)
	{
		for(int k=0;k<S.gauss.n;k++)
		{
			for(int i=0;i<S.elementDim;i++)
			{
				for(int j=0;j<S.elementDim;j++)
				{
					double valMs=S.T.d[n][k]*ddot(S.val[i][k],S.val[j][k]);
					double valKs=S.T.d[n][k]*pf(S.Dval[i][k]*S.T.Binv[n],S.Dval[j][k]*S.T.Binv[n]);
					divec iD=index(i,S.elementDim,S.spaceDim,S.T.dim);
					divec jD=index(j,S.elementDim,S.spaceDim,S.T.dim);
					int i0=S.nodes.T[n][iD(0)]+iD(1);
					int i1=S.nodes.T[n][jD(0)]+jD(1);
					int j0=find(S.notEdge,i0);
					int j1=find(S.notEdge,i1);
					Ms.push_back(EigenTriplet(i0,i1,valMs));
					if(valMs!=0)
					{
						if(!(j0==-1 || j1==-1))
						{
							C.push_back(EigenTriplet(j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),j1+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),parameters.deltarho/(parameters.deltat*parameters.deltat)*valMs));
						}
					}    
					if(valKs!=0)
					{
						if(!(j0==-1 || j1==-1))
						{
							C.push_back(EigenTriplet(j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),j1+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),parameters.kappa*valKs));
						}
					}
				}
			}
		}
	}
}

void Simulation::buildMultiplierMatrices_old()
{
	for(int n=0;n<L.nodes.T.size();n++)
	{

		for(int k=0;k<L.gauss.n;k++)
		{
			for(int i=0;i<L.elementDim;i++)
			{
				for(int j=0;j<L.elementDim;j++)
				{
					double valLs=S.T.d[n][k]*(pf(S.Dval[i][k]*S.T.Binv[n],S.Dval[j][k]*S.T.Binv[n])+ddot(S.val[i][k],S.val[j][k]));
					if(valLs!=0)
					{
						divec iD=index(i,L.elementDim,L.spaceDim,L.T.dim);
						divec jD=index(j,L.elementDim,L.spaceDim,L.T.dim);
						int i0=L.nodes.T[n][iD(0)]+iD(1);
						int i1=L.nodes.T[n][jD(0)]+jD(1);
						int j0=find(S.notEdge,i0);
						int j1=find(S.notEdge,i1);
						Ls.push_back(EigenTriplet(i0,i1,valLs));
						if(!(j0==-1 || j1==-1))
						{
							C.push_back(EigenTriplet(j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),j1+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),(-1.0)*valLs));
							C.push_back(EigenTriplet(j1+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT),(-1.0)/parameters.deltat*valLs));
						}
					}
				}
			}
		}
	}
}


void Simulation::buildK2f_old()
{ //TODO: gestire 1d e 3d
	for(int n=0;n<V.nodes.T.size();++n)
	{
		for(int k=0;k<V.gauss.n;++k)
		{
			xDx vdv;
			vdv.x.size = 2; 
			vdv.x(0) = 0; vdv.x(1) = 0;

			for(int l=0;l<V.elementDim;++l)
			{
				divec lD=index(l,V.elementDim,V.spaceDim,V.T.dim);
				vdv.x=vdv.x+timesteps[time-1].u[V.nodes.T[n][lD(0)]+lD(1)]*V.val[l][k];
			}

			for(int i=0;i<V.elementDim;++i)
			{
				for(int j=0;j<V.elementDim;++j)
				{
					double valK2f=V.T.d[n][k]*(ddot(dotdiv(vdv.x,V.Dval[i][k]*V.T.Binv[n]),V.val[j][k])-ddot(dotdiv(vdv.x,V.Dval[j][k]*V.T.Binv[n]),V.val[i][k]));
					if(valK2f!=0)
					{          
						divec iD=index(i,V.elementDim,V.spaceDim,V.T.dim);
						divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
						int i0=V.nodes.T[n][iD(0)]+iD(1);
						int i1=V.nodes.T[n][jD(0)]+jD(1);
						int j0=find(V.notEdge,i0);
						int j1=find(V.notEdge,i1);
						if(!(j0==-1 || j1==-1))
						{
						  Ctime.push_back(EigenTriplet(j0,j1,parameters.rho/2.0*valK2f));
						}
					}  
				}  
			}
		}
	}
}


void Simulation::buildLf_old()
{
std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();
////////////////// CUDA TIME 2 ////////////////////////////
	//xDx ydy[S.nodes.T.size()][gauss.n]; //TODO??
	dvec dX_sp[S.nodes.T.size()][S.gauss.n];

	/**/int q = V.nodes.T.size()/1024;
	/**/int mod = V.nodes.T.size()%1024;

	/*if(q>0)*/bool N[q][S.nodes.T.size()][S.gauss.n][1024];
	/*if(mod>0)*/bool Nq[S.nodes.T.size()][S.gauss.n][mod];

	for(int n=0;n<S.nodes.T.size();n++){
		std::vector<xDx> temp;
		ydy.push_back(temp);
		for(int k=0;k<S.gauss.n;k++){
			xDx temp2;
			temp2.x.size=2;
			temp2.x(0)=0; temp2.x(1)=0;
			temp2.dx.rows=2; temp2.dx.cols=2;
			temp2.dx(0,0)=0; temp2.dx(0,1)=0;
			temp2.dx(1,0)=0; temp2.dx(1,1)=0;
			ydy[n].push_back(temp2);
			for(int l=0;l<S.elementDim;l++){
				divec lD=index(l,S.elementDim,S.spaceDim,S.T.dim);
				//ydy[n][k].x=ydy[n][k].x+timesteps[time-1].x[S.nodes.T[n](lD(0))+lD(1)]*S.baseFunction.f[l](S.T.Binv[n]*(S.T.p[n][k]-S.T.b[n]));
				//ydy[n][k].dx=ydy[n][k].dx+timesteps[time-1].x[S.nodes.T[n](lD(0))+lD(1)]*S.baseFunction.Df[l](S.T.Binv[n]*(S.T.p[n][k]-S.T.b[n]))*S.T.Binv[n];
				ydy[n][k].x=ydy[n][k].x+timesteps[time-1].x[S.nodes.T[n][lD(0)]+lD(1)]*S.baseFunction[l].x(S.T.Binv[n]*(S.T.p[n][k]-S.T.b[n]));
				ydy[n][k].dx=ydy[n][k].dx+timesteps[time-1].x[S.nodes.T[n][lD(0)]+lD(1)]*S.baseFunction[l].dx(S.T.Binv[n]*(S.T.p[n][k]-S.T.b[n]))*S.T.Binv[n];
						
			}
			
			dX_sp[n][k]=ydy[n][k].x;
		}
	}
	dvec *dev_X_sp;
	/*/if(q>0)*/bool *dev_N[q];
	/*if(mod>0)*/bool *dev_Nq;
	HANDLE_ERROR(cudaMalloc((void**)&dev_X_sp,S.nodes.T.size()*S.gauss.n*sizeof(dvec)));
	/**/for(int i=0;i<q;i++)HANDLE_ERROR(cudaMalloc((void**)&dev_N[i],S.nodes.T.size()*S.gauss.n*1024*sizeof(bool)));
	/**/if(mod>0)HANDLE_ERROR(cudaMalloc((void**)&dev_Nq,S.nodes.T.size()*S.gauss.n*mod*sizeof(bool)));
	HANDLE_ERROR(cudaMemcpy(dev_X_sp,dX_sp,S.nodes.T.size()*S.gauss.n*sizeof(dvec),cudaMemcpyHostToDevice));
	for(int i=0;i<q;i++){
		parallel_accurate<<<S.nodes.T.size()*S.gauss.n,1024>>>(V.T.devP,V.T.devT[i],dev_X_sp,dev_N[i]);
		HANDLE_ERROR(cudaMemcpy(N[i],dev_N[i],S.nodes.T.size()*S.gauss.n*1024*sizeof(bool),cudaMemcpyDeviceToHost));}
	if(mod>0){
		parallel_accurate<<<S.nodes.T.size()*S.gauss.n,mod>>>(V.T.devP,V.T.devTq,dev_X_sp,dev_Nq);
		HANDLE_ERROR(cudaMemcpy(Nq,dev_Nq,S.nodes.T.size()*S.gauss.n*mod*sizeof(bool),cudaMemcpyDeviceToHost));}

	cudaFree(dev_X_sp);
	for(int i=0;i<q;i++)cudaFree(dev_N[i]);
	if(mod>0)cudaFree(dev_Nq);
	
	for(int n=0;n<S.nodes.T.size();n++){
		for(int k=0;k<S.gauss.n;k++){
			M[n][k]=-1;
			for(int i=0;i<q;i++){
				for(int m=0;m<1024;m++){
					if(N[i][n][k][m]==1){
						this->M[n][k]=m+i*1024;
						break;
					}
				}
			} 
		}
	}

	for(int n=0;n<S.nodes.T.size();n++){
		for(int k=0;k<S.gauss.n;k++){
			for(int m=0;m<mod;m++){
				if(Nq[n][k][m]==1){
					M[n][k]=m+q*1024;
					break;
				}
			}
		}
	}
std::chrono::high_resolution_clock::time_point c2 = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>( c2 - c1 ).count();
std::cout << "old " << duration << std::endl;

////////////////////////////////////////////////////////////
	for(int n=0;n<L.nodes.T.size();n++){
		for(int k=0;k<L.gauss.n;k++){
			int m=M[n][k];
			for(int j=0;j<V.elementDim;j++){
				divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
				xDx vdv;
				vdv.x=V.baseFunction[j].x(V.T.Binv[m]*(ydy[n][k].x-V.T.b[m]));
				vdv.dx=V.baseFunction[j].dx(V.T.Binv[m]*(ydy[n][k].x-V.T.b[m]))*V.T.Binv[m];
				for(int i=0;i<L.elementDim;i++){
					double valLf=S.T.d[n][k]*(pf(S.Dval[i][k]*S.T.Binv[n],vdv.dx*ydy[n][k].dx)+ddot(S.val[i][k],vdv.x));
					if(valLf!=0){
						divec iD=index(i,L.elementDim,L.spaceDim,L.T.dim);
						int i0=L.nodes.T[n][iD(0)]+iD(1);
						int i1=V.nodes.T[n][jD(0)]+jD(1);
						int j0=find(S.notEdge,i0);
						int j1=find(V.notEdge,i1);
						Lf.push_back(EigenTriplet(i0,i1,valLf));
						if(!(j0==-1 || j1==-1)){
						  Ctime.push_back(EigenTriplet(j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),j1,valLf));
						  Ctime.push_back(EigenTriplet(j1,j0+(V.spaceDim-V.nBT)+(Q.spaceDim-Q.nBT)+(S.spaceDim-S.nBT),valLf));
						}
					}
				}       
			}
		}
	}
}


//COLLISION DETECTION
	/*dvec dX_sp[S.nodes.T.size()][S.T.gauss.n];

	int q = V.nodes.T.size()/MAX_BLOCKS;
	int mod = V.nodes.T.size()%MAX_BLOCKS;

	bool N[q][S.nodes.T.size()][S.T.gauss.n][MAX_BLOCKS];
	bool Nq[S.nodes.T.size()][S.T.gauss.n][mod];

	for(int n=0;n<S.nodes.T.size();++n)
	{
		std::vector<xDx> temp;
		ydy.push_back(temp);
		for(int k=0;k<S.T.gauss.n;++k)
		{
			ydy[n].push_back(S(timesteps[time-1].x,n)(S.T.B[n]*S.T.gauss.nodes[k]+S.T.b[n]));
			dX_sp[n][k]=ydy[n][k].x;
		}
	}

	dvec *dev_X_sp;
	bool *dev_N[q];
	bool *dev_Nq;

	HANDLE_ERROR(cudaMalloc((void**)&dev_X_sp,S.nodes.T.size()*S.T.gauss.n*sizeof(dvec)));
	HANDLE_ERROR(cudaMemcpy(dev_X_sp,dX_sp,S.nodes.T.size()*S.T.gauss.n*sizeof(dvec),cudaMemcpyHostToDevice));

	for(int i=0;i<q;++i)
		HANDLE_ERROR(cudaMalloc((void**)&dev_N[i],S.nodes.T.size()*S.T.gauss.n*MAX_BLOCKS*sizeof(bool)));
	if(mod>0)
		HANDLE_ERROR(cudaMalloc((void**)&dev_Nq,S.nodes.T.size()*S.T.gauss.n*mod*sizeof(bool)));

	for(int i=0;i<q;++i)
	{
		parallel_accurate<<<S.nodes.T.size()*S.T.gauss.n,MAX_BLOCKS>>>(V.T.devP,V.T.devT[i],dev_X_sp,dev_N[i]);
		HANDLE_ERROR(cudaMemcpy(N[i],dev_N[i],S.nodes.T.size()*S.T.gauss.n*MAX_BLOCKS*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	if(mod>0)
	{
		parallel_accurate<<<S.nodes.T.size()*S.T.gauss.n,mod>>>(V.T.devP,V.T.devTq,dev_X_sp,dev_Nq);
		HANDLE_ERROR(cudaMemcpy(Nq,dev_Nq,S.nodes.T.size()*S.T.gauss.n*mod*sizeof(bool),cudaMemcpyDeviceToHost));
	}

//std::cout << "serial " << serial_accurate(V.T.mesh.P.data(),V.T.mesh.T[i],dX_sp[0][0]) << std::endl;

	cudaFree(dev_X_sp);
	for(int i=0;i<q;++i)cudaFree(dev_N[i]);
	if(mod>0)cudaFree(dev_Nq);
	
	MM.clear();

	for(int n=0;n<S.nodes.T.size();++n)
	{
		std::vector<int> temp;
		MM.push_back(temp);
		for(int k=0;k<gauss.n;++k)
		{
			M[n][k]=-1;
			for(int i=0;i<q;++i)
			{
				for(int m=0;m<MAX_BLOCKS;++m)
				{
					if(N[i][n][k][m]==1)
					{
						if(find(MM[n],m+i*MAX_BLOCKS)==-1)
							MM[n].push_back(m+i*MAX_BLOCKS);
						this->M[n][k]=m+i*MAX_BLOCKS;
						break;
					}
				}
			} 
		}
	}

	for(int n=0;n<S.nodes.T.size();++n)
	{
		std::vector<int> temp;
		MM.push_back(temp);
		for(int k=0;k<gauss.n;++k)
		{
			for(int m=0;m<mod;++m)
			{
				if(Nq[n][k][m]==1)
				{
					if(find(MM[n],m+q*MAX_BLOCKS)==-1)					
						MM[n].push_back(m+q*MAX_BLOCKS);
					M[n][k]=m+q*MAX_BLOCKS;
					break;
				}
			}
		}
	}*/

//double valLf = S.T.integrate(pf(Da,compose(dB,S(timesteps[time-1].x,n)).dx)+ddot(a,b),n);
//double valLf = 0;
//double valLf = S.T.integrate(pf(Da,Da),n);
//double valLf = S.T.integrate(pf(Da,compose(V.functions[j_],S(timesteps[time-1].x,n)).dx)+ddot(a,V.functions[j_].x),n);
//double valLf=S.T.d[n][k]*(pf(S.Dval[i][k]*S.T.Binv[n],vdv.dx*ydy[n][k].dx)+ddot(S.val[i][k],vdv.x));

void Simulation::buildF_old()
{
  	FF = Eigen::Matrix<double,Eigen::Dynamic,1>::Zero(V.spaceDim);
	for(int n=0;n<S.nodes.T.size();n++){
		for(int k=0;k<S.gauss.n;k++){
			int m=M[n][k];
			for(int i=0;i<V.elementDim;i++){
				divec iD=index(i,V.elementDim,V.spaceDim,V.T.dim);
				int i0=V.nodes.T[m][iD(0)]+iD(1);
				xDx v_i;
				v_i.x=V.baseFunction[i].x(V.T.Binv[m]*(ydy[n][k].x-V.T.b[m]));
				v_i.dx=V.baseFunction[i].dx(V.T.Binv[m]*(ydy[n][k].x-V.T.b[m]))*V.T.Binv[m];
				double valF=-pf(ydy[n][k].dx,v_i.dx*ydy[n][k].dx);
				FF(i0)+=valF;
				for(int j=0;j<V.elementDim;j++){
					divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
					xDx v_j;
					v_j.x=V.baseFunction[j].x(V.T.Binv[m]*(ydy[n][k].x-V.T.b[m]));
					v_j.dx=V.baseFunction[j].dx(V.T.Binv[m]*(ydy[n][k].x-V.T.b[m]))*V.T.Binv[m];

					double valMB=S.T.d[n][k]*ddot(v_i.x,v_j.x);

					if(valMB!=0){
						int i1=V.nodes.T[n][jD(0)]+jD(1);
						int j0=find(V.notEdge,i0);
						int j1=find(V.notEdge,i1);
						MB.push_back(EigenTriplet(i0,i1,valMB));
						if(!(j0==-1 || j1==-1)){
							Ctime.push_back(EigenTriplet(j0,j1,parameters.deltarho/parameters.deltat*valMB));
						}
					}
				}
			}
		}
	}
}

void Simulation::updateX_old()
{
////////////////////////////////////////////////////////////////////////////////
//CUDA
	int q = V.nodes.T.size()/MAX_BLOCKS;
	int mod = V.nodes.T.size()%MAX_BLOCKS;
	dvec X_t[S.spaceDim/S.T.dim];
	for(int i=0;i<S.spaceDim/S.T.dim;++i) //TODO
	{
		X_t[i].size = 2;
		X_t[i](0)=timesteps[time-1].x[i];
		X_t[i](1)=timesteps[time-1].x[i+S.spaceDim/S.T.dim];
	}

	dvec *dev_X_t;
	bool *dev_NN[q];
	bool *dev_NNq;
	bool NN[q][S.spaceDim/S.T.dim][MAX_BLOCKS];
	bool NNq[S.spaceDim/S.T.dim][mod];
	HANDLE_ERROR(cudaMalloc((void**)&dev_X_t,S.spaceDim/S.T.dim*sizeof(dvec)));
	for(int i=0;i<q;++i)HANDLE_ERROR(cudaMalloc((void**)&dev_NN[i],S.spaceDim/S.T.dim*MAX_BLOCKS*sizeof(bool)));
	if(mod>0)HANDLE_ERROR(cudaMalloc((void**)&dev_NNq,S.spaceDim/S.T.dim*mod*sizeof(bool)));
	HANDLE_ERROR(cudaMemcpy(dev_X_t,X_t,S.spaceDim/S.T.dim*sizeof(dvec),cudaMemcpyHostToDevice));

	for(int i=0;i<q;++i)
	{
		parallel_accurate<<<S.spaceDim/S.T.dim,MAX_BLOCKS>>>(V.T.devP,V.T.devT[i],dev_X_t,dev_NN[i]); //S.T.devT?????
		HANDLE_ERROR(cudaMemcpy(NN[i],dev_NN[i],S.spaceDim/S.T.dim*MAX_BLOCKS*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	if(mod>0)
	{
		parallel_accurate<<<S.spaceDim/S.T.dim,mod>>>(V.T.devP,V.T.devTq,dev_X_t,dev_NNq); //S.T.devT?????
		HANDLE_ERROR(cudaMemcpy(NNq,dev_NNq,S.spaceDim/S.T.dim*mod*sizeof(bool),cudaMemcpyDeviceToHost));
	}


	cudaFree(dev_X_t);
	for(int i=0;i<q;++i)cudaFree(dev_NN[i]);
	if(mod>0)cudaFree(dev_NNq);

	int MM[S.spaceDim/S.T.dim];

	for(int n=0;n<S.spaceDim/S.T.dim;++n){
		MM[n]=-1;
		for(int i=0;i<q;++i){
			for(int m=0;m<MAX_BLOCKS;++m){
				if(NN[i][n][m]==1){
					MM[n]=m+i*MAX_BLOCKS;
					break;
				}
			}
		} 
	}

	for(int n=0;n<S.spaceDim/S.T.dim;n++){
		for(int m=0;m<mod;m++){
			if(NNq[n][m]==1){
				MM[n]=m+q*MAX_BLOCKS;
				break;
			}
		} 
	}
////////////////////////////////////////////////////////////////////////////////
	std::cout << "TEST!! ELIMINA QUESTA LINEA!!" << std::endl;
	
	std::vector<double> uu; for(int i=0;i<V.spaceDim;i++)uu.push_back(0);
	timestep newTimestep;	
	timesteps.push_back(newTimestep);
	timesteps[time].u=uu;
	std::vector<dvec> vel_x;
	for(int i=0;i<S.spaceDim/S.T.dim;i++){
		int m=MM[i];
		dvec t; t.size = 2;
		t(0) = 0; t(1) = 0;
		vel_x.push_back(t);
		dvec XXX; XXX.size=2; 
		XXX(0)=timesteps[time-1].x[i];
		XXX(1)=timesteps[time-1].x[i+S.spaceDim/S.T.dim];
		for(int j=0;j<V.elementDim;j++){
			divec jD=index(j,V.elementDim,V.spaceDim,V.T.dim);
			int i1=V.nodes.T[m][jD(0)]+jD(1);
			vel_x[i]=vel_x[i]+timesteps[time].u[i1]*V.baseFunction[j].x(V.T.Binv[m]*(XXX-V.T.b[m]));
		}     
	}
	std::vector<double> vv(S.spaceDim);
	for(int i=0;i<S.spaceDim/S.T.dim;i++){
		vv[i]=vel_x[i](0);
		vv[i+S.spaceDim/S.T.dim]=vel_x[i](1);
	}

	timesteps[time].x = timesteps[time-1].x+parameters.deltat*vv;
}

void Simulation::updateX_newold()
{
////////////////////////////////////////////////////////////////////////////////
//CUDA
/*	int q = V.nodes.T.size()/MAX_BLOCKS;
	int mod = V.nodes.T.size()%MAX_BLOCKS;
	dvec X_t[S.spaceDim/S.T.dim];

	for(int i=0;i<S.nodes.P.size();++i) //TODO
	{
		X_t[i]=S(timesteps[time-1].x).x(S.nodes.P[i]);
	}*/

	std::vector<dvec> X_t = S.getValuesInMeshNodes(timesteps[time-1].x);

	/*dvec *dev_X_t;
	bool *dev_NN[q];
	bool *dev_NNq;
	bool NN[q][S.spaceDim/S.T.dim][MAX_BLOCKS];
	bool NNq[S.spaceDim/S.T.dim][mod];
	HANDLE_ERROR(cudaMalloc((void**)&dev_X_t,S.spaceDim/S.T.dim*sizeof(dvec)));
	for(int i=0;i<q;++i)HANDLE_ERROR(cudaMalloc((void**)&dev_NN[i],S.spaceDim/S.T.dim*MAX_BLOCKS*sizeof(bool)));
	if(mod>0)HANDLE_ERROR(cudaMalloc((void**)&dev_NNq,S.spaceDim/S.T.dim*mod*sizeof(bool)));
	HANDLE_ERROR(cudaMemcpy(dev_X_t,X_t,S.spaceDim/S.T.dim*sizeof(dvec),cudaMemcpyHostToDevice));

	for(int i=0;i<q;++i)
	{
		parallel_accurate<<<S.spaceDim/S.T.dim,MAX_BLOCKS>>>(V.T.devP,V.T.devT[i],dev_X_t,dev_NN[i]); //S.T.devT?????
		HANDLE_ERROR(cudaMemcpy(NN[i],dev_NN[i],S.spaceDim/S.T.dim*MAX_BLOCKS*sizeof(bool),cudaMemcpyDeviceToHost));
	}
	if(mod>0)
	{
		parallel_accurate<<<S.spaceDim/S.T.dim,mod>>>(V.T.devP,V.T.devTq,dev_X_t,dev_NNq); //S.T.devT?????
		HANDLE_ERROR(cudaMemcpy(NNq,dev_NNq,S.spaceDim/S.T.dim*mod*sizeof(bool),cudaMemcpyDeviceToHost));
	}


	cudaFree(dev_X_t);
	for(int i=0;i<q;++i)cudaFree(dev_NN[i]);
	if(mod>0)cudaFree(dev_NNq);

	int MM[S.spaceDim/S.T.dim];

	for(int n=0;n<S.spaceDim/S.T.dim;++n){
		MM[n]=-1;
		for(int i=0;i<q;++i){
			for(int m=0;m<MAX_BLOCKS;++m){
				if(NN[i][n][m]==1){
					MM[n]=m+i*MAX_BLOCKS;
					break;
				}
			}
		} 
	}

	for(int n=0;n<S.spaceDim/S.T.dim;++n){
		for(int m=0;m<mod;++m){
			if(NNq[n][m]==1){
				MM[n]=m+q*MAX_BLOCKS;
				break;
			}
		} 
	}*/
	std::vector<int> MMM = V.collisionDetection(X_t);
////////////////////////////////////////////////////////////////////////////////
	std::cout << "TEST!! ELIMINA QUESTA LINEA!!" << std::endl;

	std::vector<dvec> u;
	u.reserve(S.nodes.P.size());
	u.resize(S.nodes.P.size());
	for(int n=0;n<S.nodes.T.size();++n)
	{
		for(int k=0;k<S.nodes.T[n].size();++k)
		{
			int i = S.nodes.T[n][k];
			int m=MMM[i];

			u[i]= (compose(V(timesteps[time].u,m),S(timesteps[time-1].x,n)).x(S.nodes.P[i]));
		}
	}
	std::vector<double> uu(S.spaceDim);
	for(int i=0;i<S.spaceDim/S.T.dim;++i)
	{
		for(int j=0;j<S.T.dim;++j)
		{
			uu[i+j*S.spaceDim/S.T.dim]=u[i](j);
		}
	}
	timesteps[time].x = timesteps[time-1].x+parameters.deltat*uu;
}

