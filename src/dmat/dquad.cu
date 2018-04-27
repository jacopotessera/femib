/*
*	dquad.cu
*/

#include "dquad.h"
#include "dquad_impl.h"

__host__ __device__ dquad::dquad(){}
__host__ __device__ diquad::diquad(){}

__host__ __device__ dquad::dquad(const std::initializer_list<dvec> &list)
{
	if(list.size()<=M_DQUAD)
	{
		for(auto i : list)
		{
			p[size++] = i;
		}
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("dquad: Invalid size!");
	#endif
}

__host__ __device__ diquad::diquad(const std::initializer_list<int> &list)
{
	if(list.size()<=M_DQUAD)
	{
		for(auto i : list)
		{
			p[size++] = i;
		}
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("diquad: Invalid size!");
	#endif
}

__host__ __device__ dvec& dquad::operator()(int row)
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("dquad: Index out of buond!");
	#endif
}

__host__ __device__ dvec dquad::operator()(int row) const
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("dquad: Index out of buond!");
	#endif
}

__host__ __device__ int& diquad::operator()(int row)
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("diquad: Index out of buond!");
	#endif
}

__host__ __device__ int diquad::operator()(int row) const
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("diquad: Index out of buond!");
	#endif
}

std::ostream& operator<<(std::ostream& out, const dquad &t)
{
	out << "dquad size: " << t.size << std::endl << std::endl;
	for(int i=0;i<t.size;++i)
	{
		out << t(i) <<  std::endl;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const diquad &t)
{
	out << "diquad size: " << t.size << std::endl;
	for(int i=0;i<t.size;++i)
	{
		out << t(i) <<  std::endl;
	}
	return out;
}

__host__ __device__ bool operator==(const dquad &lhs, const dquad &rhs)
{
	bool e = (lhs.size == rhs.size);
	for(int i=0;e && i<lhs.size;++i)
	{
		e = e && (lhs(i) == rhs(i));
	}
	return e;
}

__host__ __device__ bool operator==(const diquad &lhs, const diquad &rhs)
{
	bool e = (lhs.size == rhs.size);
	for(int i=0;e && i<lhs.size;++i)
	{
		e = e && (lhs(i) == rhs(i));
	}
	return e;
}

__host__ __device__ double dmax(const dquad &A, int j)
{
	double M = A(0)(j);
	for(int i=1;i<A.size;++i)
	{
		if(A(i)(j)>M)
		{
			M = A(i)(j);
		}
	}
	return M;
}

__host__ __device__ double dmin(const dquad &A, int j)
{
	double m = A(0)(j);
	for(int i=1;i<A.size;++i)
	{
		if(A(i)(j)<m)
		{
			m = A(i)(j);
		}
	}
	return m;
}

__host__  __device__ bool in_box(const dvec &P, const dquad &T)
{
	bool e = true;
	for(int i=0;e && i<P.size;++i)
	{
		e = e && P(i)>(dmin(T,i)-M_EPS) && P(i)<(dmax(T,i)+M_EPS);
	}
	return e;	
}

__host__ __device__ bool in_std(const dvec &P)
{
	bool e = true;
	double q = 0;
	for(int i=0;e && i<P.size;++i)
	{
		e = e && P(i)>=0;
		q += P(i);
	}
	return e && q<=1;
}

__host__ __device__ bool in_triangle(const dvec &P, const dquad &T)
{
	if(P.size == 1 && T.size == 2)
	{
		dvec b[1];
		b[0] = T(1) - T(0);
		dvec p = P - T(0);
	 
		dmat M; M.rows = 1; M.cols = 1;
		for(int i=0;i<M.rows;++i)
		{
			for(int j=0;j<M.cols;++j)
			{
				M(i,j) = b[i](j);
	 		}
		}

		dvec x = inv(M)*p;
		bool res=(x(0) >= 0) && (x(0) <= 1);

		return res; //in_std(x);
	}
	else if(P.size == 2 && T.size == 3)
	{
		dvec b[2];
		b[0] = T(1) - T(0);
		b[1] = T(2) - T(0);
		dvec p = P - T(0);
		dmat M; M.rows = 2; M.cols = 2;
		for(int i=0;i<M.rows;++i)
		{
			for(int j=0;j<M.cols;++j)
			{
				M(i,j) = b[j](i);
	 		}
		}

		dvec x = inv(M)*p;
		bool res=(x(0) >= 0) && (x(1) >= 0) && ((x(0) + x(1)) <= 1);

		return res; //in_std(x);
	}
	else if (P.size == 3 && T.size == 4)
	{
		dvec b[3];
		b[0] = T(1) - T(0);
		b[1] = T(2) - T(0);
		b[2] = T(3) - T(0);
		dvec p = P - T(0);

		dmat M; M.rows = 3; M.cols = 3;
		for(int i=0;i<M.rows;++i)
		{
			for(int j=0;j<M.cols;++j)
			{
				M(i,j) = b[j](i);
	 		}
		}

		dvec x = inv(M)*p;
		bool res=(x(0) >= 0) && (x(1) >= 0) && (x(2) >= 0) && (x(0) + x(1) + x(2) <= 1);

		return res; //in_std(x);
	}
	else
	{
		return false;
	}
}

__host__ __device__ double distancePointPoint(const dvec &P, const dvec &Q)
{
	return std::sqrt(ddot(P-Q,P-Q));
}

__host__ __device__ double distancePointSegment(const dvec &P, const dquad &T)
{
	double P1P2 = ddot(T(1)-T(0),T(1)-T(0));
	double dd = ddot(P-T(0),T(1)-T(0))/P1P2;
	if(dd<0){return ddot(P-T(0),P-T(0));}
	else if (dd>=0 && dd<=1){return ddot(T(0)-P,T(0)-P)-dd*dd*P1P2;}
	else{return ddot(P-T(1),P-T(1));}
}

__host__ __device__ double distancePointTriangle(const dvec &P, const dquad &T)
{
	dvec z = {0,0,0};
	dvec x = T(1)-T(0);
	dvec y = T(2)-T(0);
	dvec p = P-T(0);

	dvec p_ = RR_(x,y)*p;

	dvec xx = PP_(NN_(R_(x)*x));
	dvec yy = PP_(NN_(RR_(x,y)*y));
	dvec zz = PP_(NN_(z));
	dvec pp = PP_(RR_(x,y)*p);

	if(accurate(pp,{xx,yy,zz}))
	{
		return _abs(p_(0));	
	}
	else
	{
		double dxxyy = distancePointSegment(pp,{xx,yy});
		double dyyzz = distancePointSegment(pp,{yy,zz});
		double dzzxx = distancePointSegment(pp,{zz,xx});
		double d = dmin({{dxxyy},{dyyzz},{dzzxx}},0);
		return sqrt(d*d+p_(0)*p_(0));	
	}
}

__host__ __device__ bool accurate(const dvec &P, const dquad &T)
{
	bool N;
	if(not in_box(P,T)){N=0;}
	else
	{
		if(in_triangle(P,T)){N=1;}
		else
		{
			if(P.size == 1 && T.size == 2)
			{
				if(false){N=0;}
				else if(distancePointPoint(P,T(0))<=M_EPS2){N=1;}
				else if(distancePointPoint(P,T(1))<=M_EPS2){N=1;}
				else{N=0;}
			}
			else if(P.size == 2 && T.size == 3)
			{
				if(false){N=0;}
				else if(distancePointSegment(P,{T(0),T(1)})<=M_EPS2){N=1;}
				else if(distancePointSegment(P,{T(1),T(2)})<=M_EPS2){N=1;}
				else if(distancePointSegment(P,{T(2),T(0)})<=M_EPS2){N=1;}
				else{N=0;}
			}
			else if(P.size == 3 && T.size == 4)
			{
				if(false){N=0;}
				else if(distancePointTriangle(P,{T(0),T(1),T(2)})<=M_EPS2){N=1;}
				else if(distancePointTriangle(P,{T(0),T(1),T(3)})<=M_EPS2){N=1;}
				else if(distancePointTriangle(P,{T(1),T(2),T(3)})<=M_EPS2){N=1;}			
				else if(distancePointTriangle(P,{T(2),T(0),T(3)})<=M_EPS2){N=1;}
				else{N=0;}
			}
			else{N=0;}
		}
	}
	return N;
}

__global__ void parallel_accurate(dvec *P,diquad *T,dvec *X,bool *N)
{
	int blockId = blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;

	diquad t=T[threadIdx.x];
	dvec p=X[blockId];
	
	N[threadId] = accurate(p,diquad2dquad(t,P));
}

__host__ bool serial_accurate(dvec *P,diquad T,dvec X)
{
	diquad t=T;
	dvec p=X;

	return accurate(p,diquad2dquad(t,P));
}

__host__ __device__ dquad diquad2dquad(const diquad &T, const dvec *P)
{
	dquad q;
	for(int i=0;i<T.size;i++)
	{
		q(q.size++)= P[T(i)];
	}
	return q;
}

__host__ __device__ dquad[M_QUAD] dquad2dquad(const dquad &T)
{
	dquad[M_QUAD] t;
	if(T.size == 0)
	{

	}
	else if(T.size == 1)
	{

	}
	else if(T.size == 2)
	{

	}
	else if(T.size == 4)
	{

	}
	else if(T.size == 4)
	{

	}
	else
	{

	}
	return t;
}

