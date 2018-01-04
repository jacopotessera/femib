/*
*	dmat.cu
*/

#include "dmat.h"
#include "dmat_impl.h"

__host__ __device__ dvec::dvec(){}
__host__ __device__ divec::divec(){}
__host__ __device__ dmat::dmat(){}
__host__ __device__ dtrian::dtrian(){}
__host__ __device__ ditrian::ditrian(){}

/*__host__ __device__ dvec::dvec(int s)
{
	if(s<=M_DVEC)
	{
		size = s;
		for(int i=0;i<=M_DVEC;++i)
		{
			v[i] = 0;	
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dvec: Invalid size!");
	#endif
}*/

__host__ __device__ dvec::dvec(const std::initializer_list<double> &list)
{
	if(list.size()<=M_DVEC)
	{
		for(auto i : list)
		{
			v[size++] = i;
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dvec: Invalid size!");
	#endif
}

__host__ __device__ divec::divec(const std::initializer_list<int> &list)
{
	if(list.size()<=M_DVEC)
	{
		for(auto i : list)
		{
			v[size++] = i;
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("divec: Invalid size!");
	#endif
}

/*__host__ __device__ dmat::dmat(int r, int c)
{
	if(r<=M_DVEC && c <=M_DVEC)
	{
		rows = r;
		cols = c;
		for(int i=0;i<=M_DVEC;++i)
		{
			for(int j=0;j<=M_DVEC;++j)
			{
				m[i][j] = 0;	
			}	
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dmat: Invalid size!");
	#endif
}*/

__host__ __device__ dmat::dmat(const std::initializer_list<std::initializer_list<double> > &list)
{
	rows = 0; cols = 0;
	for (auto i = list.begin(); i != list.end() && i-list.begin()<M_DVEC; i++)
	{
		++rows;
		cols = 0;
		for (auto j = i->begin(); j != i->end() && j-i->begin()<M_DVEC; j++)
		{
			++cols;
			m[i-list.begin()][j-i->begin()] = *j;
		}	
	}
}

__host__ __device__ dtrian::dtrian(const std::initializer_list<dvec> &list)
{
	if(list.size()<=M_DTRIAN)
	{
		for(auto i : list)
		{
			p[size++] = i;
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dtrian: Invalid size!");
	#endif
}

__host__ __device__ ditrian::ditrian(const std::initializer_list<int> &list)
{
	if(list.size()<=M_DTRIAN)
	{
		for(auto i : list)
		{
			p[size++] = i;
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("ditrian: Invalid size!");
	#endif
}

__host__ __device__ double& dvec::operator()(int row)
{
	if(row<size)return v[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dvec: Index out of buond!");
	#endif
}

__host__ __device__ double dvec::operator()(int row) const
{
	if(row<size)return v[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dvec: Index out of buond!");
	#endif
}

__host__ __device__ int& divec::operator()(int row)
{
	if(row<size)return v[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("divec: Index out of buond!");
	#endif
}

__host__ __device__ int divec::operator()(int row) const
{
	if(row<size)return v[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("divec: Index out of buond!");
	#endif
}

__host__ __device__ double& dmat::operator()(int row, int col)
{
	if(row<rows && col<cols)return m[row][col];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dmat: Index out of buond!");
	#endif
}

__host__ __device__ double dmat::operator()(int row, int col) const
{
	if(row<rows && col<cols)return m[row][col];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dmat: Index out of buond!");
	#endif
}

__host__ __device__ dvec& dtrian::operator()(int row)
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dtrian: Index out of buond!");
	#endif
}

__host__ __device__ dvec dtrian::operator()(int row) const
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dtrian: Index out of buond!");
	#endif
}

__host__ __device__ int& ditrian::operator()(int row)
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("ditrian: Index out of buond!");
	#endif
}

__host__ __device__ int ditrian::operator()(int row) const
{
	if(row<size)return p[row];
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("ditrian: Index out of buond!");
	#endif
}

std::ostream& operator<<(std::ostream& out, const dvec &v)
{
	std::cout << "dvec size: " << v.size << std::endl;
	for(int i=0;i<v.size;++i)
	{
		std::cout << v(i) <<  std::endl;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const divec &v)
{
	std::cout << "divec size: " << v.size << std::endl;
	for(int i=0;i<v.size;++i)
	{
		std::cout << v(i) <<  std::endl;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const dmat &A)
{
	std::cout << "dmat size: " << A.rows << "x" << A.cols << std::endl;
	for(int i=0;i<A.rows;++i)
	{
		for(int j=0;j<A.cols;++j)
		{
			std::cout << A(i,j) << " ";
		}
		std::cout << std::endl;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const dtrian &t)
{
	std::cout << "dtrian size: " << t.size << std::endl << std::endl;
	for(int i=0;i<t.size;++i)
	{
		std::cout << t(i) <<  std::endl;
	}
	return out;
}

std::ostream& operator<<(std::ostream& out, const ditrian &t)
{
	std::cout << "ditrian size: " << t.size << std::endl;
	for(int i=0;i<t.size;++i)
	{
		std::cout << t(i) <<  std::endl;
	}
	return out;
}

__host__ __device__ double _abs(double x)
{
	if(x>=0){return x;}
	else{return -x;}
}

__host__ __device__ bool operator==(const dvec &lhs, const dvec &rhs)
{
	bool e = (lhs.size == rhs.size);
	for(int i=0;e && i<lhs.size;++i)
	{
		e = e && (_abs(lhs(i)-rhs(i))<M_EPS);
	}
	return e;
}

/*__host__ size_t Hash()
{
	size_t h = std::hash<int>(size);
	for(int i=0;i<M_DVEC;i++)
	{	
		hashCombine(h,std::hash<double>(v[i]));
	}
	return h
}*/

__host__ __device__ bool operator<(const dvec &lhs, const dvec &rhs)
{
	if(lhs.size == rhs.size)
	{
		bool e = true;
		for(int i=0;e && i<lhs.size;++i)		
			e = e && (lhs(i)<rhs(i));
		return e;
	}
	else if(lhs.size<rhs.size)
		return true;
	else if(lhs.size>rhs.size)
		return false;
	else
		return false;
}

__host__ __device__ bool operator==(const divec &lhs, const divec &rhs)
{
	bool e = (lhs.size == rhs.size);
	for(int i=0;e && i<lhs.size;++i)
	{
		e = e && (lhs(i) == rhs(i));
	}
	return e;
}

__host__ __device__ bool operator==(const dmat &lhs, const dmat &rhs)
{
	bool e = (lhs.rows == rhs.rows && lhs.cols == rhs.cols);
	for(int i=0;e && i<lhs.rows;++i)
	{
		for(int j=0;e && j<lhs.cols;++j)
		{
			e = e && (_abs(lhs(i,j)-rhs(i,j))<M_EPS);
		}
	}
	return e;
}

__host__ __device__ bool operator==(const dtrian &lhs, const dtrian &rhs)
{
	bool e = (lhs.size == rhs.size);
	for(int i=0;e && i<lhs.size;++i)
	{
		e = e && (lhs(i) == rhs(i));
	}
	return e;
}

__host__ __device__ bool operator==(const ditrian &lhs, const ditrian &rhs)
{
	bool e = (lhs.size == rhs.size);
	for(int i=0;e && i<lhs.size;++i)
	{
		e = e && (lhs(i) == rhs(i));
	}
	return e;
}

__host__ __device__ dvec operator*(double a, const dvec &v)
{
	dvec c;
	c.size = v.size;
	for(int i=0;i<v.size;++i)
	{
		c(i) = a*v(i);
	}
	return c;
}

__host__ __device__ dmat& dmat::operator*=(double a)
{
	for(int i=0;i<rows;++i)
	{
		for(int j=0;j<cols;++j)
		{
			m[i][j] = a*m[i][j];
		}
	}
	return *this;
}

__host__ __device__ dmat operator*(double a, const dmat &B)
{
	dmat C = B;
	return C *= a;
}

__host__ __device__ dvec& dvec::operator+=(const dvec &a)
{
	if(size == a.size)
	{
		for(int i=0;i<size;++i)
		{
			v[i] += a(i);
		}
	}
	return *this;
}

__host__ __device__ dvec operator+(const dvec &a, const dvec &b)
{
	dvec c;
	if(a.size == b.size)
	{
		c.size = a.size;
		for(int i=0;i<a.size;++i)
		{
			c(i) = a(i)+b(i);
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dvec +: Incompatible size!");
	#endif
	return c;
}

__host__ __device__ dmat operator+(const dmat &A, const dmat &B)
{
	dmat C;
	if(A.rows == B.rows && A.cols == B.cols)
	{
		C.rows = A.rows;
		C.cols = A.cols;
		for(int i=0;i<C.rows;++i)
		{
			for(int j=0;j<C.cols;++j)
			{
				C(i,j) = A(i,j)+B(i,j);
			}
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dmat +: Incompatible size!");
	#endif
	return C;
}

__host__ __device__ dvec operator-(const dvec &a, const dvec &b)
{
	dvec c;
	if(a.size == b.size)
	{
		c.size = a.size;
		c = a+(-1.0)*b;
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dvec -: Incompatible size!");
	#endif
	return c;
}

__host__ __device__ dmat operator-(const dmat &A, const dmat &B)
{
	dmat C;
	if(A.rows == B.rows && A.cols == B.cols)
	{
		C.rows = A.rows;
		C.cols = A.cols;
		C = A+(-1.0)*B;
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dmat -: Incompatible size!");
	#endif
	return C;
}

__host__ __device__ dmat operator*(const dmat &A, const dmat &B)
{
	dmat C;
	if(A.cols == B.rows)
	{
		C.rows = A.rows;
		C.cols = B.cols;
		for(int i=0;i<C.rows;++i)
		{
			for(int j=0;j<C.cols;++j)
			{
				for(int k=0;k<A.cols;++k)
				{
					C(i,j) += A(i,k)*B(k,j);
				}
			}
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dmat *: Incompatible size!");
	#endif
	return C;
}

__host__ __device__ dvec operator*(const dmat &A, const dvec &v)
{
	dvec c;
	if(A.cols == v.size)
	{
		c.size = A.rows;
		for(int i=0;i<c.size;++i)
		{
			for(int j=0;j<A.cols;++j)
			{
				c(i) += A(i,j)*v(j);
			}
		} 
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dvec *: Incompatible size!");
	#endif
	return c;
}

__host__ __device__ double ddot(const dvec &a, const dvec &b)
{
	double c = 0;
	if(a.size == b.size)
	{
		for(int i=0;i<a.size;++i)
		{
			c += a(i)*b(i);
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("ddot: Incompatible size!");
	#endif
	return c;
}

__host__ __device__ dvec dhat(const dvec &a, const dvec &b)
{
	dvec c;
	if(a.size == 3 && b.size == 3)
	{
		c = {a(1)*b(2)-a(2)*b(1),a(2)*b(0)-a(0)*b(2),a(0)*b(1)-a(1)*b(0)};
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dhat: Incompatible size!");
	#endif
	return c;
}

__host__ __device__ double ddet(const dmat &A)
{
	double det = 0;
	if(A.rows==1 && A.cols==1)
		det = A(0,0);
	else if(A.rows==2 && A.cols==2)
		det = A(0,0)*A(1,1)-A(0,1)*A(1,0);
	else if(A.rows==3 && A.cols==3)
		det = 	A(0,0)*A(1,1)*A(2,2)+A(0,1)*A(1,2)*A(2,0)+A(0,2)*A(1,0)*A(2,1)
				-A(2,0)*A(1,1)*A(0,2)-A(2,1)*A(1,2)*A(0,0)-A(2,2)*A(1,0)*A(0,1);
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("ddet: Non-squared matrix!");
	#endif	
	return det;
}

__host__ __device__ double dtrace(const dmat &A)
{
	double trace = 0;
	if(A.rows == A.cols)
	{
		for(int i=0;i<A.rows;++i)
		{
			trace += A(i,i);
		}
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("dtrace: Non-squared matrix!");
	#endif	
	return trace;
}

__host__ __device__ double dnorm(const dmat &A)
{
	return std::sqrt(dtrace(A*trans(A)));
}

__host__ __device__ dmat inv(const dmat &A)
{
	dmat I = {{1,0,0},{0,1,0},{0,0,1}};
	dmat B;
	double detA = ddet(A);
	if(_abs(detA)>M_EPS)
	{
		B.rows = A.rows;
		B.cols = A.cols;
		if(A.rows==1 && A.cols==1)
		{
			B(0,0) = 1.0/detA;
		}
		else if(A.rows==2 && A.cols==2)
		{
			I.rows = 2; I.cols = 2;
			B = 1.0/detA*(dtrace(A)*I-A);
		}
		else if(A.rows==3 && A.cols==3)
		{
			I.rows = 3; I.cols = 3;
			B = 1.0/detA*(0.5*(pow(dtrace(A),2.0) - dtrace(A*A))*I-dtrace(A)*A+A*A);
		}
		#ifndef __CUDA_ARCH__
		else throw std::invalid_argument("inv: Non-squared matrix!");
		#endif	
	}
	#ifndef __CUDA_ARCH__
	else throw std::invalid_argument("inv: Singular matrix!");
	#endif
	return B; 
}
__host__ __device__ dmat pinv(const dmat &A)
{
	if(A.rows == A.cols && _abs(ddet(A))>M_EPS)
	{
		return inv(A);	
	}
	else{
		if(A.rows>A.cols && _abs(ddet(trans(A)*A))>M_EPS)
		{
			return inv(trans(A)*A)*trans(A);
		}
		else if(A.rows<A.cols && _abs(ddet(A*trans(A)))>M_EPS)
		{
			return trans(A)*inv(A*trans(A));
		}
		else return {{}};
	}
}

__host__ __device__ dmat trans(const dmat &A)
{
	dmat B;
	B.rows = A.cols;
	B.cols = A.rows;
	for(int i=0;i<B.rows;++i)
	{
		for(int j=0;j<B.cols;++j)
		{
			B(i,j) = A(j,i);
		}
	}
	return B;  
}

__host__ __device__ double dmax(const dtrian &A, int j)
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

__host__ __device__ double dmin(const dtrian &A, int j)
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

__host__  __device__ bool in_box(const dvec &P, const dtrian &T)
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

__host__ __device__ bool in_triangle(const dvec &P, const dtrian &T)
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

__host__ __device__ double distancePointSegment(const dvec &P, const dtrian &T)
{
	double P1P2 = ddot(T(1)-T(0),T(1)-T(0));
	double dd = ddot(P-T(0),T(1)-T(0))/P1P2;
	if(dd<0){return ddot(P-T(0),P-T(0));}
	else if (dd>=0 && dd<=1){return ddot(T(0)-P,T(0)-P)-dd*dd*P1P2;}
	else{return ddot(P-T(1),P-T(1));}
}

__host__ __device__ double t_(const dvec &a)
{
	return atan2(a(1),a(0));
}

__host__ __device__ double a_(const dvec &a)
{
	return atan2(a(2),a(0));
}

__host__ __device__ double b_(const dvec &a)
{
	return atan2(a(1),a(0));
}

__host__ __device__ dmat B_(double t)
{
	dmat A = {	{cos(t),	sin(t),	0},
				{-sin(t),	cos(t),	0},
				{0,			0,		1}};
	return A;
}

__host__ __device__ dmat A_(double a)
{
	double t = -M_PI/2+a;
	dmat A = {	{cos(t),	0,		sin(-t)},
				{0,			1,		0},
				{-sin(t),	0,		cos(t)}};
	return A;
}

__host__ __device__ dmat C_(double b)
{
	double t = b-M_PI/2;
	return B_(t);
}

__host__ __device__ dmat R_(const dvec &x)
{
	return A_(a_(B_(t_(x))*x))*B_(t_(x));
}

__host__ __device__ dmat RR_(const dvec &x, const dvec &y)
{
	return C_(b_(R_(x)*y))*R_(x);
}

__host__ __device__ dvec NN_(const dvec &x)
{
	dvec xx; xx.size = x.size;
	for(int i = 0; i<xx.size; i++)
	{
		if(xx(i)<M_EPS){xx(i)=0;}
		else{xx(i)=x(i);}
	}
	return xx;
}

__host__ __device__ dvec PP_(const dvec &x)
{
	dvec xx = {x(1),x(2)};
	return xx;
}

__host__ __device__ double distancePointTriangle(const dvec &P, const dtrian &T)
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

__host__ __device__ bool accurate(const dvec &P, const dtrian &T)
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

__global__ void parallel_accurate(dvec *P,ditrian *T,dvec *X,bool *N)
{
	int blockId = blockIdx.x;
	int threadId = blockId * blockDim.x + threadIdx.x;

	ditrian t=T[threadIdx.x];
	dvec p=X[blockId];
	
	N[threadId] = accurate(p,ditrian2dtrian(t,P));
}

__host__ bool serial_accurate(dvec *P,ditrian T,dvec X)
{
	ditrian t=T;
	dvec p=X;

	return accurate(p,ditrian2dtrian(t,P));
}

__host__ __device__ dtrian ditrian2dtrian(const ditrian &T, const dvec *P)
{
	dtrian q;
	for(int i=0;i<T.size;i++)
	{
		q(q.size++)= P[T(i)];
	}
	return q;
}

