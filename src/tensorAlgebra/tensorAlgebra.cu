/*
*	tensorAlgebra.cu
*/

#include "tensorAlgebra.h"

dmat vec2mat(const dvec &b)
{
	dmat B; B.rows = b.size; B.cols = 1;
	for(int i=0;i<b.size;++i)
	{
		B(i,0) = b(i);
	}
	return B;
}


// divergenza
double div(const dmat &A)
{
	double d = 0;
	if(A.rows == A.cols)
	{
		for(int i=0;i<A.rows;++i)
		{
			d += A(i,i);
		}
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("div: Non-squared matrix!");
	#endif
	return d;
}

// doppio prodotto interno
double dpi(const dmat &A, const dmat &B)
{
	if(A.rows == 1 && A.cols == 1 && B.rows == 1 && B.cols == 1)
	{
		return A(0,0)*B(0,0);
	}
	else if(A.rows == 2 && A.cols == 2 && B.rows == 2 && B.cols == 2)
	{
		return A(0,0)*B(0,0)+A(0,1)*B(1,0)+A(1,0)*B(0,1)+A(1,1)*B(1,1);
	}
	else if(A.rows == 3 && A.cols == 3 && B.rows == 3 && B.cols == 3)
	{
		return A(0,0)*B(0,0)+A(0,1)*B(1,0)+A(0,2)*B(2,0)+
				A(1,0)*B(0,1)+A(1,1)*B(1,1)+A(1,2)*B(2,1)+
				A(2,0)*B(0,2)+A(2,1)*B(1,2)+A(2,2)*B(2,2);
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("dpi: Non-squared matrix!");
	#endif
}

double pf(const dmat &A, const dvec &B)
{
	if(A.rows == B.size && A.cols == 1)
	{
		return (trans(A)*vec2mat(B))(0,0);
		/*double r = 0.0;
		for(int i=0;i<A.rows;++i)
		{
			for(int j=0;j<A.cols;++j)
			{
				r += A(i,j)*B(i,j);
			}
		}*/
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("pf: Incompatible size!");
	#endif*/
}

// prodotto scalare
double pf(const dmat &A, const dmat &B)
{
	if(A.rows == B.rows && A.cols == B.cols)
	{
		return dtrace(A*trans(B));
		/*double r = 0.0;
		for(int i=0;i<A.rows;++i)
		{
			for(int j=0;j<A.cols;++j)
			{
				r += A(i,j)*B(i,j);
			}
		}*/
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("pf: Incompatible size!");
	#endif*/
		
	/*if(A.rows == 1 && A.cols == 1 && B.rows == 1 && B.cols == 1)
	{
		return A(0,0)*B(0,0);
	}
  	else if(A.rows == 2 && A.cols == 2 && B.rows == 2 && B.cols == 2)
	{
		return A(0,0)*B(0,0)+A(0,1)*B(0,1)+A(1,0)*B(1,0)+A(1,1)*B(1,1);
	}
	else if(A.rows == 3 && A.cols == 3 && B.rows == 3 && B.cols == 3)
	{
		return A(0,0)*B(0,0)+A(0,1)*B(0,1)+A(0,2)*B(0,2)+
				A(1,0)*B(1,0)+A(1,1)*B(1,1)+A(1,2)*B(1,2)+
				A(2,0)*B(2,0)+A(2,1)*B(2,1)+A(2,2)*B(2,2);
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("pf: Non-squared matrix!");
	#endif*/
}

// gradiente simmetrico
dmat symm(const dmat &A){
  return A+trans(A);
}

// dot-divergenza
dvec dotdiv(const dvec &b, const dmat &B)
{
	dvec v;
	if(b.size == 1 && B.rows == 1 && B.cols == 1)
	{
		v = {b(0)*B(0,0)};
	}
	else if(b.size == 2 && B.rows == 2 && B.cols == 2)
	{
		v = {b(0)*B(0,0)+b(1)*B(0,1),b(0)*B(1,0)+b(1)*B(1,1)};
	}
	else if(b.size == 3 && B.rows == 3 && B.cols == 3)
	{
		v = {b(0)*B(0,0)+b(1)*B(0,1)+b(2)*B(0,2),
		b(0)*B(1,0)+b(1)*B(1,1)+b(2)*B(1,2),
		b(0)*B(2,0)+b(1)*B(2,1)+b(2)*B(2,2)};
	}
	#ifndef __CUDA_ARCH__
	else throw EXCEPTION("dotdiv: Incompatible dimensions!");
	#endif
	return v;
}

std::function<double(dvec)> div(const std::function<dmat(dvec)> &A)
{
	return ([&](const dvec &x){return div(A(x));});
}

std::function<double(dvec)> dpi(const std::function<dmat(dvec)> &A, const std::function<dmat(dvec)> &B)
{
	return ([&](const dvec &x){return dpi(A(x),B(x));});
}

std::function<double(dvec)> pf(const std::function<dmat(dvec)> &A, const std::function<dvec(dvec)> &B)
{
	return ([&](const dvec &x){return pf(A(x),B(x));});
}

std::function<double(dvec)> pf(const std::function<dmat(dvec)> &A, const std::function<dmat(dvec)> &B)
{
	return ([&](const dvec &x){return pf(A(x),B(x));});
}

std::function<dmat(dvec)> symm(const std::function<dmat(dvec)> &A)
{
	return ([&](const dvec &x){return symm(A(x));});
}

std::function<dvec(dvec)> dotdiv(const std::function<dvec(dvec)> &b, const std::function<dmat(dvec)> &A)
{
	return ([&](const dvec &x){return dotdiv(b(x),A(x));});
}

std::function<double(dvec)> ddot(const std::function<dvec(dvec)> &a, const std::function<dvec(dvec)> &b)
{
	return ([&](const dvec &x){return ddot(a(x),b(x));});
}

std::function<double(dvec)> operator*(double a, const std::function<double(dvec)> &b)
{
	return ([&](const dvec &x){return a*b(x);});
}

std::function<double(dvec)> operator-(const std::function<double(dvec)> &a, const std::function<double(dvec)> &b)
{
	//return ([&](const dvec &x){return a(x)-b(x);});
	return a+(-1.0)*b;
}

std::function<double(dvec)> operator*(const std::function<double(dvec)> &a, const std::function<double(dvec)> &b)
{
	return ([&](const dvec &x){return a(x)*b(x);});
}

std::function<dmat(dvec)> operator*(const std::function<dmat(dvec)> &a, const dmat &b)
{
	//return ([&](const dvec &x){return a(x)*b;});
	return a*constant<dmat>(b);
}

std::function<dmat(dvec)> operator*(const std::function<dmat(dvec)> &a, const std::function<dmat(dvec)> &b)
{
	return ([&](const dvec &x){return a(x)*b(x);});
}

std::function<double(dvec)> project(const std::function<dvec(dvec)> &a, int i)
{
	return ([&,i](const dvec &x){return a(x)(i);});
}

std::function<double(dvec)> operator+(const std::function<double(dvec)> &a, const std::function<double(dvec)> &b)
{
	return ([a,b](const dvec &x){return a(x)+b(x);});
}

template<typename T>
std::function<T(dvec)> constant(const T &c)
{
	return ([&](const dvec &x){return c;});
}

template<typename T, typename U, typename V>
std::function<T(V)> compose(const std::function<T(U)> &a, const std::function<U(V)> &b)
{
	return ([&](const V &x){return a(b(x));});
}

F compose(const F &a, const F &b)
{
	F c;
	c.x = compose<dvec,dvec,dvec>(a.x,b.x);
	c.dx = ([&](const dvec &x){return a.dx(b.x(x))*b.dx(x);});
	return c;
}

template std::function<double(dvec)> constant<double>(const double &c);
template std::function<dvec(dvec)> constant<dvec>(const dvec &c);
template std::function<dmat(dvec)> constant<dmat>(const dmat &c);

template std::function<dvec(dvec)> compose<dvec,dvec,dvec>(const std::function<dvec(dvec)> &a, const std::function<dvec(dvec)> &b);

