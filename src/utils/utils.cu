/*
*	utils.cu
*/

#include "utils.h"

std::vector<int> setdiff(std::vector<int> x, const std::vector<int> &y)
{
	for(int i=0;i<y.size();++i)
	{
		auto j = std::find(x.begin(),x.end(),y[i]);
		if (j !=  x.end())
		{
			x.erase(j);
		}
	}
	return x;
}

std::vector<int> operator+(std::vector<int> x, int y)
{
	for(int i=0;i<x.size();++i)
	{
		x[i]+=y;
	}
	return x;
}

std::vector<int> join(std::vector<int> x, const std::vector<int> &y)
{
	for(int j=0;j<y.size();++j)
	{
		x.push_back(y[j]);
	}
	return x;
}

std::vector<int> linspace(int a)
{
	std::vector<int> x;
	for(int i=0;i<a;++i)
	{
		x.push_back(i);
	}
	return x;
}

int find(const std::vector<int> &x, int i)
{
	int k=-1;
	for(int j=0;j<x.size();++j)
	{
		if(x[j]==i)
		{
			k=j;
			break;
		}
	}
	return k;
}

int find(const std::vector<dvec> &x, const dvec &v)
{
	int k=-1;
	for(int j=0;j<x.size();++j)
	{
		if(x[j]==v)
		{
			k=j;
			break;
		}
	}
	return k;
}

std::vector<double> operator+(const std::vector<double> &A, const std::vector<double> &B)
{
	std::vector<double> C;
	for(int i=0;i<B.size();++i)
	{
		C.push_back(A[i]+B[i]);
	}
	return C;
}

std::vector<double> operator*(double a, const std::vector<double> &B)
{
	std::vector<double> C;
	for(int i=0;i<B.size();++i)
	{
		C.push_back(a*B[i]);
	}
	return C;
}


std::vector<double> eigen2vector(const Eigen::Matrix<double,Eigen::Dynamic,1> &v)
{
	std::vector<double> w;
	for(int i=0;i<v.size();++i)
	{
		w.push_back(v(i));
	}
	return w;
}

Eigen::Matrix<double,Eigen::Dynamic,1> vector2eigen(const std::vector<double> &v)
{
	Eigen::Matrix<double,Eigen::Dynamic,1> w(v.size());
	for(int i=0;i<v.size();++i)
	{
		w(i)=v[i];
	}
	return w;
}

std::vector<double> dvec2vector(const dvec &t)
{
	std::vector<double> v;
	for(int i=0;i<t.size;++i)
	{
		v.push_back(t(i));
	}
	return v;
}

std::vector<int> ditrian2vector(const ditrian &t)
{
	std::vector<int> v;
	for(int i=0;i<t.size;++i)
	{
		v.push_back(t(i));
	}
	return v;
}

Eigen::SparseMatrix<double> getColumns(Eigen::SparseMatrix<double> S, const std::vector<int> &x)
{
	std::vector<Eigen::Triplet<double>> c;
	Eigen::SparseMatrix<double> C = Eigen::SparseMatrix<double>(S.cols(),x.size());
	for(int i=0;i<x.size();++i)
	{
		c.push_back(Eigen::Triplet<double>({x[i],i,1}));
	}
	C.setFromTriplets(c.begin(),c.end());
	return S*C;
}

Eigen::SparseMatrix<double> getRows(Eigen::SparseMatrix<double> S, const std::vector<int> &x)
{
	std::vector<Eigen::Triplet<double>> c;
	Eigen::SparseMatrix<double> C = Eigen::SparseMatrix<double>(S.rows(),x.size());
	for(int i=0;i<x.size();++i)
	{
		c.push_back(Eigen::Triplet<double>({x[i],i,1}));
	}
	C.setFromTriplets(c.begin(),c.end());
	return C.transpose()*S;
}

etmat esmat2etmat(const esmat& A)
{
	return esmat2etmat(A,0,0);
}

etmat esmat2etmat(const esmat& A, int rowDrift, int colDrift)
{
	etmat B;
	for(int n=0;n<A.outerSize();++n)
	{
		for(Eigen::SparseMatrix<double>::InnerIterator it(A,n);it;++it)
		{
			B.push_back(Eigen::Triplet<double>({it.row()+rowDrift,it.col()+colDrift,it.value()}));
		}
	}
	return B;
}

esmat etmat2esmat(const etmat &A, int rows, int cols)
{
	esmat B(rows,cols);
	B.setFromTriplets(A.begin(),A.end());
	return B;
}

etmat transpose(const etmat &A)
{
	etmat B;
	for(auto a : A)
		B.push_back(Eigen::Triplet<double>({a.col(),a.row(),a.value()}));
	return B;
}

etmat& operator+=(etmat& lhs, const etmat &rhs)
{
	lhs.insert(lhs.end(),rhs.begin(),rhs.end());
	return lhs;
	//(*this).insert((*this).end(),rhs.begin(),rhs.end());
	//return *this;
}


/*etmat operator+(etmat lhs, const etmat &rhs)
{
	lhs.insert(lhs.end(),rhs.begin(),rhs.end());
	return lhs;
	//(*this).insert((*this).end(),rhs.begin(),rhs.end());
	//return *this;
}*/

std::ostream& operator<<(std::ostream& out, const etmat &T)
{
	for(auto t : T)
	{
		out << t.row()<< "x" << t.col() << ": " << t.value() << std::endl;
	}
	out << std::endl;
}

std::string getTimestamp()
{
	std::time_t result = std::time(0);
	char d[16];
	std::strftime(d,16,"%Y%m%d_%H%M%S",std::localtime(&result));
	std::string s(d);
	return s;
}

