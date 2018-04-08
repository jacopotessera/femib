/*
*	read.cu
*/

#include "read.h"

template <typename T, typename W>
std::vector<T> read(std::string fileName)
{
	std::string tab = "\t";
	std::vector<T> a;
	std::string line;
	std::ifstream file (fileName);

	if(file.is_open())
	{
		for(int i=0;std::getline (file,line);++i)
		{
			T t;
			a.push_back(t);
			size_t pos = 0;
			std::string token;
			for(int j=0;(pos = line.find(tab)) != std::string::npos;++j)
			{
				token = line.substr(0, pos);
				line.erase(0, pos + tab.length());
				//t(t.size++) = castToT<W>(token);
				set<T,W>(a[i],j,token);
			}
		}
		file.close();
	}
	else
	{
		throw EXCEPTION("Unable to open file "+fileName);
	}

	return a;
}

template<class W>
void castToT(std::string s, W& w){}

template<>
double castToT<double>(std::string s)
{
	return std::stod(s);
}

template<>
int castToT<int>(std::string s)
{
	return std::stoi(s);
}

template<class T, class W>
void set(T& a,int i, std::string token)
{
	a.size++; 
	a(i) = castToT<W>(token);
}

template<>
void set<int,int>(int& a,int i, std::string token)
{
	a = castToT<int>(token);
}

template std::vector<dvec> read<dvec,double>(std::string file);
template std::vector<ditrian> read<ditrian,int>(std::string file);
template std::vector<int> read<int,int>(std::string file);

template double castToT<double>(std::string file);
template int castToT<int>(std::string file);

template void set<dvec,double>(dvec& a, int i, std::string token);
template void set<ditrian,int>(ditrian& a, int i, std::string token);
template void set<int,int>(int& a, int i, std::string token);

Mesh readMesh(std::string p,std::string t)
{
	Mesh m;
	m.P = read<dvec,double>(p);
	m.T = read<ditrian,int>(t);
	return m;
}

Mesh readMesh(std::string p,std::string t, std::string e)
{
	Mesh m;
	m.P = read<dvec,double>(p);
	m.T = read<ditrian,int>(t);
	m.E = read<int,int>(e);
	return m;
}

