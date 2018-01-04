/*
*	read.cu
*/

#include "read_old.h"

template <typename T, typename W>
std::vector<T> read_old(std::string fileName)
{
	std::string tab = "\t";
	std::vector<T> a;
	std::string line;
	std::ifstream file (fileName);

	if(file.is_open())
	{
		for(int i=0;std::getline (file,line);i++)
		{
			size_t pos = 0;
			std::string token;
			for(int j=0;(pos = line.find(tab)) != std::string::npos;j++)
			{
				token = line.substr(0, pos);
				line.erase(0, pos + tab.length());
				if(i==0)
				{
					T t;
					a.push_back(t);
				}
				set<T,W>(a[j],i,token);
			}

		}
		file.close();
	}
	else
	{
		throw std::invalid_argument("Unable to open file "+fileName); 
	}

	return a;
}

template<class W>
void castToT_old(std::string s, W& w){}

template<>
double castToT_old<double>(std::string s)
{
	return std::stod(s);
}

template<>
int castToT_old<int>(std::string s)
{
	return std::stoi(s);
}

template<class T, class W>
void set_old(T& a,int i, std::string token)
{
	a.size++; 
	a(i) = castToT_old<W>(token);
}

template<>
void set_old<int,int>(int& a,int i, std::string token)
{
	a = castToT_old<int>(token);
}

template std::vector<dvec> read_old<dvec,double>(std::string file);
template std::vector<ditrian> read_old<ditrian,int>(std::string file);
template std::vector<int> read_old<int,int>(std::string file);

template double castToT_old<double>(std::string file);
template int castToT_old<int>(std::string file);

template void set_old<dvec,double>(dvec& a, int i, std::string token);
template void set_old<ditrian,int>(ditrian& a, int i, std::string token);
template void set_old<int,int>(int& a, int i, std::string token);

Mesh readMesh_old(std::string p,std::string t)
{
	Mesh m;
	m.P = read_old<dvec,double>(p);
	m.T = read_old<ditrian,int>(t);
	return m;
}

Mesh readMesh_old(std::string p,std::string t, std::string e)
{
	Mesh m;
	m.P = read_old<dvec,double>(p);
	m.T = read_old<ditrian,int>(t);
	m.E = read_old<int,int>(e);
	return m;
}

