/*
*	TestMongo.cu
*/

#include "../lib/cppunit.h"

#include <chrono>
#include <ctime>
#include <stdexcept>

#include "../src/dmat/dmat.h"
#include "../src/mongodb/dbconfig.h"

#include "../src/utils/Mesh.h"
#include "../src/mongodb/struct.h"
#include "../src/mongodb/mongodb.h"
#include "../src/read/read.h"

class TestMongo : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestMongo);
	CPPUNIT_TEST(testMongo);	
	CPPUNIT_TEST_SUITE_END();

protected:
	void testMongo(void);

public:
	void setUp(void);
	void tearDown(void);

private:

	std::string id;
	std::string date;
	std::string finiteElementV;
	std::string finiteElementQ;
	std::string finiteElementS;
	std::string finiteElementL;

	miniSim s;
	Parameters p;
	miniFE V;
	miniFE Q;
	miniFE S;
	miniFE L;

	timestep t;
	int time;
	std::vector<double> u;
	std::vector<double> q;
	std::vector<double> x;
	std::vector<double> l;

	dbconfig testdb;
	miniSim ss;
	timestep tt;
	int ttt;

};

void
TestMongo::testMongo(void)
{

	CPPUNIT_ASSERT(ss.V.finiteElement == s.V.finiteElement);
	CPPUNIT_ASSERT(ss.V.mesh.P[0](1) == s.V.mesh.P[0](1));
	CPPUNIT_ASSERT(ss.V.mesh.T[1](2) == s.V.mesh.T[1](2));
	CPPUNIT_ASSERT(ss.V.mesh.E[2] == s.V.mesh.E[2]);

	CPPUNIT_ASSERT(ss.Q.finiteElement == s.Q.finiteElement);
	CPPUNIT_ASSERT(ss.Q.mesh.P[4](1) == s.Q.mesh.P[4](1));
	CPPUNIT_ASSERT(ss.Q.mesh.T[5](2) == s.Q.mesh.T[5](2));
	CPPUNIT_ASSERT(ss.Q.mesh.E[1] == s.Q.mesh.E[1]);

	CPPUNIT_ASSERT(ss.S.finiteElement == s.S.finiteElement);
	CPPUNIT_ASSERT(ss.S.mesh.P[6](1) == s.S.mesh.P[6](1));
	CPPUNIT_ASSERT(ss.S.mesh.T[7](2) == s.S.mesh.T[7](2));
	CPPUNIT_ASSERT(ss.S.mesh.E[8] == s.S.mesh.E[8]);

	CPPUNIT_ASSERT(ss.L.finiteElement == s.L.finiteElement);
	CPPUNIT_ASSERT(ss.L.mesh.P[9](1) == s.L.mesh.P[9](1));
	CPPUNIT_ASSERT(ss.L.mesh.T[10](2) == s.L.mesh.T[10](2));
	CPPUNIT_ASSERT(ss.L.mesh.E[11] == s.L.mesh.E[11]);

	CPPUNIT_ASSERT(ss.parameters.TMAX == s.parameters.TMAX);
	CPPUNIT_ASSERT(tt.u[4] == t.u[4]);
	CPPUNIT_ASSERT(tt.q[6] == t.q[6]);
	CPPUNIT_ASSERT(tt.x[8] == t.x[8]);
	CPPUNIT_ASSERT(tt.l[2] == t.l[2]);

	CPPUNIT_ASSERT(ttt == 1);

}

void TestMongo::setUp(void)
{
	testdb = {"testMongo"};
	id = "qwerty";
	std::cout << std::endl;
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
	date = std::ctime(&now_time);

	finiteElementV = "P1";
	finiteElementQ = "P2";
	finiteElementS = "P3";
	finiteElementL = "P4";

	p = {0.1,0.2,0.3,0.4,0.5,6,0.7};
	s.V.finiteElement = finiteElementV;
	s.Q.finiteElement = finiteElementQ;
	s.S.finiteElement = finiteElementS;
	s.L.finiteElement = finiteElementL;

	s.V.mesh.P = read<dvec,double>("mesh/perugia/p0.mat");
	s.V.mesh.T = read<ditrian,int>("mesh/perugia/t0.mat");
	s.V.mesh.E = read<int,int>("mesh/perugia/e0.mat");

	s.Q.mesh.P = read<dvec,double>("mesh/perugia/p1.mat");
	s.Q.mesh.T = read<ditrian,int>("mesh/perugia/t1.mat");
	s.Q.mesh.E = read<int,int>("mesh/perugia/e1.mat");

	s.S.mesh.P = read<dvec,double>("mesh/perugia/p2.mat");
	s.S.mesh.T = read<ditrian,int>("mesh/perugia/t2.mat");
	s.S.mesh.E = read<int,int>("mesh/perugia/e2.mat");
	
	s.L.mesh.P = read<dvec,double>("mesh/perugia/p3.mat");
	s.L.mesh.T = read<ditrian,int>("mesh/perugia/t3.mat");
	s.L.mesh.E = read<int,int>("mesh/perugia/e3.mat");

	s.id = id;
	s.date = date;
	s.parameters = p;

	time = 1;
	u = {0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0};
	q = {0.1,1.1,2.1,3.1,4.1,5.1,6.1,7.1,8.1,9.1};
	x = {0.2,1.2,2.2,3.2,4.2,5.2,6.2,7.2,8.2,9.2};
	l = {0.3,1.3,2.3,3.3,4.3,5.3,6.3,7.3,8.3,9.3};
	t = {id,"",time,u,q,x,l};

	drop(testdb);

	save_sim(testdb,s);
	save_timestep(testdb,t);
	ss = get_sim(testdb,id);
	tt = get_timestep(testdb,id,time);
	ttt = get_time(testdb,id);

	drop(testdb);
}

void TestMongo::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestMongo );

int main(int argc, char* argv[])
{
	// informs test-listener about testresults
	CPPUNIT_NS::TestResult testresult;

	// register listener for collecting the test-results
	CPPUNIT_NS::TestResultCollector collectedresults;
	testresult.addListener (&collectedresults);

	// register listener for per-test progress output
	CPPUNIT_NS::BriefTestProgressListener progress;
	testresult.addListener (&progress);

	// insert test-suite at test-runner by registry
	CPPUNIT_NS::TestRunner testrunner;
	testrunner.addTest (CPPUNIT_NS::TestFactoryRegistry::getRegistry().makeTest ());
	testrunner.run(testresult);

	// output results in compiler-format
	CPPUNIT_NS::CompilerOutputter compileroutputter(&collectedresults, std::cerr);
	compileroutputter.write ();

	// Output XML for Jenkins CPPunit plugin
	std::ofstream xmlFileOut("test/results/cppTestMongoResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

