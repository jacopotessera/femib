/*
*	TestSimplicialMesh.cu
*/

#include "../lib/cppunit.h"

#include "../src/read/read.h"
#include "../src/TriangleMesh/SimplicialMesh.h"
#include "../src/Gauss/GaussService.h"

class TestSimplicialMesh : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestSimplicialMesh);
	CPPUNIT_TEST(test1);	
	CPPUNIT_TEST_SUITE_END();

protected:
	void test1(void);

public:
	void setUp(void);
	void tearDown(void);

private:
	Mesh m0;
	Mesh m;
	GaussService gaussService;
	Gauss g;
	SimplicialMesh<Parallelogram> t;
	dvec P;
	double h;
	std::function<double(dvec)> f;
	std::function<double(dvec)> F;
};

void
TestSimplicialMesh::test1(void)
{
	CPPUNIT_ASSERT(abs(F(m.P[0])-1.0)<M_EPS);
}

void TestSimplicialMesh::setUp(void)
{

	m0.P = {{0.0,0.0},{1.0,0.0},{1.0,1.0},{0.0,1.0}};
	m0.T = {{0,1,3},{2,3,1}};

	m.P = {{2.0,3.0},{5.0,4.0},{3.0,6.0},{0.0,5.0}};
	m.T = {{0,1,2,3}};
	m.E = {};
	P = {2.0,5.0};

	f = [](const dvec &x){
		return -x(0)-x(1)+x(0)*x(1)+1;
	};

	F = [](const dvec &x){
		return (-19.0/16.0)*x(0)+(-15.0/16.0)*x(1)+(1.0/4.0)*x(0)*x(1)+(75.0/16.0);
	};

	std::function<dvec(dvec)> psi = [](const dvec &x){
		double x1 = (1.0/4.0)*(x(0)+x(1)+0*x(0)*x(1)-5.0);
		double x2 = (1.0/8.0)*(-1.0*x(0)+3.0*x(1)-7.0);
		return dvec({x1,x2});
	};

	g = gaussService.getGauss("gauss2_2d");

	t = SimplicialMesh<Parallelogram>(m,g);
	h = t.triangleMesh.getMeshRadius();	
	t.triangleMesh.loadOnGPU();

	/*std::cout << std::endl;
	std::cout << F(P) << std::endl;

	std::cout << psi(m.P[0]) << std::endl;
	std::cout << psi(m.P[1]) << std::endl;
	std::cout << psi(m.P[2]) << std::endl;
	std::cout << psi(m.P[3]) << std::endl;


	std::cout << psi(P) << std::endl;
	std::cout << f(psi(P)) << std::endl;

	std::cout << t.triangleMesh.B[0] << std::endl;
	std::cout << t.triangleMesh.B[1] << std::endl;
	std::cout << t.triangleMesh.b[0] << std::endl;
	std::cout << t.triangleMesh.b[1] << std::endl;

	dvec p0 = t.triangleMesh.Binv[0]*(P-t.triangleMesh.b[0]);
	dvec p1 = t.triangleMesh.Binv[1]*(P-t.triangleMesh.b[1]);
	std::cout << p0 << std::endl;
	std::cout << p1 << std::endl;

	std::cout << affineB(1,m0) << std::endl;
	std::cout << affineb(1,m0)<< std::endl;

	std::cout << affineB(0,m0)*(t.triangleMesh.Binv[0]*(dvec({0.0,5.0})-t.triangleMesh.b[0]))+affineb(0,m0) << std::endl;
	std::cout << affineB(1,m0)*(t.triangleMesh.Binv[1]*(dvec({0.0,5.0})-t.triangleMesh.b[1]))+affineb(1,m0) << std::endl;

	std::cout << affineB(0,m0)*(t.triangleMesh.Binv[0]*(dvec({5.0,4.0})-t.triangleMesh.b[0]))+affineb(0,m0) << std::endl;
	std::cout << affineB(1,m0)*(t.triangleMesh.Binv[1]*(dvec({5.0,4.0})-t.triangleMesh.b[1]))+affineb(1,m0) << std::endl;

	std::cout << affineB(0,m0)*(t.triangleMesh.Binv[0]*(dvec({2.5,4.5})-t.triangleMesh.b[0]))+affineb(0,m0) << std::endl;
	std::cout << affineB(1,m0)*(t.triangleMesh.Binv[1]*(dvec({2.5,4.5})-t.triangleMesh.b[1]))+affineb(1,m0) << std::endl;

	std::cout << affineB(0,m0)*(t.triangleMesh.Binv[0]*(dvec({1,-0.2*1 +5.0})-t.triangleMesh.b[0]))+affineb(0,m0) << std::endl;
	std::cout << affineB(1,m0)*(t.triangleMesh.Binv[1]*(dvec({1,-0.2*1 +5.0})-t.triangleMesh.b[1]))+affineb(1,m0) << std::endl;

	std::cout << affineB(0,m0)*(t.triangleMesh.Binv[0]*(dvec({2,-0.2*2 +5.0})-t.triangleMesh.b[0]))+affineb(0,m0) << std::endl;
	std::cout << affineB(1,m0)*(t.triangleMesh.Binv[1]*(dvec({2,-0.2*2 +5.0})-t.triangleMesh.b[1]))+affineb(1,m0) << std::endl;

	std::cout << affineB(0,m0)*(t.triangleMesh.Binv[0]*(dvec({3,-0.2*3 +5.0})-t.triangleMesh.b[0]))+affineb(0,m0) << std::endl;
	std::cout << affineB(1,m0)*(t.triangleMesh.Binv[1]*(dvec({3,-0.2*3 +5.0})-t.triangleMesh.b[1]))+affineb(1,m0) << std::endl;

	std::cout << f(affineB(1,m0)*p1+affineb(1,m0)) << std::endl;*/

}

void TestSimplicialMesh::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestSimplicialMesh );

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
	std::ofstream xmlFileOut("test/results/cppTestSimplicialMeshResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

