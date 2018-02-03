/*
*	TestTriangleMesh.cu
*/

#include "../lib/cppunit.h"

#include "../src/read/read.h"
#include "../src/TriangleMesh/TriangleMesh.h"
#include "../src/Gauss/GaussService.h"

class TestTriangleMesh : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestTriangleMesh);
	CPPUNIT_TEST(test1);	
	CPPUNIT_TEST_SUITE_END();

protected:
	void test1(void);

public:
	void setUp(void);
	void tearDown(void);

private:
	std::string sp;
	std::string st;
	std::string se;
	Mesh m;
	GaussService gaussService;
	Gauss g;
	TriangleMesh t;
	double h;
};

void
TestTriangleMesh::test1(void)
{
	CPPUNIT_ASSERT(h == 4.0);
}

void TestTriangleMesh::setUp(void)
{

	sp = "mesh/perugia/p0.mat";
	st = "mesh/perugia/t0.mat";
	se = "mesh/perugia/e0.mat";
	m = readMesh(sp,st,se);
	g = gaussService.getGauss("gauss2_2d");

	t = TriangleMesh(m,g);
	t.setDim();
	t.setAffineTransformation();
	h = t.getMeshRadius();	
	t.loadOnGPU();
}

void TestTriangleMesh::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestTriangleMesh );

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
	std::ofstream xmlFileOut("test/results/cppTestTriangleMeshResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

