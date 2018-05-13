/*
*	TestFiniteElement.cu
*/

#include "../lib/cppunit.h"

#include "../src/read/read.h"
#include "../src/Gauss/Gauss.h"
#include "../src/Gauss/GaussService.h"
#include "../src/FiniteElement/FiniteElement.h"
#include "../src/FiniteElement/FiniteElementService.h"
#include "../src/FiniteElementSpace/FiniteElementSpace.h"
#include "../src/FiniteElementSpace/FiniteElementSpaceS.h"
#include "../src/tensorAlgebra/tensorAlgebra.h"

class TestFiniteElement : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestFiniteElement);
	CPPUNIT_TEST(test1);
	CPPUNIT_TEST(testCheck);
	CPPUNIT_TEST_SUITE_END();

protected:
	void test1(void);
	void testCheck(void);

public:
	void setUp(void);
	void tearDown(void);

private:

	GaussService gaussService;
	FiniteElementService finiteElementService;
	std::string p, t, e;
	Gauss g;
	FiniteElement f;
	SimplicialMesh<Parallelogram> triMesh;
	FiniteElementSpace<Parallelogram> finElem;
	FiniteElement checkFiniteElement;
};

void
TestFiniteElement::test1(void)
{
	CPPUNIT_ASSERT(true);
}

void
TestFiniteElement::testCheck(void)
{
	//CPPUNIT_ASSERT(checkFiniteElement.check());
	//CPPUNIT_ASSERT(f.check());
}

void TestFiniteElement::setUp(void)
{
	std::cout << std::endl;
	//logx::Logger::getInstance()->setLogLevel("read.cu",LOG_LEVEL_DEBUG);
	//logx::Logger::getInstance()->setLogLevel("affine.cu",LOG_LEVEL_TRACE);

	p = "mesh/perugia/qp2.mat";
	t = "mesh/perugia/qt2.mat";
	e = "mesh/perugia/qe2.mat";

	g = gaussService.getGauss("gauss5_2d");
	f = finiteElementService.getFiniteElement("Q1_2d1d");

	checkFiniteElement = finiteElementService.getFiniteElement("P2_2d2d");

	triMesh = SimplicialMesh<Parallelogram>(readMesh(p,t,e),g);

	triMesh.triangleMesh.setDim();
	triMesh.triangleMesh.setAffineTransformation();
	triMesh.triangleMesh.loadOnGPU();

	finElem = FiniteElementSpace<Parallelogram>(triMesh,f,g);
	finElem.buildFiniteElementSpace();
	finElem.buildEdge();

	//std::vector<double> q = {1,0,0,0,1,0,1,0,1};
	std::vector<double> q = {0,0,1,0,0,0,0,0,0};
	//sLOG_OK(finElem(q,0).x({0,0}) << finElem(q,0).x({0.5,0.5}) << finElem(q,0).x({1,1}));
	//sLOG_OK(finElem(q,2).x({0,0}) << finElem(q,2).x({0.5,0.5}) << finElem(q,2).x({1,1}));
	//sLOG_OK(finElem(q,0).dx({0,0}) << finElem(q,0).dx({0.5-0.001,0.5-0.001}) << finElem(q,0).dx({1,1}));
	//sLOG_OK(finElem(q,2).dx({0,0}) << finElem(q,2).dx({0.5+0.001,0.5+0.001}) << finElem(q,2).dx({1,1}));

	//sLOG_OK(finElem.T.integrate(project(finElem(q,0).x,0),0));

	/*sLOG_OK(
		finElem(q,0).x({0.5-0.001,0.5-0.001}) <<
		finElem(q,1).x({0.5+0.001,0.5-0.001}) <<
		finElem(q,2).x({0.5+0.001,0.5+0.001}) <<
		finElem(q,3).x({0.5-0.001,0.5+0.001})
	);*/

	finElem.calc(q);
	sLOG_OK(finElem(q,0).x({0.25,1.0/6.0}) );
	sLOG_OK(finElem.getPreCalc(0).x({0.25,1.0/6.0}));
}

void TestFiniteElement::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestFiniteElement );

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
	std::ofstream xmlFileOut("test/results/cppTestFiniteElementResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

