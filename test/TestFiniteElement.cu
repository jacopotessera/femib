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
	TriangleMesh triMesh;
	FiniteElementSpaceS finElem;
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
	p = "mesh/pSd_unif.mat";
	t = "mesh/tSd_unif.mat";
	e = "mesh/eSd_unif.mat";

	g = gaussService.getGauss("gauss5_1d");
	f = finiteElementService.getFiniteElement("P1_1d2d");

	checkFiniteElement = finiteElementService.getFiniteElement("P2_2d2d");

	triMesh = TriangleMesh(readMesh(p,t,e),g);

	triMesh.setDim();
	triMesh.setAffineTransformation();
	triMesh.loadOnGPU();

	FiniteElementSpaceS finElem(triMesh,f,g);
	finElem.buildEdge();

	std::cout << finElem.E << std::endl;

	evec a(128);
	a(0)=100;
	a(64)=200;

	std::cout << getColumns(finElem.E.sparseView(),finElem.notEdge)*a << std::endl;

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

