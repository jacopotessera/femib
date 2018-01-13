/*
*	TestTensorAlgebra.cu
*/

#include "../lib/cppunit.h"

#include "../src/read/read.h"
#include "../src/Gauss/Gauss.h"
#include "../src/Gauss/GaussService.h"
#include "../src/TriangleMesh/TriangleMesh.h"
#include "../src/FiniteElementSpace/FiniteElementSpace.h"
#include "../src/tensorAlgebra/tensorAlgebra.h"

class TestFiniteElement : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestFiniteElement);
	CPPUNIT_TEST(test1);	
	CPPUNIT_TEST_SUITE_END();

protected:
	void test1(void);

public:
	void setUp(void);
	void tearDown(void);

private:

	std::string p, t, e;
	GaussService gaussService;
	Gauss g;
	TriangleMesh triMesh;
	FiniteElementService finiteElementService;
	FiniteElementSpace finElem;
};

void
TestFiniteElement::test1(void)
{
	CPPUNIT_ASSERT(true);
}

void TestFiniteElement::setUp(void)
{
	p = "mesh/perugiamesh/p0.mat";
	t = "mesh/perugiamesh/t0.mat";
	e = "mesh/perugiamesh/e0.mat";
	g = gaussService.getGauss("gauss2_2d");
	TriangleMesh triMesh = TriangleMesh(readMesh(p,t,e),g);

	triMesh.setDim();
	triMesh.setAffineTransformation();
	triMesh.loadOnGPU();

	FiniteElement f = finiteElementService.getFiniteElement("P1_2d2d");
	FiniteElementSpace finElem(triMesh,f,g);
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

