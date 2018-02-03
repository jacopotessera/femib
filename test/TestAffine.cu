/*
*	TestAffine.cu
*/

#include "../lib/cppunit.h"

#include "../src/dmat/dmat_impl.h"
#include "../src/read/read.h"
#include "../src/affine/affine.h"

class TestAffine : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestAffine);
	CPPUNIT_TEST(testAffineB);
	CPPUNIT_TEST(testAffineb);
	CPPUNIT_TEST_SUITE_END();

protected:
	void testAffineB(void);
	void testAffineb(void);

public:
	void setUp(void);
	void tearDown(void);

private:

	Mesh mesh1d1d;
	Mesh mesh1d2d;
	Mesh mesh1d3d;

	dmat identity_1d;
	dvec zero_1d;
	dvec one_1d;
	dmat identity_2d;
	dvec zero_2d;

	dvec punto_medio;
	dvec punto_medio_n;
};

void
TestAffine::testAffineB(void)
{
	CPPUNIT_ASSERT(identity_1d==affineB(2,mesh1d1d));
}

void
TestAffine::testAffineb(void)
{
	CPPUNIT_ASSERT(one_1d==affineb(3,mesh1d1d));
}

void TestAffine::setUp(void)
{
	mesh1d1d = readMesh("mesh/testMesh/p1d1d.mat","mesh/testMesh/t1d1d.mat");
	mesh1d2d = readMesh("mesh/testMesh/p1d2d.mat","mesh/testMesh/t1d2d.mat");
	mesh1d3d = readMesh("mesh/testMesh/p1d3d.mat","mesh/testMesh/t1d3d.mat");

	punto_medio = {0.5};
	punto_medio_n = {-1.0};

	identity_1d = {{1}};
	identity_2d = {{1,0},{0,1}};
	zero_1d = {0};
	one_1d = {1};
	zero_2d = {0,0};
}

void TestAffine::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestAffine );

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
	std::ofstream xmlFileOut("test/results/cppTestAffineResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

