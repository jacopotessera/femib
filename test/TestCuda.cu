/*
*	TestCuda.cu
*/

#include "../lib/cppunit.h"

#include "../src/Cuda/Cuda.h"

class TestCuda : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestCuda);
	CPPUNIT_TEST(TestCudaConfig);
	CPPUNIT_TEST_SUITE_END();

protected:
	void TestCudaConfig(void);

public:
	void setUp(void);
	void tearDown(void);

private:
	Cuda c;
};

void
TestCuda::TestCudaConfig(void)
{
	CPPUNIT_ASSERT(true);
}

void TestCuda::setUp(void)
{
	std::cout << std::endl;
	c.getSize();
}

void TestCuda::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestCuda );

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
	std::ofstream xmlFileOut("test/results/cppTestCudaResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

