/*
*	TestRead.cu
*/

#include "../lib/cppunit.h"

#include "../src/read/read.h"

class TestRead : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestRead);
	CPPUNIT_TEST(testRead);
	CPPUNIT_TEST_SUITE_END();

protected:
	void testRead(void);

public:
	void setUp(void);
	void tearDown(void);

private:

	std::vector<dvec> meshP;
	std::vector<ditrian> meshT;
};

void
TestRead::testRead(void)
{
	CPPUNIT_ASSERT(meshP[1](0)==-2);
	CPPUNIT_ASSERT(meshT[1](1)==2);
}

void TestRead::setUp(void)
{
	meshP = read<dvec,double>("mesh/P1d1d.mat");
	meshT = read<ditrian,int>("mesh/T1d1d.mat");
	try
	{
		meshP = read<dvec,double>("mat.mat");
	}
	catch(const std::invalid_argument& e){}
}

void TestRead::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestRead );

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
	std::ofstream xmlFileOut("test/results/cppTestReadResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

