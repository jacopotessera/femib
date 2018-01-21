/*
*	TestGauss.cu
*/

#include "../lib/cppunit.h"

#include "../src/Gauss/Gauss.h"
#include "../src/Gauss/GaussService.h"

class TestGauss : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestGauss);
	CPPUNIT_TEST(testIntegration);
	CPPUNIT_TEST(testIntegration2);
	CPPUNIT_TEST_SUITE_END();

protected:
	void testIntegration(void);
	void testIntegration2(void);

public:
	void setUp(void);
	void tearDown(void);

private:

	GaussService gaussService;
	Gauss g2_1d;
	Gauss g3_1d;
	Gauss g4_1d;
	Gauss g2_2d;
	Gauss g3_2d;
	Gauss g5_2d;
	Gauss g3d;
	std::function<double(dvec)> poli2_1d;
	std::function<double(dvec)> poli3_1d;
	std::function<double(dvec)> poli1_2d;
	std::function<double(dvec)> poli2_2d;
	std::function<double(dvec)> poli4_2d;

};

void
TestGauss::testIntegration(void)
{
	CPPUNIT_ASSERT(abs(g2_1d.integrate(poli2_1d)-5.0/3.0)<0.01);
	CPPUNIT_ASSERT(abs(g3_1d.integrate(poli2_1d)-5.0/3.0)<0.01);
	CPPUNIT_ASSERT(abs(g4_1d.integrate(poli2_1d)-5.0/3.0)<0.01);

	CPPUNIT_ASSERT(abs(g2_2d.integrate(poli2_2d)-29.0/24.0)<0.01);
	CPPUNIT_ASSERT(abs(g5_2d.integrate(poli4_2d)-437.0/360.0)<0.01);
}

void
TestGauss::testIntegration2(void)
{
	/*CPPUNIT_ASSERT_ASSERTION_FAIL(*/CPPUNIT_ASSERT(abs(g2_2d.integrate(poli4_2d)-437.0/360.0)<0.01);
}

void TestGauss::setUp(void)
{
	g2_1d = gaussService.getGauss("gauss2_1d");
	g3_1d = gaussService.getGauss("gauss3_1d");
	g4_1d = gaussService.getGauss("gauss4_1d");
	g2_2d = gaussService.getGauss("gauss2_2d");
	g3_2d = gaussService.getGauss("gauss3_2d");	
	g5_2d = gaussService.getGauss("gauss5_2d");
	//g3d = gaussService.getGauss("not_found");
	poli2_1d = [](dvec x){return 2*x(0)*x(0) + 1;};
	poli3_1d = [](dvec x){return x(0)*x(0)*x(0) +2*x(0) + 2;};
	poli1_2d = [](dvec x){return 2*x(0)+2*x(1) + 1;};
	poli2_2d = [](dvec x){return x(0)*x(1)+2*x(0)+2*x(1) + 1;};
	poli4_2d = [](dvec x){return x(0)*x(0)*x(0)*x(1) + x(0)*x(1)+2*x(0)+2*x(1) + 1;};
}

void TestGauss::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestGauss );

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
	std::ofstream xmlFileOut("test/results/cppTestGaussResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

