/*
*	TestLog.cpp
*/

#include "../../cppunit.h"
#include "../src/Log.h"

using namespace std;

class TestLog : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE(TestLog);
	CPPUNIT_TEST(test);
	CPPUNIT_TEST_SUITE_END();

protected:
	void test(void);

public:
	void setUp(void);
	void tearDown(void);

private:
	logx::Logger* pLog;
	std::string logFile;
};

void
TestLog::test(void)
{
	CPPUNIT_ASSERT(true);
}

void TestLog::setUp(void)
{
	std::cout << std::endl;
	pLog = NULL;
	logFile = "log/logFile.log";
	pLog = logx::Logger::getInstance();
	pLog->updateLogFile(logFile);
	pLog->enable(logx::FILE_LOG);
	LOG_OK("<=============================== START OF PROGRAM ===============================>");
	pLog->setLogLevel("Log.cpp",LOG_LEVEL_WARNING);

	LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");
	LOG_WARNING("Message Logged using Direct Interface, Log level: LOG_WARNING");
	LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
	LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");
	LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");

	pLog->setLogLevel("Log.cpp",LOG_LEVEL_TRACE);
	pLog->setLogLevel("Log.cpp",LOG_LEVEL_DEBUG);
	pLog->setLogLevel("Log.cpp",LOG_LEVEL_INFO);

	pLog->setLogLevel("TestLog.cpp",LOG_LEVEL_TRACE);
	LOG_OK("<========================= UPDATED LOG LEVEL: ENABLE LOG ========================>");
	//pLog->updateLogLevel(ENABLE_LOG);

	LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");
	LOG_WARNING("Message Logged using Direct Interface, Log level: LOG_WARNING");
	LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
	LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");
	LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");

	LOG_OK("<========================= UPDATED LOG LEVEL: DEBUG =============================>");
	//pLog->updateLogLevel(LOG_LEVEL_DEBUG);
	pLog->setLogLevel("TestLog.cpp",LOG_LEVEL_DEBUG);
	LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");
	LOG_WARNING("Message Logged using Direct Interface, Log level: LOG_WARNING");
	LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
	LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");
	LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");

	LOG_OK("<========================= UPDATED LOG LEVEL: DISABLE LOG =======================>");
	pLog->updateLogLevel(DISABLE_LOG);
	pLog->setLogLevel("TestLog.cpp",DISABLE_LOG);

	LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");
	LOG_WARNING("Message Logged using Direct Interface, Log level: LOG_WARNING");
	LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
	LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");
	LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");

	LOG_OK("<================================= LOG DISABLED =================================>");
	pLog->disable(logx::CONSOLE_LOG);

	LOG_ERROR("Message Logged using Direct Interface, Log level: LOG_ERROR");
	LOG_WARNING("Message Logged using Direct Interface, Log level: LOG_WARNING");
	LOG_INFO("Message Logged using Direct Interface, Log level: LOG_INFO");
	LOG_DEBUG("Message Logged using Direct Interface, Log level: LOG_DEBUG");
	LOG_TRACE("Message Logged using Direct Interface, Log level: LOG_TRACE");

	pLog->enable(logx::CONSOLE_LOG);
	for(int i=0;i<1000000;++i)
	{
		//std::ostringstream ss;
		//ss << "Log " << i;
		//LOG_OK(ss);
		//std::cout << "Log " << i << std::endl;
	}	
	LOG_OK("<================================ END OF PROGRAM ================================>");
}

void TestLog::tearDown(void)
{
	//delete ;
}

CPPUNIT_TEST_SUITE_REGISTRATION( TestLog );

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
	std::ofstream xmlFileOut("test/results/cppTestLogResults.xml");
	XmlOutputter xmlOut(&collectedresults, xmlFileOut);
	xmlOut.write();

	// return 0 if tests were successful
	return collectedresults.wasSuccessful() ? 0 : 1;
}

