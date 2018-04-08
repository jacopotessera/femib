/*
*	Log.cpp
*/

#include "Log.h"
#include "../../stacktrace/stacktrace.h"

using namespace logx;

Logger* Logger::m_Instance = 0;

Logger::Logger()
{
	m_LogFile = "log/default.log";
	m_File.open(m_LogFile.c_str(), std::ios::out|std::ios::app);
	m_LogLevel = DISABLE_LOG;
	m_LogPID = false;
	m_LogType = CONSOLE_LOG;
	m_LogLevels["Log.cpp"] = LOG_LEVEL_INFO;
}

Logger::~Logger()
{
	m_File.close();
}

static void handler(int sig)
{
	std::ostringstream ss;
	ss << "Caught signal " << sig;
	LOG_ERROR(ss);
	exit(1);
}

Logger* Logger::getInstance() throw()
{
	if(m_Instance == 0) 
	{
		m_Instance = new Logger();

		signal(SIGSEGV, handler);
		signal(SIGINT, handler);
		signal(SIGQUIT, handler);
		signal(SIGKILL, handler);

		LOG_INFO("Logger inizialized.");
		std::ostringstream ss;
		ss << "Global LogLevel set to " << m_Instance->logLevelName[m_Instance->m_LogLevel] << ".";
		LOG_INFO(ss);

		if(m_Instance->m_LogPID)
			LOG_INFO("PID logging is enabled.");
		else
			LOG_INFO("PID logging not is enabled.");

		if(m_Instance->m_LogType & CONSOLE_LOG)
		{
			std::ostringstream tt;
			tt << "CONSOLE_LOG is enabled.";
			LOG_INFO(tt);
		}
		else
		{
			std::ostringstream tt;
			tt << "CONSOLE_LOG is not enabled.";
			LOG_INFO(tt);
		}
		if(m_Instance->m_LogType & FILE_LOG)
		{
			std::ostringstream tt;
			tt << "FILE_LOG is enabled.";
			LOG_INFO(tt);
		}
		else
		{
			std::ostringstream tt;
			tt << "FILE_LOG is not enabled.";
			LOG_INFO(tt);
		}
		std::ostringstream tt;
		tt << "LogFile set to " << m_Instance->m_LogFile << ".";
		LOG_INFO(tt);
		LOG_INFO("Disabling logs for Logger.");
		m_Instance->setLogLevel("Log.cpp",DISABLE_LOG);
	}
	return m_Instance;
}

void Logger::lock()
{
	m_Mutex.lock();
}

void Logger::unlock()
{
	m_Mutex.unlock();
}

void Logger::logOnFile(std::string& data)
{
	lock();
	m_File << getCurrentTime() << "" << data << std::endl;
	unlock();
}

void Logger::logOnConsole(std::string& data)
{
	lock();
	std::cout << getCurrentTime() << "" << data << std::endl;
	unlock();
}

std::string Logger::getCurrentTime()
{
	char currTime[22];
	std::time_t now = std::time(0);
	std::strftime(currTime,22,"[%Y-%m-%d %H:%M:%S]",std::localtime(&now));
	std::string currentTime(currTime);
	return currentTime;
}

void Logger::log(const char* file, int line, const char* function, const char* text, LOG_LEVEL logLevel) throw()
{
	std::string data;
	data.append("["+logLevelNameFormatted[logLevel]+"][");
	if(m_LogPID)
	{
		char cpid[6];
		sprintf(cpid, "%d", ::getpid());
		data.append(cpid);
		data.append("-");
		char cppid[6];
		sprintf(cppid, "%d", ::getppid());
		data.append(cppid);
		data.append(" - ");
	}
	data.append(function);
	data.append("@");
	data.append(file);
	data.append(":");
	char cline[6];
	sprintf(cline, "%d", line);
	data.append(cline);
	data.append("]: ");
	data.append(text);

	//print stacktrace on error
	if(logLevel == 1)
	{
		data.append("\n"+printStacktrace());
	}

	if((m_LogType & FILE_LOG) && (getLogLevel(file) >= logLevel))
	{
		logOnFile(data);
	}
	if((m_LogType & CONSOLE_LOG) && (getLogLevel(file) >= logLevel))
	{
		logOnConsole(data);
	}
}

void Logger::log(const char* file, int line, const char* function, std::string& text, LOG_LEVEL logLevel) throw()
{
	log(file, line, function, text.data(), logLevel);
}

void Logger::log(const char* file, int line, const char* function, std::ostringstream& stream, LOG_LEVEL logLevel) throw()
{
	std::string text = stream.str();
	log(file, line, function, text, logLevel);
}

void Logger::updateLogLevel(int logLevel)
{
	if(logLevel < DISABLE_LOG)
		logLevel = DISABLE_LOG;
	if(logLevel > ENABLE_LOG)
		logLevel = ENABLE_LOG;
	m_LogLevel = logLevel;
}

void Logger::enableLog()
{
	m_LogLevel = ENABLE_LOG; 
}

void Logger:: disableLog()
{
	m_LogLevel = DISABLE_LOG;
}

void Logger::updateLogFile(std::string logFile)
{
	lock();
	m_File.close();
	m_LogFile = logFile;
	m_File.open(m_LogFile.c_str(), std::ios::out|std::ios::app);
	unlock();
}

void Logger::enable(LOG_TYPE logType)
{
	m_LogType = m_LogType | logType ;
}

void Logger::disable(LOG_TYPE logType)
{
	m_LogType = m_LogType & ~logType;
}

void Logger::setLogLevel(std::string s, int logLevel)
{
	m_LogLevels[s] = logLevel;
	std::ostringstream ss; 
	ss << "Updating LogLevel for " << s << " (" << logLevelName[logLevel] << ")";
	LOG_INFO(ss);
}

int Logger::getLogLevel(const char* file)
{

	for(std::unordered_map<std::string, int>::iterator it = m_LogLevels.begin(); it != m_LogLevels.end(); ++it)
	{
		std::string a = it->first;
		std::string sfile(file);
		if(sfile==a || sfile.find("/"+a) != std::string::npos)
			return it->second;
	}
	std::ostringstream ss;
	ss << "LogLevel not set for " << file << ". Using global LogLevel (" << logLevelName[m_LogLevel] << ")";
	LOG_WARNING(ss);
	return m_LogLevel;

	/*std::unordered_map<std::string, int>::iterator i = m_LogLevels.find(file);
	if(i!=m_LogLevels.end())
		return m_LogLevels[file];
	else
	{
		std::ostringstream ss;
		ss << "LogLevel not set for " << file << ". Using global LogLevel (" << logLevelName[m_LogLevel] << ")";
		LOG_WARNING(ss);
		return m_LogLevel;
	}*/
}

