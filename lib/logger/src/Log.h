/*
*	Logger.h
*/

#ifndef _LOGGER_H_
#define _LOGGER_H_

#include <ctime>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

//LOG_LEVEL
#define LOG_LEVEL_TABLE \
X(OK, "     \033[1;32mOK\033[0m", 0) \
X(ERROR, "  \033[1;31mERROR\033[0m", 1) \
X(WARNING, "\033[1;33mWARNING\033[0m", 2) \
X(INFO, "   \033[1;34mINFO\033[0m", 3) \
X(DEBUG, "  \033[1;35mDEBUG\033[0m", 4) \
X(TRACE, "  \033[1;35mTRACE\033[0m", 5) \

//USER_LOG_LEVEL
#define DISABLE_LOG			1
#define LOG_LEVEL_WARNING	2
#define LOG_LEVEL_INFO		3
#define LOG_LEVEL_DEBUG		4
#define LOG_LEVEL_TRACE		5
#define ENABLE_LOG			6

// Direct Interface
#define LOG_OK(x)		logx::Logger::getInstance()->log(__FILE__,__LINE__,x,logx::OK)
#define LOG_ERROR(x)	logx::Logger::getInstance()->log(__FILE__,__LINE__,x,logx::ERROR)
#define LOG_WARNING(x)	logx::Logger::getInstance()->log(__FILE__,__LINE__,x,logx::WARNING)
#define LOG_INFO(x)		logx::Logger::getInstance()->log(__FILE__,__LINE__,x,logx::INFO)
#define LOG_DEBUG(x)	logx::Logger::getInstance()->log(__FILE__,__LINE__,x,logx::DEBUG)
#define LOG_TRACE(x)	logx::Logger::getInstance()->log(__FILE__,__LINE__,x,logx::TRACE)

namespace logx
{
	#define X(a, b, c) a,
	enum LOG_LEVEL {
	  LOG_LEVEL_TABLE
	};
	#undef X

	enum LOG_TYPE 
	{
		CONSOLE_LOG = 1, FILE_LOG = 2,
	};

	class Logger
	{
		public:
			static Logger* getInstance() throw();

			#define X(a, b, c) #a,
			const std::vector<std::string> logLevelName = {
				LOG_LEVEL_TABLE
			};
			#undef X

			#define X(a, b, c) b,
			const std::vector<std::string> logLevelNameFormatted = {
			  LOG_LEVEL_TABLE
			};
			#undef X

			void log(const char* file, int line, const char* text, LOG_LEVEL logLevel) throw();
			void log(const char* file, int line, std::string& text, LOG_LEVEL logLevel) throw();
			void log(const char* file, int line, std::ostringstream& stream, LOG_LEVEL logLevel) throw();

			void setLogLevel(std::string s, int logLevel);
			int getLogLevel(const char* file);
			void updateLogLevel(int logLevel);
			void enableLog(); // Enable all log levels
			void disableLog(); // Disable all log levels, except error and alarm

			void updateLogFile(std::string logFile);
			void enable(LOG_TYPE logType);
			void disable(LOG_TYPE logType);

		protected:
			Logger();
			~Logger();
			void lock();
			void unlock();
			std::string getCurrentTime();

		private:
			void logOnFile(std::string& data);
			void logOnConsole(std::string& data);
			Logger(const Logger& obj) {}
			void operator=(const Logger& obj) {}

		private:
			static Logger* m_Instance;
			std::ofstream m_File;
			std::string m_LogFile;
			std::mutex m_Mutex;			
			int m_LogLevel;
			std::unordered_map<std::string,int> m_LogLevels;
			int m_LogType;
	};
}

#endif

