#ifndef LOGGER_HEADER
#define LOGGER_HEADER

#include <string>
#include <cstdlib>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define NOWARN -1
#define FATAL 0
#define ERROR 1
#define WARNING 2
#define CORRECT 3
#define DEBUG 4
#define UNDEF 5

namespace Logger{


	void log_fatal(std::string message);
	void log_error(std::string message);
	void log_warning(std::string message);
	void log_debug(std::string message);
	void log_correct(std::string message);
}

#endif
