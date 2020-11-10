#include "../../include/Logger/Logger.hpp"

using namespace std;

int classifyEnvironment(){
	char * environment = getenv("KELVIN");
	if(environment == NULL){
		printf("USE \"env KELVIN=\"FATAL|ERROR|...\"\" before calling the program");
		return UNDEF;
	}
	string str_environment = string(environment);
	if(str_environment.compare("FATAL") == 0)
		return FATAL;
	if(str_environment.compare("ERROR") == 0)
		return ERROR;
	if(str_environment.compare("WARNING") == 0)
		return WARNING;
	if(str_environment.compare("DEBUG") == 0)
		return DEBUG;
	if(str_environment.compare("CORRECT") == 0)
		return CORRECT;
	return NOWARN;
}

void Logger::log_fatal(std::string message){
	if(classifyEnvironment() >= FATAL){
		printf(ANSI_COLOR_RED "FATAL: %s" ANSI_COLOR_RESET "\n", message.c_str());
	}	
}

void Logger::log_error(std::string message){
	if(classifyEnvironment() >= ERROR){
		printf(ANSI_COLOR_MAGENTA "ERROR: %s" ANSI_COLOR_RESET "\n", message.c_str());
	}	
}
void Logger::log_warning(std::string message){
	if(classifyEnvironment() >= WARNING){
		printf(ANSI_COLOR_YELLOW "WARNING: %s" ANSI_COLOR_RESET "\n", message.c_str());
	}	
}
void Logger::log_correct(std::string message){
	if(classifyEnvironment() >= CORRECT){
		printf(ANSI_COLOR_GREEN "CORRECT: %s" ANSI_COLOR_RESET "\n", message.c_str());
	}	
}
void Logger::log_debug(std::string message){
	if(classifyEnvironment() >= DEBUG){
		printf(ANSI_COLOR_BLUE "DEBUG: %s" ANSI_COLOR_RESET "\n", message.c_str());
	}	
}
