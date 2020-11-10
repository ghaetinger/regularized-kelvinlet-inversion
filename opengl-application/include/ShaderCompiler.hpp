#ifndef SHADER_COMPILER_HEADER
#define SHADER_COMPILER_HEADER

#include <GL/glew.h>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

string ParseShader(const string& filepath);

unsigned int CompiledShader(unsigned int type, const string& source);

unsigned int CreateShader(const string& vertexShader,
                          const string& fragmentShader);

#endif
