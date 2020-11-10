#include "../include/ShaderCompiler.hpp"

using namespace std;

string ParseShader(const string& filepath)
{
    ifstream stream(filepath);
    if (stream.fail())
    {
        cout << "NONEXISTANT FILE : \'" << filepath << "\'!" << endl;
    }

    stringstream ss;
    string line;
    while (getline(stream, line))
    {
        ss << line << endl;
    }
    return ss.str();
}

unsigned int CompiledShader(unsigned int type, const string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);

    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);

        cout << "FAILED TO COMPILE SHADER : " << message << endl;
    }

    return id;
}

unsigned int CreateShader(const string& vertexShader,
                          const string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = CompiledShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = CompiledShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}
