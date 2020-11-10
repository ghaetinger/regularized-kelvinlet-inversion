#include "../include/glInitialization.hpp"

using namespace glm;
using namespace std;

void createTextureCanvas() {
    GLfloat vertices[] = {1.0f, 1.0f,
    1.0f, -1.0f,
    -1.0f, 1.0f,
    -1.0f, -1.0f};

    GLfloat texmap[] = {1.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 0.0f,
    0.0f, 1.0f};

    GLuint indices[] = {0, 2, 1, 3, 2, 1};

    GLuint vertexBuffer;
    GLuint indexBuffer;
    GLuint texmapBuffer;

    glGenBuffers(1, &vertexBuffer);
    glGenBuffers(1, &indexBuffer);
    glGenBuffers(1, &texmapBuffer);


    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER,
                 4 * 2 * sizeof(GLfloat),
                 vertices,
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);

    glBindBuffer(GL_ARRAY_BUFFER, texmapBuffer);
    glBufferData(GL_ARRAY_BUFFER,
                 4 * 2 * sizeof(GLfloat),
                 texmap,
                 GL_STATIC_DRAW);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 6 * sizeof(GLuint),
                 indices,
                 GL_STATIC_DRAW);
}

void initializeGLFWWindow(int width, int height, GLFWwindow * glWindow) {
    Logger::log_debug("Initializing Window!");
    glWindow = glfwCreateWindow(width, height, "KELVINLETS VIDEO", NULL, NULL);
    glfwSetWindowSizeLimits(glWindow, width, height, width, height);
    if(!glWindow){
        Logger::log_fatal("Couldn't initialize Window!");
        exit(1);
    }else{
        glfwMakeContextCurrent(glWindow);
        glfwSwapInterval(0);
        Logger::log_correct("Initialized Window!");
    }
}
void initializeGLFW() {
    Logger::log_debug("Initializing GLFW!");
    if(!glfwInit()){
        Logger::log_fatal("Couldn't initialize GLFW!");
        exit(1);
    }else
        Logger::log_correct("Initialized GLFW");
}
void initializeGLEW() {
    Logger::log_debug("Initializing GLEW!");
    if(glewInit() != GLEW_OK){
        Logger::log_fatal("Couldn't initialize GLEW!");
        exit(1);
    }else
        Logger::log_correct("Initialized GLEW");
}

int initializeShaderProgram() {
    Logger::log_debug("Initializing Shaders!");
    const char * vertexShaderPath = "shaders/vertexShader.glsl";
    const char * fragmentShaderPath = "shaders/fragmentShader.glsl";

    string vertexShader = ParseShader(vertexShaderPath);
    string fragmentShader = ParseShader(fragmentShaderPath);

    int shader = CreateShader(vertexShader, fragmentShader);
    glUseProgram(shader);

    Logger::log_correct("Initialized Shaders!");
    return shader;
}
void setShaderParams(int shader, float timestep,
                     float a, float b, float c,
                     vec3 x, vec3 force, float radius,
                     vec3 textProportions, int shouldFix) {
    GLint timestep_loc = glGetUniformLocation(shader, "timestep");
    GLint a_loc = glGetUniformLocation(shader, "a");
    GLint b_loc = glGetUniformLocation(shader, "b");
    GLint c_loc = glGetUniformLocation(shader, "c");
    GLint x_loc = glGetUniformLocation(shader, "pos");
    GLint force_loc = glGetUniformLocation(shader, "force");
    GLint radius_loc = glGetUniformLocation(shader, "radius");
    GLint textprop_loc = glGetUniformLocation(shader, "textprop");
    GLint fix_loc = glGetUniformLocation(shader, "fixBorder");

    glUniform1f(timestep_loc, timestep);
    glUniform1f(a_loc, a);
    glUniform1f(b_loc, b);
    glUniform1f(c_loc, c);
    glUniform3fv(x_loc, 1, &x[0]);
    glUniform3fv(force_loc, 1, &force[0]);
    glUniform1f(radius_loc, radius);
    glUniform3fv(textprop_loc, 1, &textProportions[0]);
    glUniform1i(fix_loc, shouldFix);
}
