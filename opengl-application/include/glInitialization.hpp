#ifndef __GLINITIALIZATION_H_
#define __GLINITIALIZATION_H_

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include "Logger/Logger.hpp"
#include "ShaderCompiler.hpp"
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <glm/glm.hpp>

using namespace glm;

void initializeGLFWWindow(int width, int height, GLFWwindow * glWindow);
void initializeGLFW();
void initializeGLEW();
int initializeShaderProgram();
void setShaderParams(int shader, float timestep,
                     float a, float b, float c,
                     vec3 x, vec3 force, float radius,
                     vec3 textProportions, int shouldFix);

#endif // __GLINITIALIZATION_H_
