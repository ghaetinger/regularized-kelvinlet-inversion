#ifndef __MAIN_H_
#define __MAIN_H_

#include "Logger/Logger.hpp"
#include "ShaderCompiler.hpp"
#include "glInitialization.hpp"
#include "textureLoading.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <ctime>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat2x2.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/glm.hpp>
#include <FreeImage.h>

using namespace std;
using namespace cv;
using namespace glm;

void preCalculateScalarsABC(float poisson, float elshear, float * buf);

#endif // __MAIN_H_
