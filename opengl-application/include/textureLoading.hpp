#ifndef __TEXTURELOADING_H_
#define __TEXTURELOADING_H_

#include "Logger/Logger.hpp"
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

using namespace cv;

void createTextureCanvas();
void setFrameByteArray(VideoCapture video, int size, uint8_t * buf);
void initializeTexture(VideoCapture video, int width, int height, int length, bool clamp);

#endif // __TEXTURELOADING_H_
