#include "../include/textureLoading.hpp"

using namespace cv;

void setFrameByteArray(VideoCapture video, int size, uint8_t * buf) {
    Mat mat;
    video >> mat;
    cvtColor(mat, mat, COLOR_RGB2BGR);
    memcpy(buf, mat.ptr(0), mat.cols * mat.rows * sizeof(uint8_t) * 3);
}

void initializeTexture(VideoCapture video, int width, int height, int length, bool clamp) {

    Logger::log_debug("Initializing 3D Texture");
    int frameSize = width * height;
    int fullSize = width * height * length;
    uint8_t * vidBytes = (uint8_t *) malloc(fullSize * sizeof(uint8_t) * 3);

    for(int i = 0; i < length; i++){
        uint8_t * buf = (uint8_t *) malloc(frameSize * sizeof(uint8_t) * 3);
        setFrameByteArray(video, frameSize, buf);
        memcpy(vidBytes + (i * frameSize * 3), buf, frameSize * 3);
        free(buf);
    }

    Logger::log_correct("Initialized 3D Texture Byte Array!");

    glEnable(GL_TEXTURE_3D);
    unsigned int texname;
    glGenTextures(1, &texname);
    glBindTexture(GL_TEXTURE_3D, texname);
    if (clamp){
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    } else {
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
    }
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, width, height, length, 0, GL_RGB,
                 GL_UNSIGNED_BYTE, vidBytes);
    free(vidBytes);
}
