#include "../include/main.hpp"

using namespace glm;
using namespace cv;


char * video_name;
float elshear, poisson, radius, a, b, c;
int shouldFix, save;
vec3 x, force;

void preCalculateScalarsABC(float poisson, float elshear) {
	a = 1 / (4 * pi<float>() * elshear);
	b = a / (4 - 4 * poisson);
	c = 2 / (3 * a - 2 * b);
}

void fillInArgs(int argc, char * argv[]) {
	if (argc < 13) {
		Logger::log_error("Input Format: ./kelvin [video path] [elastic shear mod] [possion ratio] [x] [y] [t] [fx] [fy] [ft] [radius] [should fix borders] [should save]");
		exit(1);
	}

	video_name = argv[1];

	elshear = (float) atof(argv[2]);
	poisson = (float) atof(argv[3]);

	x = vec3((float) atof(argv[4]),
			 (float) atof(argv[5]),
			 (float) atof(argv[6]));

	force = vec3((float) atof(argv[7]),
				 (float) atof(argv[8]),
				 (float) atof(argv[9]));

	radius = (float) atof(argv[10]);
	shouldFix = atoi(argv[11]);

	save = bool(atoi(argv[12]));

	preCalculateScalarsABC(poisson, elshear);
}

int main (int argc, char * argv[]) {

	fillInArgs(argc, argv);
	VideoCapture video(video_name);
	if(!video.isOpened()){
		Logger::log_error("Video wont open!");
		exit(1);
	}

	int width = video.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = video.get(cv::CAP_PROP_FRAME_HEIGHT);
	int length = video.get(cv::CAP_PROP_FRAME_COUNT);

	vec3 textProportions = vec3(width, height, length);

	initializeGLFW();
	GLFWwindow* glWindow = glfwCreateWindow(width, height, "LearnOpenGL", NULL, NULL);
	if (glWindow == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(glWindow);
	initializeGLEW();
	int shader = initializeShaderProgram();
	initializeTexture(video, width, height, length, bool(shouldFix));

	createTextureCanvas();

	glFlush();
	float start = glfwGetTime();
	int step = 0;
	while (!glfwWindowShouldClose(glWindow) && step <= length) {
		setShaderParams(shader, step,
						a, b, c,
						x, force, radius,
						textProportions, shouldFix);
		glClear(GL_COLOR_BUFFER_BIT);

		glDrawElements(GL_TRIANGLES,
					   6,
					   GL_UNSIGNED_INT, nullptr);

		glfwSwapBuffers(glWindow);
		glfwPollEvents();

		if (save) {
			BYTE* pixels = new BYTE[3 * width * height];
			glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels);
			FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, width, height, 3 * width, 24, 0x0000FF, 0xFF0000, 0x00FF00, false);
			char filename[] = "./results/result_warp_%d.bmp";
			char final_filename[1000];
			sprintf(final_filename, filename, step);
			FreeImage_Save(FIF_BMP, image, final_filename, 0);
			FreeImage_Unload(image);
			delete [] pixels;
		}

		step++;
	}
	glFlush();
	float end = glfwGetTime();
	cout << float(length/(end - start)) << " FPS" << endl;

	return 0;
}
