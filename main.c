/*
 OpenGL visualization skeleton for displaying bitmap images. Just provide a GenerateImage function.
 Good starting point for all image processing exercises for parallel programming.

 Example of generating bitmaps using GenerateImage and the prepared GLUT OpenGL visualization.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
  #include <GLUT/glut.h>
#else
  #include <GL/freeglut.h>
  #include <GL/freeglut_ext.h>
#endif

#define TEX_SIZE 2048
#define ALPHA_SIZE 128
#define C_RED 0
#define C_GREEN 1
#define C_BLUE 2

#define DISPLAY_EFFECT 0
#define DISPLAY_LAYERS 1
#define DISPLAY_FRACTAL 2
#define DISPLAY_TO_LESSBIT 3

typedef struct {
    GLubyte r;
    GLubyte g;
    GLubyte b;
} pixel;

typedef struct {
    GLubyte r;
    GLubyte g;
    GLubyte b;
    GLubyte a;
} rgba_pix;

pixel image[2*TEX_SIZE][TEX_SIZE];

pixel (*source)[TEX_SIZE] = (pixel(*)[TEX_SIZE])&image[0][0];

pixel (*result)[TEX_SIZE] = (pixel(*)[TEX_SIZE])&image[TEX_SIZE][0];

rgba_pix layer1[TEX_SIZE][TEX_SIZE];

rgba_pix layer2[TEX_SIZE][TEX_SIZE];

int alpha_x = 0;

int mode = DISPLAY_EFFECT;

int reduced = 0;

GLuint texture;

FILE *open_image_file(char *path) {
	FILE *image_file = fopen(path, "rb");
	if(image_file == NULL) {
		printf("Error openning %s\n", path);
		exit(1);
	}
	return image_file;
}

void load_rgb(pixel target[TEX_SIZE][TEX_SIZE], char *path, int size) {
	int x, y;
	FILE *image_file = open_image_file(path);
	for(x=0; x<size; x++) {
		for(y=0; y<size; y++) {
			fread(&target[x][y], sizeof(pixel), 1, image_file);
		}
	}
	fclose(image_file);
}

void load_rgba(rgba_pix target[TEX_SIZE][TEX_SIZE], char *path, int size) {
	int x, y;
	FILE *image_file = open_image_file(path);
	for(x=0; x<size; x++) {
		for(y=0; y<size; y++) {
			fread(&target[x][y], sizeof(rgba_pix), 1, image_file);
		}
	}
	fclose(image_file);
}

int get_between_0_255(int source) {
	if(source > 255) return 255;
	else if(source < 0) return 0;
	else return source;
}

int truncate(int value, int size, int trunc_size) {
	float scale = (float)size / (float) trunc_size;
	return round(round(value/scale)*scale);
}

void set_color(pixel target[TEX_SIZE][TEX_SIZE], int x, int y, int red, int green, int blue) {
	target[x][y].r = get_between_0_255(red);
	target[x][y].g = get_between_0_255(green);
	target[x][y].b = get_between_0_255(blue);
}

void set_pixel_color(pixel *pix, int red, int green, int blue) {
	pix->r = get_between_0_255(red);	
	pix->g = get_between_0_255(green);
	pix->b = get_between_0_255(blue);
}

void blend_layer_at(
	pixel image[TEX_SIZE][TEX_SIZE], 
	rgba_pix layer[TEX_SIZE][TEX_SIZE], 
	int start_x, int start_y
) {
	int x,y;
	for(x=start_x;x < TEX_SIZE; x++) {
		for(y=start_y; y < TEX_SIZE; y++) {
			int lx = x - start_x;
			int ly = y - start_y;
			float alp = (float)(layer[lx][ly].a) / 255.0;
			set_color(image, x, y,
				layer[lx][ly].r * alp + image[x][y].r*(1-alp),
				layer[lx][ly].g * alp + image[x][y].g*(1-alp),
				layer[lx][ly].b * alp + image[x][y].b*(1-alp)
			);
		}
	}
}

void pixel_add(pixel *pixel, int addition) {
	pixel->r += addition;
	pixel->g += addition;
	pixel->b += addition;
}

void convolution_transform(
	int pixel_x, int pixel_y, 
	float *kernel, 
	int kernel_half_size, 
	int kernel_size,
	float bias
) {
	int x, y, red, green, blue;
	red = green = blue = 0;

	for(x = pixel_x - kernel_half_size; x <= pixel_x + kernel_half_size; x++) {
		if(x >= 0 && x < TEX_SIZE) {
			for(y = pixel_y - kernel_half_size; y <= pixel_y + kernel_half_size; y++) {
				if(y >= 0 && y < TEX_SIZE) {
					int kernel_p = (x - pixel_x + kernel_half_size) * kernel_size;
					kernel_p += (y - pixel_y + kernel_half_size);

					red += round(source[x][y].r*kernel[kernel_p])+bias;
					green += round(source[x][y].g*kernel[kernel_p])+bias;
					blue += round(source[x][y].b*kernel[kernel_p])+bias;

				}
			}
		}
	}

	set_color(result, pixel_x, pixel_y, red, green, blue);
}

void convolution(float *kernel, int kernel_size, float bias) {
	int x, y;
	int kernel_half_size = floor(kernel_size/2);

	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			convolution_transform(x, y, kernel, kernel_half_size, kernel_size, bias);
		}
	}
}

void blur() {
	float kernel[3][3] = {
		{ (1.0/9.0), (1.0/9.0), (1.0/9.0) },
		{ (1.0/9.0), (1.0/9.0), (1.0/9.0) },
		{ (1.0/9.0), (1.0/9.0), (1.0/9.0) }
	};

	convolution((float*)&kernel, 3, 0);
}

void blur5x() {
	float kernel[5][5] = {
		{ (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0) },
		{ (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0) },
		{ (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0) },
		{ (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0) },
		{ (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0), (1.0/25.0) }
	};

	convolution((float*)&kernel, 5, 0);
}

void edge_detection1() {
	float kernel[3][3] = {
		{ 1, 0,-1},
		{ 0, 0, 0},
		{-1, 0, 1}
	};

	convolution((float*) kernel, 3, 0);
}

void edge_detection3() {
	float kernel[3][3] = {
		{-1,-1,-1},
		{-1, 8,-1},
		{-1,-1,-1}
	};

	convolution((float*)kernel, 3, 0);
}

void sharpen() {
	float kernel[3][3] = {
		{ 0,-1, 0},
		{-1, 5,-1},
		{ 0,-1, 0}
	};

	convolution((float*) kernel, 3, 0);
}

void emboss() {
	float kernel[3][3] = {
		{ (-1.0/2.0), (-1.0/2.0),      0.0  },
		{ (-1.0/2.0),         0 , (1.0/2.0) },
		{         0 , ( 1.0/2.0), (1.0/2.0) }
	};

	convolution((float*) kernel, 3, 2);
}

void (*effect)() = sharpen;

void print_fractal() {

	#define MaxIters 400
	#define LEFT     -2.0
	#define RIGHT    1.0
	#define TOP      1.0
	#define BOTTOM   -1.0

	short   x, y, count;
	long double zr, zi, cr, ci;
	long double rsquared, isquared;

	for (y = 0; y < TEX_SIZE; y++) {
		for (x = 0; x < TEX_SIZE; x++) {
			zr = 0.0;
			zi = 0.0;

			cr = LEFT + x * (RIGHT - LEFT) / TEX_SIZE;
			ci = TOP + y * (BOTTOM - TOP) / TEX_SIZE;

			rsquared = zr * zr;
			isquared = zi * zi;

			for (count = 0; rsquared + isquared <= 4.0 && count < MaxIters; count++) {
				zi = zr * zi * 2;
				zi += ci;
				zr = rsquared - isquared;
				zr += cr;
				rsquared = zr * zr;
				isquared = zi * zi;
			}
			if (rsquared + isquared <= 4.0)
				set_color(result,x,y,50,50,50);
			else
				set_color(result,x,y,0,0,0);
		}
	}
}

void to_3bit() {
	int x, y;
	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			result[x][y].r = (source[x][y].r > 125) ? 255 : 0;
			result[x][y].g = (source[x][y].g > 125) ? 255 : 0;
			result[x][y].b = (source[x][y].b > 125) ? 255 : 0;
		}
	}
}

void to_1bit() {
	int x, y;
	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			int power = (source[x][y].r + source[x][y].g + source[x][y].b);
			if(power > 380) {
				set_pixel_color(&result[x][y], 255,255,255);
			} else {
				set_pixel_color(&result[x][y], 0, 0, 0);
			}
		} }
}

void to_grayscale() {
	int x, y;

	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			int power = (0.3 * source[x][y].r + 0.59 * source[x][y].g + 0.11 * source[x][y].b);
			result[x][y].r = result[x][y].g = result[x][y].b = power;
		}
	}
}

void to_grayscale_custom(pixel target[TEX_SIZE][TEX_SIZE]) {
	int x, y;

	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			int power = (0.3 * source[x][y].r + 0.59 * source[x][y].g + 0.11 * source[x][y].b);
			target[x][y].r = target[x][y].g = target[x][y].b = power;
		}
	}
}

void to_1bit_custom(pixel target[TEX_SIZE][TEX_SIZE]) {
	int x, y;
	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			int power = (source[x][y].r + source[x][y].g + source[x][y].b);
			if(power > 380) {
				set_pixel_color(&target[x][y], 255,255,255);
			} else {
				set_pixel_color(&target[x][y], 0, 0, 0);
			}
		}
	}
}

void random_dithering_1bit() {
	int x, y;
	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			int power = (source[x][y].r + source[x][y].g + source[x][y].b);
			if(power > 250 + (rand()%130)) {
				set_pixel_color(&result[x][y], 255,255,255);
			} else {
				set_pixel_color(&result[x][y], 0, 0, 0);
			}
		}
	}
}

void to_8bit() {
	int x, y;
	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			result[x][y].r = truncate(source[x][y].r, 255, 8);
			result[x][y].g = truncate(source[x][y].g, 255, 8);
			result[x][y].b = truncate(source[x][y].b, 255, 4); 
		}
	}
}

void random_dithering_8bit() {
	int x, y;
	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			set_pixel_color(&result[x][y],
				truncate((source[x][y].r - 20) + rand()%40, 255, 8),
				truncate((source[x][y].g - 20) + rand()%40, 255, 8),
				truncate((source[x][y].b - 20) + rand()%40, 255, 4)
			);
		}
	}
}

float threshold[2][2] = {
	{1.0/5.0, 3.0/5.0},
	{4.0/5.0, 2.0/5.0}
};

void ordered_dithering_1bit() {
	int x, y;

	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) { 
			int power = (source[x][y].r + source[x][y].g + source[x][y].b);
			power += threshold[x%2][y%2];

			if(power > 335) {
				set_pixel_color(&result[x][y], 255, 255, 255);
			} else {
				set_pixel_color(&result[x][y], 0, 0, 0);
			}
		}
	}
}

void ordered_dithering_8bit() {
	int x, y;
	float threshld;
	
	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			threshld = threshold[x%2][y%2];
			set_pixel_color(&result[x][y],
				truncate((float)source[x][y].r + (threshld*20), 255, 8),
				truncate((float)source[x][y].g + (threshld*20), 255, 8),
				truncate((float)source[x][y].b + (threshld*20), 255, 4)
			);
		}
	}
}

void error_diff_dither_1bit() {
	int x, y;

	static float workspace[TEX_SIZE][TEX_SIZE];
	float error;

	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			workspace[x][y] = (
				0.3 * source[x][y].r +
				0.59 * source[x][y].g +
				0.11 * source[x][y].b
			) / 255.0;
		}
	}

	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			if(workspace[x][y] > 0.5) {
				set_pixel_color(&result[x][y], 255,255,255);
				error = workspace[x][y] - 1;
			} else {
				set_pixel_color(&result[x][y], 0, 0, 0);
				error = workspace[x][y];
			}

			if(x<TEX_SIZE) {
				workspace[x+1][y] += 7.0/16.0*error;
			}

			if(x>0 && y<TEX_SIZE) {
				workspace[x-1][y+1] += 3.0/16.0*error;
			}

			if(y < TEX_SIZE) {
				workspace[x][y+1] += 5.0/16.0*error;
			}

			if(x < TEX_SIZE && y < TEX_SIZE) {
				workspace[x+1][y+1] += 1.0/16.0*error;
			}
		}
	}
}

void error_diff_dither_8bit() {
	int x, y, c;
	float error;

	static pixel workspace[TEX_SIZE][TEX_SIZE];
	memcpy(&workspace, source, sizeof(workspace)); 

	int channel_sizes[3] = {8,8,4};

	for(x = 0; x < TEX_SIZE; x++) {
		for(y = 0; y < TEX_SIZE; y++) {
			char unsigned *channels = (char unsigned *)&workspace[x][y];
			char unsigned *res_channels = (char unsigned *)&result[x][y];
			char unsigned *error_channels;
			for(c = 0; c < 3; c++) {
				res_channels[c] = truncate(channels[c], 255, channel_sizes[c]);
				error = channels[c] - res_channels[c];

				if(x<TEX_SIZE) {
					error_channels = (char unsigned *)&workspace[x+1][y]; 
					error_channels[c] += 7.0/16.0*error;
				}
				if(x>0 && y<TEX_SIZE) {
					error_channels = (char unsigned *)&workspace[x-1][y+1]; 
					error_channels[c] += 3.0/16.0*error;
				}
				if(y < TEX_SIZE) {
					error_channels = (char unsigned *)&workspace[x][y+1]; 
					error_channels[c] += 5.0/16.0*error;
				}
				if(y < TEX_SIZE && y < TEX_SIZE) {
					error_channels = (char unsigned *)&workspace[x+1][y+1]; 
					error_channels[c] += 1.0/16.0*error;
				}
			}
		}
	}
}

void (*reduce)() = to_1bit;

/*
 * Handles keyboard input, switch modes by char 'm' and 
 * controll chars for variations are q,w,e,r,t,z,u,i
 * in layers you can use a and d to move animation
 */
void handle_keyboard(unsigned char ch, int x, int y) {
    switch(ch) {
        case 'q':
            effect = blur;
			reduce = to_3bit;
            break;
        case 'w':
            effect = sharpen;
			reduce = to_1bit;
            break;
        case 'e':
            effect = edge_detection1;
			reduce = to_8bit;
            break;
		case 'r':
			effect = edge_detection3;
			reduce = random_dithering_1bit;
			break;
		case 't':
			effect = blur5x;
			reduce = random_dithering_8bit;
			break;
		case 'z':
			effect = emboss;
			reduce = ordered_dithering_1bit;
			break;
		case 'u':
			effect = to_grayscale;
			reduce = ordered_dithering_8bit;
			break;
		case 'i':
			reduce = error_diff_dither_1bit;
			break;
		case 'o':
			reduce = error_diff_dither_8bit;
			to_grayscale(result);
		case 'a':
			if(alpha_x < 255) {
				alpha_x++;
			}
			break;
		case 'd':
			if(alpha_x > 0) {
				alpha_x--;
			}
			break;
		case 'm':
			if(mode == DISPLAY_TO_LESSBIT) {
				mode = DISPLAY_EFFECT;
			} else {
				mode++;
			}
			break;
    }

	reduced = 0;
}

// Initialize OpenGL state
void init() {
	// Texture setup
    glEnable(GL_TEXTURE_2D);
    glGenTextures( 1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

    // Other
    glClearColor(0,0,0,0);
    gluOrtho2D(-1,1,-1,1);
    glLoadIdentity();
    glColor3f(1,1,1);
    load_rgb(source, "./img/image.rgb", TEX_SIZE);
	load_rgba(layer1,"./img/image.rgba", TEX_SIZE);
	load_rgba(layer2, "./img/top.rgba", TEX_SIZE);

}

// Generate and display the image.
void display() {
    // Call user image generation
	
	switch(mode) {
		case DISPLAY_LAYERS:
			reduced = 0;
			memcpy(*result, *source, TEX_SIZE*TEX_SIZE*sizeof(pixel));
			blend_layer_at(result, layer1, alpha_x, 0);
			blend_layer_at(result, layer2, 50, 50);
			break;
		case DISPLAY_EFFECT:
			reduced = 0;
			effect();
			break;
		case DISPLAY_FRACTAL:
			reduced = 0;
			print_fractal();
			break;
		case DISPLAY_TO_LESSBIT:
			if(!reduced) {
				reduce();
				reduced = 1;
			}
			break;
	}

    // Copy image to texture memory
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, TEX_SIZE, TEX_SIZE, 0, GL_RGB, GL_UNSIGNED_BYTE, result);
    // Clear screen buffer
    glClear(GL_COLOR_BUFFER_BIT);
    // Render a quad
    glBegin(GL_QUADS);
        glTexCoord2f(0,0); glVertex2f(1,1);
        glTexCoord2f(0,1); glVertex2f(1,-1);
        glTexCoord2f(1,1); glVertex2f(-1,-1);
        glTexCoord2f(1,0); glVertex2f(-1,1);
    glEnd();
    // Display result
    glFlush();
    glutPostRedisplay();
    glutSwapBuffers();
}

// Main entry function
int main(int argc, char ** argv) {
    // Init GLUT
    glutInit(&argc, argv);
    glutInitWindowSize(TEX_SIZE, TEX_SIZE);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGB);
    glutCreateWindow("OpenGL Window");
    // Set up OpenGL state
    init();
    // Run the control loop
    glutDisplayFunc(display);
    glutKeyboardFunc(handle_keyboard);
    glutMainLoop();
    return EXIT_SUCCESS;
}
