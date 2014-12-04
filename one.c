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
#include <omp.h>

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
void (*effect)() = blur;

int measure = 0;
/*
 * Handles keyboard input, switch modes by char 'm' and 
 * controll chars for variations are q,w,e,r,t,z,u,i
 * in layers you can use a and d to move animation
 */
void handle_keyboard(unsigned char ch, int x, int y) {
    switch(ch) {
        case 'q':
            effect = blur;
            break;
        case 't':
            effect = blur5x;
            break;
    }
    measure = 1;

    reduced = 0;
}

int start_time;

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
}

// Generate and display the image.
void display() {
    // Call user image generation
    
    if(measure) printf("Start effect\n");
    start_time = clock();
    effect();
    if(measure) {
        printf("Time of generation %d\n\n", (int)(clock() - start_time));
        measure = 0;
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
