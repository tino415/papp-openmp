#
# Makefile for OpenGL demo programs
#
# "make program" to make one program
# "make" or "make all" to make all executables
# "make clean" to remove executables
#

CC      = mpicc
CFLAGS  = -O0 -Wall -g
UNAME := $(shell uname -s)

ALL =   main_mpi one

all:  $(ALL)

%: %.c ; $(CC) -o $@ $(CFLAGS) $< $(LFLAGS)

LFLAGS = -lm -lGLEW -lGL -lGLU -lglut -fopenmp
clean: ; -rm $(ALL)
