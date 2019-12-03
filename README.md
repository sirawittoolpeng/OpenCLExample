# OpenCLExample
dumb ass rep for learning OpenCL for me

# Compiling
$ gcc main.c -o vectorAddition -l OpenCL

g++ `pkg-config --cflags --libs /installation/OpenCV-3.4.4/lib/pkgconfig/opencv.pc` gaussianblur.cpp -o gaussianblur