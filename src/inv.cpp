#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include <iostream>
#include <fstream>
#include <string>
#include <iterator>
#include <cassert>

#include <gperftools/profiler.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
// This function is used for 2nd approach described in next section (standard OpenCL kernel dispatch)
extern void ProcRawCL(Mat &mat_src, const string &kernel_name);
int main()
{
  // ProfilerStart("inv.prof");
    if (!ocl::haveOpenCL())
    {
        cout << "OpenCL is not avaiable..." << endl;
        return 0;
    }
    ocl::Context context;
    if (!context.create(ocl::Device::TYPE_GPU))
    {
        cout << "Failed creating the context..." << endl;
        return 0;
    }
    // Select the first device
    ocl::Device(context.device(0));
    // Read the OpenCL kernel code into a string
    ifstream ifs("../OpenCL/kernel_inv.cl");
    if (ifs.fail()) return 0;
    std::string kernelSource((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ocl::ProgramSource programSource(kernelSource);
    // Compile the kernel code
    cv::String errmsg;
    cv::String buildopt = "-DDBG_VERBOSE "; // We can set various clocl build options here, e.g. define-s to compile-in/out parts of CL code
    ocl::Program program = context.getProg(programSource, buildopt, errmsg);
    ocl::Kernel kernel("invert_img", program);
    // Transfer Mat data to the device
    Mat mat_src = imread("../image/1.jpeg", IMREAD_GRAYSCALE);
    // imshow("src", mat_src);

    // return 1;
    UMat umat_src = mat_src.getUMat(ACCESS_READ, USAGE_ALLOCATE_DEVICE_MEMORY);
    cout << "Input image size: " << mat_src.size() << endl << flush;
    UMat umat_dst(mat_src.size(), mat_src.type(), ACCESS_WRITE, USAGE_ALLOCATE_DEVICE_MEMORY);
    kernel.args(ocl::KernelArg::ReadOnlyNoSize(umat_src), ocl::KernelArg::ReadWrite(umat_dst));
    size_t globalThreads[2] = { (unsigned int)mat_src.cols, (unsigned int)mat_src.rows };
    size_t localThreads[2] = { 16, 16 };
    bool success = kernel.run(2, globalThreads, localThreads, false);
    if (!success){
      cout << "Failed running the kernel..." << endl;
      return 0;
    } else {
      cout << "Kernel OK!" << endl;
    }
    GaussianBlur(umat_dst, umat_dst, Size(5, 5), 1.25);
    Canny(umat_dst, umat_dst, 0, 50);
    // Fetch the dst data from the device
    Mat mat_dst = umat_dst.getMat(ACCESS_READ);
    imwrite("out1.jpg", mat_dst);
    // ProcRawCL(mat_src, "kernel_direct.cl");
    imshow("src", mat_src);
    imshow("dst", mat_dst);
  //  ProfilerStop();
   waitKey();
    return 1;
}