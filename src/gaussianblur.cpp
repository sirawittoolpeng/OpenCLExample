#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <CL/cl.h>

int main(int argc, char **argv){

    cv::Mat src = cv::imread("../image/1.jpeg");
    cv::UMat uorig, ublur;
    src.copyTo(uorig);
    cv::ocl::setUseOpenCL(true);
     cv::GaussianBlur(uorig, ublur, cv::Size(5, 5), 0.8);
     cv::imshow("orig", src);
     cv::imshow("blured", ublur);

   cv::waitKey();
   
   return 0;
}