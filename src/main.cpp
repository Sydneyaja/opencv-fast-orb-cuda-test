#include <opencv2/opencv.hpp>

using namespace std;

int main( int argc, char** argv )
{
    cout << "Hello, GpuTest!" << endl;

    // Open camera
    cv::VideoCapture videoCapture(0);

    // CPU FAST detector
    cv::Ptr<cv::FastFeatureDetector> ptrFAST_CPU = cv::FastFeatureDetector::create();

    // CUDA FAST detector
    cv::Ptr<cv::cuda::FastFeatureDetector> ptrFAST_CUDA = cv::cuda::FastFeatureDetector::create();

    // CPU ORB descriptor
    cv::Ptr<cv::ORB> ptrORB_CPU = cv::ORB::create();

    // CUDA ORB descriptor
    cv::Ptr<cv::cuda::ORB> ptrORB_CUDA = cv::cuda::ORB::create();

    // Frame grabbing loop
    while(true)
    {
        cout << "------------------" << endl;

        // Get frame
        cv::Mat img;
        videoCapture >> img;

        // Quit if empty frame
        if(img.empty())
            break;

        // Convert frame to gray
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        // Create CUDA mat and upload image to GPU
        cv::cuda::GpuMat gpuGray;
        gpuGray.upload(gray);

        // Chrono to measure duration
        std::chrono::steady_clock::time_point Tbefore, Tafter;
        Tbefore = std::chrono::steady_clock::now();

        // Detect FAST CPU
        std::vector<cv::KeyPoint> keypoints_CPU;
        ptrFAST_CPU->detect(gray, keypoints_CPU);

        // Display duration
        Tafter = std::chrono::steady_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double> >(Tafter - Tbefore).count();
        cout << std::setprecision(5) << "FAST CPU duration: " << duration << " sec" << endl;

        // Chrono to measure duration
        Tbefore = std::chrono::steady_clock::now();

        // Detect FAST GPU
        std::vector<cv::KeyPoint> keypoints_GPU;
        ptrFAST_CUDA->detect(gpuGray, keypoints_GPU);

        // Display duration
        Tafter = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double> >(Tafter - Tbefore).count();
        cout << std::setprecision(5) << "FAST GPU duration: " << duration << " sec" << endl;

        // Chrono to measure duration
        Tbefore = std::chrono::steady_clock::now();

        // Describe ORB GPU
        cv::Mat descriptor_CPU;
        ptrORB_CPU->compute(gray, keypoints_CPU, descriptor_CPU);

        // Display duration
        Tafter = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double> >(Tafter - Tbefore).count();
        cout << std::setprecision(5) << "ORB CPU duration: " << duration << " sec" << endl;

        // Chrono to measure duration
        Tbefore = std::chrono::steady_clock::now();

        // Describe ORB GPU
        cv::cuda::GpuMat descriptor_GPU;
        ptrORB_CUDA->compute(gpuGray, keypoints_GPU, descriptor_GPU);

        // Display duration
        Tafter = std::chrono::steady_clock::now();
        duration = std::chrono::duration_cast<std::chrono::duration<double> >(Tafter - Tbefore).count();
        cout << std::setprecision(5) << "ORB GPU duration: " << duration << " sec" << endl;

        // Show image
        cv::imshow("Image", img);

        // Waitkey
        int key = cv::waitKey(30);

        // Manage key entry
        if(key == 113 /*Q key*/)
        {
            cout << "Exit" << endl;
            break;
        }
    }
}