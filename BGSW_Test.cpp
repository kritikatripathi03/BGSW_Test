#include <iostream>
#include<opencv2/opencv.hpp>
using namespace cv;

int main(int c, char* v[]) {

    // Parsing command line arguments
    if (c != 3) {
        std::cerr << "Usage: " << v[0] << " <input_path> <base_address>" << std::endl;
        return -1;
    }
    std::string input_path = v[1];
    std::string memory_location_str = v[2];

    // Convert the memory location string to an integer
    std::istringstream iss(memory_location_str);
    unsigned long long memory_location;
    if (!(iss >> std::hex >> memory_location)) {
        std::cerr << "Invalid memory location: " << memory_location_str << std::endl;
        return 1;
    }

    // Open the video capture device
    VideoCapture capture(input_path);
    if (!capture.isOpened()) {
        std::cerr << "Error: Couldn't open video capture device." << std::endl;
        return -1;
    }

    // Get the frame size
    size_t frame_size = capture.get(CAP_PROP_FRAME_WIDTH) * capture.get(CAP_PROP_FRAME_HEIGHT) * capture.get(CAP_PROP_FRAME_COUNT);

    // Set the base address
    uintptr_t baseAddress;
    baseAddress = memory_location;

    // Allocate memory at the specified base address
    int* base_address = reinterpret_cast<int*>(baseAddress);

    // Set the values of the base address
    new (&base_address[0]) int(capture.get(CAP_PROP_FRAME_WIDTH));
    new (&base_address[1]) int(capture.get(CAP_PROP_FRAME_HEIGHT));
    new (&base_address[2]) int(capture.get(CAP_PROP_FRAME_COUNT));
    new (&base_address[3]) int(capture.get(CAP_PROP_FPS));

    // Create a pointer to the base address
    unsigned char* base_pointer = reinterpret_cast<unsigned char*>(base_address);

    // Create a video writer object
    VideoWriter video("output.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), base_address[3], Size(base_address[0], base_address[1]), true);

    // Create matrices dynamically at the specified base address
    Mat frame(base_address[1], base_address[0], CV_8UC3, base_pointer);
    base_pointer += frame_size;
    Mat gray(base_address[1], base_address[0], CV_8UC3, base_pointer);
    base_pointer += frame_size;
    Mat kernel(base_address[1], base_address[0], CV_8UC3, base_pointer);
    base_pointer += frame_size;
    Mat foreground(base_address[1], base_address[0], CV_8UC3, base_pointer);
    base_pointer += frame_size;

    // Initialize background subtractor
    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorKNN();

    // Process the video
    while (true) {
        capture >> frame;
        if (frame.empty())
            break;

        // Convert frame to grayscale
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        // Apply background subtraction
        pBackSub->apply(gray, foreground);

        // Apply morphological operations
        cv::threshold(foreground, foreground, 50, 255, THRESH_BINARY);

        //  Apply dilation
        kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
        cv::dilate(foreground, foreground, kernel, Point(-1, -1), 2);
        std::vector<std::vector<cv::Point>> contours;// = new(base_pointer) std::vector<std::vector<cv::Point>>;

        // Find contours
        cv::findContours(foreground, contours, RETR_LIST, CHAIN_APPROX_NONE);

        // Draw contours around detected objects
        for (size_t i = 0; i < contours.size(); ++i) {
            if (cv::contourArea(contours[i]) < 500)
                continue;
            cv::drawContours(frame, contours, i, cv::Scalar(0, 255, 0), 2);
        }

        video.write(frame);

        // Break the loop if 'q' is pressed
        if (waitKey(10) == 'q')
            break;
    }

    // Release the video capture device and the video writer
    capture.release();
    video.release();
    cv::destroyAllWindows();

    return 0;
}