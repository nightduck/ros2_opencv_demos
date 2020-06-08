#include <stdlib.h>
#include <functional>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "image_transport/image_transport.h"
#include <cv_bridge/cv_bridge.h>
#include "sensor_msgs/image_encodings.hpp"

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

class ImageConverter : public rclcpp::Node {
private:
    const std::string OPENCV_WINDOW = "Image window";
//    image_transport::ImageTransport it;
    image_transport::Subscriber sub_;
    image_transport::Publisher pub_;

    void detectAndDisplay( cv::Mat& frame )
    {
        cv::Mat frame_gray;
        cv::cvtColor( frame, frame_gray, cv::COLOR_BGR2GRAY );
        cv::equalizeHist( frame_gray, frame_gray );
        //-- Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale( frame_gray, faces );
        for ( size_t i = 0; i < faces.size(); i++ )
        {
            cv::Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
            cv::ellipse( frame, center, cv::Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, cv::Scalar( 255, 0, 255 ), 4 );
            cv::Mat faceROI = frame_gray( faces[i] );
            //-- In each face, detect eyes
            std::vector<cv::Rect> eyes;
            eyes_cascade.detectMultiScale( faceROI, eyes );
            for ( size_t j = 0; j < eyes.size(); j++ )
            {
                cv::Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( frame, eye_center, radius, cv::Scalar( 255, 0, 0 ), 4 );
            }
        }
        //-- Show what you got
        imshow(OPENCV_WINDOW, frame );
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        detectAndDisplay(cv_ptr->image);
        cv::waitKey(3);

        pub_.publish(cv_ptr->toImageMsg());
    }

public:
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier eyes_cascade;

    ImageConverter() : Node("image_converter") {
//    ImageConverter() : Node("image_converter"), it(this->get()) {

        // Open demo window that will show output image
        cv::namedWindow(OPENCV_WINDOW);

        cv::String face_cascade_name = "haarcascade_frontalface_alt.xml";
        cv::String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

        //-- 1. Load the cascades
        if( !face_cascade.load( face_cascade_name ) )
        {
            std::cout << "--(!)Error loading face cascade\n";
        };
        if( !eyes_cascade.load( eyes_cascade_name ) )
        {
            std::cout << "--(!)Error loading eyes cascade\n";
        };

        rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
        pub_ = image_transport::create_publisher(this, "out_image_base_topic", custom_qos);
        sub_ = image_transport::create_subscription(this, "in_image_base_topic",
                std::bind(&ImageConverter::imageCallback, this, std::placeholders::_1), "raw", custom_qos);

//        pub = it.advertise("out_image_base_topic", 1);
//        sub = it.subscribe("in_image_base_topic", 1, imageCallback);
    }

    ~ImageConverter()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageConverter>());
    rclcpp::shutdown();
    return 0;
}