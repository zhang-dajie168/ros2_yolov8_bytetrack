#include <opencv2/opencv.hpp>
#include "task/yolov8.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"
#include "bytetrack/BYTETracker.h"

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/polygon_stamped.hpp"

#include <chrono>
#include<image_transport/image_transport.hpp>
#include<iostream>
class Yolov8TrackerNode : public rclcpp::Node
{
public:
    Yolov8TrackerNode() : Node("yolov8_tracker_node")
    {
        // 获取参数
        this->declare_parameter<std::string>("model_file", "yolov8s.int.rknn");
        
        std::string model_file = this->get_parameter("model_file").as_string();

        // 初始化 YOLO 模型
        //Yolov8Custom yolo;
        //yolo.LoadModel(model_file.c_str());
        
        // 初始化订阅器，订阅图像话题
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>
        (
            "/camera/color/image_raw", 
            rclcpp::SensorDataQoS().keep_last(30),
            [this](sensor_msgs::msg::Image::ConstSharedPtr msg){this->image_callback(msg);}  
         );
        image_pub_=this->create_publisher<sensor_msgs::msg::Image>("output_image",30);
        
        tracked_pub_=this->create_publisher<geometry_msgs::msg::PolygonStamped>("/tracked_objects",30);
        
        yolo_=std::make_unique<Yolov8Custom>();
        yolo_->LoadModel(model_file.c_str());

	//result_pub_=it.advertise("result_topic",30);
	
        // 跟踪器初始化
        //std::make_unique<BYTETracker>tracker_(30,30);
        tracker_=std::make_unique<BYTETracker>(30, 30);
    }


private:
    void image_callback(sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        try
        {  
            // 使用 cv_bridge 将 ROS 图像消息转换为 OpenCV 图像
            cv::Mat img = cv_bridge::toCvCopy(msg, "bgr8")->image;
            //cv_bridge::CvImagePtr img=cv_bridge::toCvCopy(msg, "bgr8");
                    // 开始计时
            auto start_1 = std::chrono::high_resolution_clock::now();
            // 进行目标检测
            if (img.empty()){
            
            std::cerr<<"Error:Image not loaded!!!"<<std::endl;
            }
            std::vector<Detection> objects;
            if(yolo_){
            yolo_->Run(img, objects);
		}
		
	    auto start_2 = std::chrono::high_resolution_clock::now();
	    auto yolo_time = std::chrono::duration_cast<std::chrono::microseconds>(start_2 - start_1).count() / 1000.0;
	    std::cout<<"yolo_time:"<<yolo_time<<"ms"<<std::endl;
            
            // 将检测结果转为跟踪对象
            std::vector<Object> trackobj;
            publish_decobj_to_trackobj(objects, trackobj);

            // 更新追踪器
            auto output_stracks = tracker_->update(trackobj);
            
	      // 结束计时
            auto end = std::chrono::high_resolution_clock::now();
            auto track_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start_2).count() / 1000.0;
            auto all_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start_1).count() / 1000.0;
            std::cout<<"track_time:"<<track_time<<"ms"<<std::endl; 
	    std::cout<<"all_time:"<<all_time<<"ms"<<std::endl;
            
            // 在图像上绘制跟踪结果
            draw_tracking_results(img, output_stracks);
	    
            sensor_msgs::msg::Image::SharedPtr output_msg=cv_bridge::CvImage(msg->header,"bgr8",img).toImageMsg();
            image_pub_->publish(*output_msg);



        } catch (const cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
        }
    }

    void publish_decobj_to_trackobj(std::vector<Detection> &objects, std::vector<Object> &trackobj)
    {
        geometry_msgs::msg::PolygonStamped polygon_msg;
        polygon_msg.header.stamp=this->get_clock()->now();
        polygon_msg.header.frame_id="camera_link";
        
        if (!objects.empty())
        {
            trackobj.clear();
        }
        for (auto &obj : objects)
        {
            Object trackobj_temp;
            trackobj_temp.classId = obj.class_id;
            trackobj_temp.score = obj.confidence;
            trackobj_temp.box = obj.box;
            trackobj.push_back(trackobj_temp);
        }
        
        tracked_pub_->publish(polygon_msg);
    }

    void draw_tracking_results(cv::Mat &img, const std::vector<STrack> &tracks)
    {
        std::cout<<"Tracking result (output_stracks):"<<std::endl;
        for (const auto &track : tracks)
        {
            // 画出跟踪框
            std::cout<<"track_id: "<<track.track_id<<",Track  Bounding Box:["<<track.tlbr[0]<<","<<track.tlbr[1]<<","<<track.tlbr[2]<<","<<track.tlbr[3]<<"]"<<",sorce:"<<track.score<<std::endl;
            int x1=static_cast<int>(track.tlbr[0]);
            int y1=static_cast<int>(track.tlbr[1]);
            int x2=static_cast<int>(track.tlbr[2]);
            int y2=static_cast<int>(track.tlbr[3]);
            cv::Rect rect(x1,y1,x2-x1,y2-y1);
            cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

            // 显示跟踪 ID
            int text_x=x1;
            int text_y=y1-5;
            
            //float b1=track.tlbr.tl[1];
            //cv::Point test_origin(static_cast<int>(a1),static_cast<int>(b1));
            cv::putText(img, std::to_string(track.track_id), cv::Point(text_x,text_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            //sensor_msgs::msg::Image::SharedPtr img=
            //cv_bridge::CvImage(std_msgs::msg::Header(),"bgr8",img).toImageMsg();
            //result_pub_.publish(img);
        }
        
    }
    
    std::unique_ptr<BYTETracker>tracker_;
    std::unique_ptr<Yolov8Custom>yolo_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PolygonStamped>::SharedPtr tracked_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    //image_transport::Publisher result_pub_;
    //std::unique_ptr<BYTETracker> tracker_;
};

int main(int argc, char **argv)
{
    rclcpp::InitOptions init_options;
    init_options.shutdown_on_signal=true;
    
    
    rclcpp::init(argc, argv,init_options);
    rclcpp::spin(std::make_shared<Yolov8TrackerNode>());
    rclcpp::shutdown();
    return 0;
}

