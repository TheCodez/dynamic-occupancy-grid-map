/*
MIT License

Copyright (c) 2019 Michael Kösel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once

#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <dogm/dogm.h>
#include <dogm/dogm_types.h>

namespace dogm_ros
{

class DOGMRos
{
public:
	DOGMRos(ros::NodeHandle nh, ros::NodeHandle private_nh);
	
	virtual ~DOGMRos() = default;
	
	void process(const sensor_msgs::LaserScan::ConstPtr& scan);
	
private:
	ros::NodeHandle nh_;
	ros::NodeHandle private_nh_;
	
	ros::Subscriber subscriber_;
	ros::Publisher publisher_;
	
	dogm::GridParams params_;
	dogm::LaserSensorParams laser_params_;
	
	float last_time_stamp_;
	bool is_first_measurement_;
	
	boost::shared_ptr<dogm::DOGM> grid_map_;
};

} // namespace dogm_ros
