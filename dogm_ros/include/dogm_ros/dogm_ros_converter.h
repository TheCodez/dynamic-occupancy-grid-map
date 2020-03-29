#pragma once

#include <dogm/dogm.h>
#include <dogm/dogm_types.h>
#include <dogm_msgs/DynamicOccupancyGrid.h>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>

namespace dogm_ros
{

class DOGMRosConverter
{
public:
 
  DOGMRosConverter();

  virtual ~DOGMRosConverter();

  static void toDOGMMessage(const dogm::DOGM& dogm, dogm_msgs::DynamicOccupancyGrid& message);

  static void toOccupancyGridMessage(const dogm::DOGM& dogm, nav_msgs::OccupancyGrid& message);
};

} /* namespace dogm_ros */
