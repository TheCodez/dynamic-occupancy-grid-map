#include "dogm_ros/DOGMRosConverter.h"

namespace dogm
{

DOGMRosConverter::DOGMRosConverter()
{
}

DOGMRosConverter::~DOGMRosConverter()
{
}

void DOGMRosConverter::toDOGMMessage(const dogm::DOGM& dogm, dogm_msgs::DynamicOccupancyGrid& message)
{
  message.header.stamp = ros::Time::now();
  message.info.resolution = dogm.getResolution();
  message.info.length = dogm.getGridSize() * dogm.getResolution();
  message.info.size = dogm.getGridSize();
  message.info.pose.position.x = dogm.getPositionX();
  message.info.pose.position.y = dogm.getPositionY();
  message.info.pose.position.z = 0.0;
  message.info.pose.orientation.x = 0.0;
  message.info.pose.orientation.y = 0.0;
  message.info.pose.orientation.z = 0.0;
  message.info.pose.orientation.w = 1.0;

  message.data.clear();
  message.data.resize(dogm.getGridSize() * dogm.getGridSize());

  #pragma omp parallel for
  for (int i = 0; i < message.data.size(); i++)
  {
    dogm::GridCell& cell = dogm.grid_cell_array[i];

    message.data[i].free_mass = cell.free_mass;
    message.data[i].occ_mass = cell.occ_mass;

    message.data[i].mean_x_vel = cell.mean_x_vel;
    message.data[i].mean_y_vel = cell.mean_y_vel;
    message.data[i].var_x_vel = cell.var_x_vel;
    message.data[i].var_y_vel = cell.var_y_vel;
    message.data[i].covar_xy_vel = cell.covar_xy_vel;
  }
}

void DOGMRosConverter::toOccupancyGridMessage(const dogm::DOGM& dogm, nav_msgs::OccupancyGrid& message)
{
  message.header.stamp = ros::Time::now();
  message.info.map_load_time = message.header.stamp;
  message.info.resolution = dogm.getResolution();
  message.info.width = dogm.getGridSize();
  message.info.height = dogm.getGridSize();
 
  float positionX = dogm.getPositionX() - 0.5 * dogm.getGridSize();
  float positionY = dogm.getPositionY() - 0.5 * dogm.getGridSize();
  message.info.origin.position.x = positionX;
  message.info.origin.position.y = positionY;
  message.info.origin.position.z = 0.0;
  message.info.origin.orientation.x = 0.0;
  message.info.origin.orientation.y = 0.0;
  message.info.origin.orientation.z = 0.0;
  message.info.origin.orientation.w = 1.0;

  message.data.clear();
  message.data.resize(dogm.getGridSize() * dogm.getGridSize());

  #pragma omp parallel for
  for (int i = 0; i < message.data.size(); i++)
  {
    dogm::GridCell& cell = dogm.grid_cell_array[i];
    float free_mass = cell.free_mass;
    float occ_mass = cell.occ_mass;
    
	float prob = occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
	
	if (prob == 0.5f)
	{
        message.data[i] = -1;
    }
    else
    {      
        message.data[i] = static_cast<int>(prob * 100.0f);
    } 
  }
}

} /* namespace dogm */
