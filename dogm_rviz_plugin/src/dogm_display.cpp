/*
 * Based on rviz/default_plugin/map_display.cpp
 * 
 * Copyright (c) 2008, Willow Garage, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Willow Garage, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include "dogm_rviz_plugin/dogm_display.h"

#include <OGRE/OgreManualObject.h>
#include <OGRE/OgreMaterialManager.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreTextureManager.h>

#include <ros/ros.h>

#include <tf/transform_listener.h>

#include "rviz/frame_manager.h"
#include "rviz/ogre_helpers/grid.h"
#include "rviz/properties/float_property.h"
#include "rviz/properties/int_property.h"
#include "rviz/properties/property.h"
#include "rviz/properties/quaternion_property.h"
#include "rviz/properties/ros_topic_property.h"
#include "rviz/properties/vector_property.h"
#include "rviz/validate_floats.h"
#include "rviz/display_context.h"

#include <Eigen/Dense>

namespace dogm_rviz_plugin
{

DOGMDisplay::DOGMDisplay()
  : loaded_(false)
  , resolution_(0.0f)
  , size_(0)
{
  alpha_property_ = new rviz::FloatProperty("Alpha", 1.0,
                                            "Amount of transparency to apply to the map.",
                                            this, SLOT( updateAlpha()));
  alpha_property_->setMin(0.0f);
  alpha_property_->setMax(1.0f);

  draw_under_property_ = new rviz::BoolProperty("Draw Behind", false,
                                                "Rendering option, controls whether or not the map is always"
                                                " drawn behind everything else.",
                                                this, SLOT(updateDrawUnder()));

  resolution_property_ = new rviz::FloatProperty("Resolution", 0,
                                                 "Resolution of the map. (not editable)", this);
  resolution_property_->setReadOnly(true);

  size_property_ = new rviz::IntProperty("Size", 0,
                                         "Size of the map, in pixel. (not editable)", this);
  size_property_->setReadOnly(true);
  
  position_property_ = new rviz::VectorProperty("Position", Ogre::Vector3::ZERO,
                                                "Position of the bottom left corner of the map, in meters. (not editable)",
                                                this);
  position_property_->setReadOnly(true);

  orientation_property_ = new rviz::QuaternionProperty("Orientation", Ogre::Quaternion::IDENTITY,
                                                       "Orientation of the map. (not editable)",
                                                       this);
  orientation_property_->setReadOnly(true);

  occ_property_ = new rviz::FloatProperty("Occupancy threshold", 1.0,
                                          "Occupancy amount at which object is considered dynamic.",
                                          this);
  occ_property_->setMin(0.0f);
  occ_property_->setMax(1.0f);

  mahalanobis_property_ = new rviz::FloatProperty("Mahalanobis distance", 1.0,
                                                  "Mahalanobis distance at which object is considered dynamic.",
                                                  this);
  mahalanobis_property_->setMin(0.0f);
}

DOGMDisplay::~DOGMDisplay()
{
  clear();
}

void DOGMDisplay::onInitialize()
{
  MFDClass::onInitialize();

  static int count = 0;
  std::stringstream ss;
  ss << "DOGMObjectMaterial" << count++;
  material_ = Ogre::MaterialManager::getSingleton().create(ss.str(),
                                                           Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
  material_->setReceiveShadows(false);
  material_->getTechnique(0)->setLightingEnabled(false);
  material_->setDepthBias(-16.0f, 0.0f);
  material_->setCullingMode(Ogre::CULL_NONE);
  material_->setDepthWriteEnabled(false);

  updateAlpha();
}

void DOGMDisplay::onDisable()
{
  clear();
}

void DOGMDisplay::reset()
{
  MFDClass::reset();
  clear();
}

void DOGMDisplay::updateAlpha()
{
  float alpha = alpha_property_->getFloat();

  Ogre::Pass* pass = material_->getTechnique(0)->getPass(0);
  Ogre::TextureUnitState* tex_unit = NULL;

  if (pass->getNumTextureUnitStates() > 0)
  {
    tex_unit = pass->getTextureUnitState(0);
  }
  else
  {
    tex_unit = pass->createTextureUnitState();
  }

  tex_unit->setAlphaOperation(Ogre::LBX_SOURCE1, Ogre::LBS_MANUAL, Ogre::LBS_CURRENT, alpha);

  if (alpha < 0.9998)
  {
    material_->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
    material_->setDepthWriteEnabled(false);
  }
  else
  {
    material_->setSceneBlending(Ogre::SBT_REPLACE);
    material_->setDepthWriteEnabled(!draw_under_property_->getValue().toBool());
  }
}

void DOGMDisplay::updateDrawUnder()
{
  bool draw_under = draw_under_property_->getValue().toBool();

  if (alpha_property_->getFloat() >= 0.9998)
  {
    material_->setDepthWriteEnabled(!draw_under);
  }

  if (manual_object_)
  {
    if (draw_under)
    {
      manual_object_->setRenderQueueGroup(Ogre::RENDER_QUEUE_4);
    }
    else
    {
      manual_object_->setRenderQueueGroup(Ogre::RENDER_QUEUE_MAIN);
    }
  }
}

void DOGMDisplay::clear()
{
  setStatus(rviz::StatusProperty::Warn, "Message", "No map received");

  if (!loaded_)
  {
    return;
  }

  scene_manager_->destroyManualObject(manual_object_);
  manual_object_ = NULL;

  std::string tex_name = texture_->getName();
  texture_.setNull();
  Ogre::TextureManager::getSingleton().unload(tex_name);

  loaded_ = false;
}

float pignisticTransformation(float free_mass, float occ_mass)
{
    return occ_mass + 0.5f * (1.0f - occ_mass - free_mass);
}

void hsvToRGB(float hue, float saturation, float value, int& R, int& G, int& B)
{
    float r, g, b = 0.0f;

    if (saturation == 0.0f)
    {
        r = g = b = value;
    }
    else
    {
        int i = static_cast<int>(hue * 6.0f);
        float f = (hue * 6.0f) - i;
        float p = value * (1.0f - saturation);
        float q = value * (1.0f - saturation * f);
        float t = value * (1.0f - saturation * (1.0f - f));
        int res = i % 6;

        switch (res)
        {
        case 0:
            r = value;
            g = t;
            b = p;
            break;
        case 1:
            r = q;
            g = value;
            b = p;
            break;
        case 2:
            r = p;
            g = value;
            b = t;
            break;
        case 3:
            r = p;
            g = q;
            b = value;
            break;
        case 4:
            r = t;
            g = p;
            b = value;
            break;
        case 5:
            r = value;
            g = p;
            b = q;
            break;
        default:
            r = g = b = value;
        }
    }

    R = static_cast<int>(r * 255.0f);
    G = static_cast<int>(g * 255.0f);
    B = static_cast<int>(b * 255.0f);
}

void DOGMDisplay::processMessage(const dogm_msgs::DynamicOccupancyGrid::ConstPtr& msg)
{
  if (msg->data.empty())
  {
    return;
  }

  if (msg->info.size == 0)
  {
    std::stringstream ss;
    ss << "Map is zero-sized (" << msg->info.size << ")";
    setStatus(rviz::StatusProperty::Error, "DOGM", QString::fromStdString(ss.str()));
    return;
  }

  clear();

  setStatus(rviz::StatusProperty::Ok, "Message", "Map received" );

  ROS_DEBUG("Received a %d X %d map @ %.3f m/pix\n",
             msg->info.size,
             msg->info.size,
             msg->info.resolution);

  float resolution = msg->info.resolution;
  int size = msg->info.size;

  Ogre::Vector3 position(msg->info.pose.position.x,
                          msg->info.pose.position.y,
                          msg->info.pose.position.z);
                          
  Ogre::Quaternion orientation(msg->info.pose.orientation.w,
                                msg->info.pose.orientation.x,
                                msg->info.pose.orientation.y,
                                msg->info.pose.orientation.z);

  frame_ = msg->header.frame_id;
  if (frame_.empty())
  {
    frame_ = "/dogm";
  }

  // Expand it to be RGB data
  unsigned int pixels_size = size * size;
  unsigned char* pixels = new unsigned char[pixels_size * 4];
  memset(pixels, 255, pixels_size * 4);

  bool map_status_set = false;
  unsigned int num_pixels_to_copy = pixels_size;
  if( pixels_size != msg->data.size() )
  {
    std::stringstream ss;
    ss << "Data size doesn't match size * size: size = " << size
       << ", data size = " << msg->data.size();
    setStatus( rviz::StatusProperty::Error, "Map", QString::fromStdString(ss.str()));
    map_status_set = true;

    // Keep going, but don't read past the end of the data.
    if(msg->data.size() < pixels_size)
    {
      num_pixels_to_copy = msg->data.size();
    }
  }

  unsigned char* pixels_ptr = pixels;
  
  for (unsigned int pixel_index = 0; pixel_index < num_pixels_to_copy; pixel_index++)
  {
    dogm_msgs::GridCell cell = msg->data[pixel_index];
    float occ = pignisticTransformation(cell.free_mass, cell.occ_mass);

    Eigen::Vector2f vel;
    vel << cell.mean_x_vel, cell.mean_y_vel;

    Eigen::Matrix2f covar;
    covar << cell.var_x_vel, cell.covar_xy_vel,
             cell.covar_xy_vel, cell.var_y_vel;

    auto mdist = vel.transpose() * covar.inverse() * vel;

    if (occ >= occ_property_->getFloat()
        && mdist >= mahalanobis_property_->getFloat())
    {
      float angle = fmodf((atan2(cell.mean_y_vel, cell.mean_x_vel) * (180.0f / Ogre::Math::PI)) + 360, 360);

      int r, g, b;
      hsvToRGB(angle / 360.0f, 1.0f, 1.0f, r, g, b);

      *pixels_ptr++ = r; // red
      *pixels_ptr++ = g; // green
      *pixels_ptr++ = b; // blue
      *pixels_ptr++ = 255; // alpha
    }
    else
    {
      unsigned char val = 255 - static_cast<unsigned char>(occ * 255.0f);

      *pixels_ptr++ = val; // red
      *pixels_ptr++ = val; // green
      *pixels_ptr++ = val; // blue
      *pixels_ptr++ = 255; // alpha
    }
  }

  Ogre::DataStreamPtr pixel_stream;
  pixel_stream.bind(new Ogre::MemoryDataStream(pixels, pixels_size * 4));
  static int tex_count = 0;
  std::stringstream ss;
  ss << "DOGMTexture" << tex_count++;
  try
  {
    texture_ = Ogre::TextureManager::getSingleton().loadRawData(ss.str(), Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME,
                                                                 pixel_stream, size, size, Ogre::PF_BYTE_RGBA, Ogre::TEX_TYPE_2D,
                                                                 0);

    if(!map_status_set)
    {
      setStatus( rviz::StatusProperty::Ok, "DOGM", "Map OK" );
    }
  }
  catch(Ogre::RenderingAPIException&)
  {
    Ogre::Image image;
    pixel_stream->seek(0);
    float fsize = size;

    {
      std::stringstream ss;
      ss << "Map is larger than your graphics card supports.  Downsampled from [" << size << "x" << size << "] to [" << fsize << "x" << fsize << "]";
      setStatus(rviz::StatusProperty::Ok, "DOGM", QString::fromStdString(ss.str()));
    }

    ROS_WARN("Failed to create full-size map texture, likely because your graphics card does not support textures of size > 2048.  Downsampling to [%d x %d]...", (int)fsize, (int)fsize);
    image.loadRawData(pixel_stream, size, size, Ogre::PF_BYTE_RGBA);
    image.resize(fsize, fsize, Ogre::Image::FILTER_NEAREST);
    ss << "DOGMDownsampled";
    texture_ = Ogre::TextureManager::getSingleton().loadImage(ss.str(), Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME, image);
  }

  delete [] pixels;

  Ogre::Pass* pass = material_->getTechnique(0)->getPass(0);
  Ogre::TextureUnitState* tex_unit = NULL;
  if (pass->getNumTextureUnitStates() > 0)
  {
    tex_unit = pass->getTextureUnitState(0);
  }
  else
  {
    tex_unit = pass->createTextureUnitState();
  }

  tex_unit->setTextureName(texture_->getName());
  tex_unit->setTextureFiltering( Ogre::TFO_NONE );

  static int map_count = 0;
  std::stringstream ss2;
  ss2 << "DOGMObject" << map_count++;
  manual_object_ = scene_manager_->createManualObject(ss2.str());
  scene_node_->attachObject(manual_object_);

  manual_object_->begin(material_->getName(), Ogre::RenderOperation::OT_TRIANGLE_LIST);
  {
    // First triangle
    {
      // Bottom left
      manual_object_->position(0.0f, 0.0f, 0.0f);
      manual_object_->textureCoord(0.0f, 0.0f);
      manual_object_->normal(0.0f, 0.0f, 1.0f);

      // Top right
      manual_object_->position(resolution * size, resolution * size, 0.0f);
      manual_object_->textureCoord(1.0f, 1.0f);
      manual_object_->normal(0.0f, 0.0f, 1.0f);

      // Top left
      manual_object_->position(0.0f, resolution * size, 0.0f);
      manual_object_->textureCoord(0.0f, 1.0f);
      manual_object_->normal(0.0f, 0.0f, 1.0f);
    }

    // Second triangle
    {
      // Bottom left
      manual_object_->position(0.0f, 0.0f, 0.0f);
      manual_object_->textureCoord(0.0f, 0.0f);
      manual_object_->normal(0.0f, 0.0f, 1.0f);

      // Bottom right
      manual_object_->position(resolution * size, 0.0f, 0.0f);
      manual_object_->textureCoord(1.0f, 0.0f);
      manual_object_->normal(0.0f, 0.0f, 1.0f);

      // Top right
      manual_object_->position(resolution * size, resolution * size, 0.0f);
      manual_object_->textureCoord(1.0f, 1.0f);
      manual_object_->normal(0.0f, 0.0f, 1.0f);
    }
  }
  manual_object_->end();

  if (draw_under_property_->getBool())
  {
    manual_object_->setRenderQueueGroup(Ogre::RENDER_QUEUE_4);
  }

  resolution_property_->setValue(resolution);
  size_property_->setValue(size);
  position_property_->setVector(position);
  orientation_property_->setQuaternion(orientation);

  latest_map_pose_ = msg->info.pose;
  transformMap();

  loaded_ = true;

  context_->queueRender();
}

void DOGMDisplay::transformMap()
{
  Ogre::Vector3 position;
  Ogre::Quaternion orientation;
  if (!context_->getFrameManager()->transform(frame_, ros::Time(), latest_map_pose_, position, orientation))
  {
    ROS_DEBUG("Error transforming map '%s' from frame '%s' to frame '%s'",
               qPrintable(getName()), frame_.c_str(), qPrintable(fixed_frame_));

    setStatus(rviz::StatusProperty::Error, "Transform",
               "No transform from [" + QString::fromStdString( frame_ ) + "] to [" + fixed_frame_ + "]");
  }
  else
  {
    setStatus(rviz::StatusProperty::Ok, "Transform", "Transform OK");
  }

  scene_node_->setPosition(position);
  scene_node_->setOrientation(orientation);
}

void DOGMDisplay::fixedFrameChanged()
{
  transformMap();
}

}  // end namespace dogm_rviz_plugin

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(dogm_rviz_plugin::DOGMDisplay, rviz::Display)
