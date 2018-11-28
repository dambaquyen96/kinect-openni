//
// Created by hunglv on 18/11/2018.
//

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/filter_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/surface/convex_hull.h>
#include <cmath>

#ifndef OPENNI_GRABBER_DETERMINEPOSITION_H
#define OPENNI_GRABBER_DETERMINEPOSITION_H

#endif //OPENNI_GRABBER_DETERMINEPOSITION_H

class DeterminePosition {
public:
//    pcl::PointCloud<pcl::PointXYZRGBA> pointCloud;
//    Eigen::VectorXf plane;
//    pcl::PointXYZRGBA OXYZ;
//    DeterminePosition(pcl::PointCloud<pcl::PointXYZRGBA> pointCloud, Eigen::VectorXf plane) {
//        this->pointCloud = pointCloud;
//        this->plane = plane;
//        this->OXYZ.x = 0;
//        this->OXYZ.y = 0;
//        this->OXYZ.z = 0;
//    }
    /*
    * (a*x + b*y + c*z + d)/sqrt(a*a+b*b+c*c)
    */
    double DistanceFromPointToPlance(Eigen::VectorXf OXYZ, pcl::PointXYZRGBA point) {
        double a = OXYZ[0], b = OXYZ[1] ,c = OXYZ[2], d = OXYZ[3];
        double x = point.x, y = point.y, z = point.z;
        double distance;
        distance = abs((a*x + b*y + c*z + d)/sqrt(a*a+b*b+c*c));
        return distance;
    }

    double DistanceFromPointToPoint(pcl::PointXYZRGBA point1, pcl::PointXYZRGBA point2) {
        double x = point1.x - point2.x, y = point1.y - point2.y, z = point1.z - point2.z;
        return sqrt(x*x+y*y+z*z);
    }

    double DistanceHeight(double pointToPoint, double point1ToPlane, double point2ToPlane) {
        return sqrt(pointToPoint*pointToPoint - (point1ToPlane-point2ToPlane)*(point1ToPlane-point2ToPlane));
    }
};