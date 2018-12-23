#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_representation.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/time.h>
#include <pcl/common/common.h>
#include <pcl/common/angles.h>
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
#include <pcl/search/organized.h>
#include <pcl/search/impl/kdtree.hpp>
#include <pcl/search/impl/organized.hpp>
#include <pcl/surface/convex_hull.h>
#include <cmath>
#include <math.h>
#include "DeterminePosition.h"
#include "Location.h"

// init library of OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>

class SimpleOpenNIViewer {
public:
    SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") {}

    void applyRANSAC(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                     pcl::PointCloud<pcl::Normal>::Ptr &normals_out) {
        pcl::IntegralImageNormalEstimation<pcl::PointXYZRGBA, pcl::Normal> norm_est;
        norm_est.setNormalEstimationMethod (norm_est.AVERAGE_DEPTH_CHANGE);
        norm_est.setMaxDepthChangeFactor(0.02f);
        norm_est.setNormalSmoothingSize(10.0f);
        norm_est.setInputCloud (cloud);
        norm_est.compute (*normals_out);
    }

    void removeTable(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                     pcl::PointIndices::Ptr table_inliers,
                     pcl::ModelCoefficients::Ptr table_coefficients,
                     pcl::PointCloud<pcl::PointXYZRGBA> &cloud_out){
        // Project the table inliers to the estimated plane
        pcl::PointCloud<pcl::PointXYZRGBA> table_projected;
        pcl::ProjectInliers<pcl::PointXYZRGBA> proj;
        proj.setInputCloud(cloud);
        proj.setIndices(table_inliers);
        proj.setModelCoefficients(table_coefficients);
        proj.filter(table_projected);
        // Estimate the convex hull of the projected points
        pcl::PointCloud<pcl::PointXYZRGBA> table_hull;
        pcl::ConvexHull<pcl::PointXYZRGBA> hull;
        hull.setInputCloud(table_projected.makeShared());
        hull.reconstruct(table_hull);
        // Determine the points lying in the prism
        pcl::PointIndices object_indices; // Points lying over the table
        pcl::ExtractPolygonalPrismData<pcl::PointXYZRGBA> prism;
        prism.setHeightLimits(0.01,0.5); // object must lie between 1cm and 50cm
        // over the plane.
        prism.setInputCloud(cloud);
        prism.setInputPlanarHull(table_hull.makeShared());
        prism.segment(object_indices);
        // Extract the point cloud corresponding to the extracted indices.
        pcl::ExtractIndices<pcl::PointXYZRGBA> extract_object_indices;
        extract_object_indices.setInputCloud(cloud);
        extract_object_indices.setIndices(
                boost::make_shared<const pcl::PointIndices>(object_indices));
        extract_object_indices.filter(cloud_out);
    }
    void applyRANSAC2(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud_ori,
                      const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, cv::Mat frame_rgb,bool negative){
        double st_1 = pcl::getTime ();
        std::vector<int> inliers;
        pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA>::Ptr
                model(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA> (cloud));
        pcl::RandomSampleConsensus<pcl::PointXYZRGBA> ransac (model);
        ransac.setDistanceThreshold (.01);
        ransac.computeModel();
        ransac.getInliers(inliers);

        Eigen::VectorXf coefficients;
        ransac.getModelCoefficients(coefficients);

        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
//        indices->indices = inliers;
        pcl::ModelCoefficients::Ptr table_coefficients(new pcl::ModelCoefficients);
        std::vector<float> coeff;
        for (int i=0; i<coefficients.size(); i++) {
            coeff.push_back(coefficients[i]);
        }
        std::vector<int> inliers_origin;
        table_coefficients->values = coeff;
//        pcl::copyPointCloud(*cloud_origin, *cloud_out);

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::copyPointCloud(*cloud_ori, *cloud_origin);

        std::vector<int> indices_ori;
        pcl::removeNaNFromPointCloud(*cloud_origin, *cloud_origin, indices_ori);
        std::cout << cloud_ori->size() << " " << cloud_origin->size() << " " << indices_ori.size() << std::endl;

        pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA> modelPlane (cloud_origin);
        modelPlane.selectWithinDistance(coefficients, .01, inliers_origin);
        indices->indices = inliers_origin;
        pcl::copyPointCloud(*cloud_origin, *cloud_out);
//        for (int i=0; i<inliers_origin.size(); i++) {
//            cloud_out->points[inliers_origin[i]].r = 255;
//            cloud_out->points[inliers_origin[i]].b = 0;
//            cloud_out->points[inliers_origin[i]].g = 165;
//        }

//        double st_1 = pcl::getTime ();
//        removeTable(cloud, indices, table_coefficients, *cloud_out);
//        std::cout << "fps of remove plane = " << 1.0/(pcl::getTime() - st_1) <<std::endl;
//        negative = false;
        if (negative) {
            double st = pcl::getTime ();
//             Remove the plane indices from the data
            pcl::PointIndices::Ptr everything_but_the_plane (new pcl::PointIndices);
            std::vector<int> indices_fullset (cloud_origin->size ());
            for (int p_it = 0; p_it < static_cast<int> (indices_fullset.size ()); ++p_it)
                indices_fullset[p_it] = p_it;
            std::sort (inliers_origin.begin (), inliers_origin.end ());
            std::set_difference (indices_fullset.begin (), indices_fullset.end (),
                            inliers_origin.begin (), inliers_origin.end (),
                            std::inserter (everything_but_the_plane->indices, everything_but_the_plane->indices.begin ()));
//             Extract largest cluster minus the plane
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
            ec.setClusterTolerance(0.02); // 2cm
            ec.setMinClusterSize(1000);
            ec.setInputCloud(cloud_origin);
            ec.setIndices(everything_but_the_plane);
            ec.extract(cluster_indices);


            // Convert data back
            if (cluster_indices.size() > 0) {
                int color_1[10] = {255,0,0,255,0,255};
                int color_2[10] = {0,255,0,255,255,0};
                int color_3[10] = {0,0,255,0,255,255};
                int position = 0;
                double distance = 11.0;
                pcl::PointXYZRGBA OXYZ;
                OXYZ.x = 0;
                OXYZ.y = 0;
                OXYZ.z = 0;
                DeterminePosition dp;
                for (int j=0; j<cluster_indices.size(); j++) {
                    int max_x = -100, min_x = 10000, max_y = -100, min_y = 10000;
                    for (int i = 0; i < cluster_indices[j].indices.size (); ++i) {
                        cloud_out->points[cluster_indices[j].indices[i]].r = color_1[j%6];
                        cloud_out->points[cluster_indices[j].indices[i]].g = color_2[j%6];
                        cloud_out->points[cluster_indices[j].indices[i]].b = color_3[j%6];
                        int k_ = cluster_indices[j].indices[i];
                        int x = k/640;
                        int y = k%640;
                        if(max_x < x)
                            max_x = x;
                        if(min_x > x)
                            min_x = x;
                        if(max_y < y)
                            max_y = y;
                        if(min_y > y)
                            min_y = y;
//                        double pointTopoint = dp.DistanceFromPointToPoint(OXYZ, cloud_out->points[cluster_indices[j].indices[i]]);
//                        double point1ToPlane = dp.DistanceFromPointToPlance(coefficients, OXYZ);
//                        double point2ToPlane = dp.DistanceFromPointToPlance(coefficients,cloud_out->points[cluster_indices[j].indices[i]]);
//                        double dis = dp.DistanceHeight(pointTopoint, point1ToPlane, point2ToPlane);
//                        if (distance > dis) distance = dis;
                    }
//                    if (j>0) {
////                        std::cout << max_x << " " << min_x << " " << max_y << " " << min_y << "        ";
//                        int w = max_x - min_x;
//                        int h = max_y - min_y;
//                        int x = min_x;
//                        int y = max_y;
//                        x = std::max(x,0);
//                        y = std::max(y,0);
//                        // Crop image
//                        cv::Rect roi;
//                        roi.x = int(x);
//                        roi.y = int(y);
//                        roi.width = int(w);
//                        roi.height = int(h);
////                        std::cout << "x = " << roi.x << "y = " << roi.y << "w = " << roi.width << " h = " << roi.height << std::endl;
////                        cv::Mat image_crop = frame_rgb(roi);
//                    }
                }
//                std::cout << std::endl;
//                std::cout << "size of object = " << cluster_indices.size() << " " << "distance = " << distance << std::endl;
            }

//            std::cout << "fps of labeling = " << 1.0/(pcl::getTime() - st) <<std::endl;
        }
        else {
            // Convert data back
//            pcl::copyPointCloud (*cloud, inliers, *cloud_out);
        }
    }

    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud) {
        static unsigned count = 0;
        static double last = pcl::getTime ();
        if (++count == 1) {
            double now = pcl::getTime();
//            std::cout << "so la: " << (cloud->width >> 1) * (cloud->height + 1) << std::endl;
//            std::cout << "distance of center pixel :" << cloud->points [(cloud->width >> 1) * (cloud->height + 1)].z << " mm. Average framerate: " << double(count)/double(now - last) << " Hz" <<  std::endl;
            count = 0;
            last = now;
        }

        cv::Mat frame_rgb(cv::Size(640,480), CV_8UC3);
        for (int i=0; i<frame_rgb.rows; i++) {
            for (int j=0; j<frame_rgb.cols; j++) {
                cv::Vec3b rgb;
                rgb[0] = cloud->points[j+i*640].r;
                rgb[1] = cloud->points[j+i*640].g;
                rgb[2] = cloud->points[j+i*640].b;
                frame_rgb.at<cv::Vec3b>(i,j) = rgb;
            }
        }

        double st2 = pcl::getTime ();
        // Down sampling
        pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBA>);
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.01f, 0.01f, 0.01f);
        sor.filter(*cloud_out);
        applyRANSAC2(cloud, cloud_out, cloud_out, frame_rgb, true);
        std::cout << "fps of VoxelGrid = " << 1.0/(pcl::getTime() - st2) <<std::endl;
        if (!viewer.wasStopped())
            viewer.showCloud(cloud_out);
    }

    void run () {
        pcl::Grabber* interface = new pcl::OpenNIGrabber();
        boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
                boost::bind (&SimpleOpenNIViewer::cloud_cb_, this, _1);

        interface->registerCallback (f);
        interface->start ();
        while (!viewer.wasStopped()) {
            boost::this_thread::sleep (boost::posix_time::seconds (1));
        }
        interface->stop ();
    }
    pcl::visualization::CloudViewer viewer;
};

int main () {
    SimpleOpenNIViewer v;
    v.run ();
    return 0;
}
