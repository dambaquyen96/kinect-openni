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
#include <hiredis/hiredis.h>
#include <sstream>

// init library of OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>

class SimpleOpenNIViewer {
public:
    SimpleOpenNIViewer () : viewer ("PCL OpenNI Viewer") {}

    void cloud2Frame(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                     cv::Mat &frame_rgb){
        for (int i=0; i<frame_rgb.rows; i++) {
            for (int j=0; j<frame_rgb.cols; j++) {
                cv::Vec3b rgb;
                rgb[0] = cloud->points[j+i*640].r;
                rgb[1] = cloud->points[j+i*640].g;
                rgb[2] = cloud->points[j+i*640].b;
                frame_rgb.at<cv::Vec3b>(i,j) = rgb;
            }
        }
    }

    void downsample(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud_out){
        pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.01f, 0.01f, 0.01f);
        sor.filter(*cloud_out);
    }

    void removeTable(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                     pcl::PointIndices::Ptr table_inliers,
                     pcl::ModelCoefficients::Ptr table_coefficients,
                     pcl::PointCloud<pcl::PointXYZRGBA> &cloud_out) {
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
        prism.setHeightLimits(0.01, 0.5); // object must lie between 1cm and 50cm
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
                      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, cv::Mat frame_rgb){
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
        pcl::ModelCoefficients::Ptr table_coefficients(new pcl::ModelCoefficients);
        std::vector<float> coeff;
        for (int i=0; i<coefficients.size(); i++) {
            coeff.push_back(coefficients[i]);
        }
        std::vector<int> inliers_origin;
        table_coefficients->values = coeff;

        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::copyPointCloud(*cloud_ori, *cloud_origin);

        std::vector<int> indices_ori;
        pcl::removeNaNFromPointCloud(*cloud_origin, *cloud_origin, indices_ori);

        pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA> modelPlane (cloud_origin);
        modelPlane.selectWithinDistance(coefficients, .01, inliers_origin);
        indices->indices = inliers_origin;
        pcl::copyPointCloud(*cloud_origin, *cloud_out);

//        for (int i=0; i<inliers_origin.size(); i++) {
//            cloud_out->points[inliers_origin[i]].r = 255;
//            cloud_out->points[inliers_origin[i]].b = 0;
//            cloud_out->points[inliers_origin[i]].g = 165;
//        }

        bool negative = true;
        if (negative) {
            double st = pcl::getTime ();
            // Remove the plane indices from the data
            pcl::PointIndices::Ptr everything_but_the_plane (new pcl::PointIndices);
            std::vector<int> indices_fullset (cloud_origin->size ());
            for (int p_it = 0; p_it < static_cast<int> (indices_fullset.size ()); ++p_it)
                indices_fullset[p_it] = p_it;
            std::sort (inliers_origin.begin (), inliers_origin.end ());
            std::set_difference (indices_fullset.begin (), indices_fullset.end (),
                            inliers_origin.begin (), inliers_origin.end (),
                            std::inserter (everything_but_the_plane->indices, everything_but_the_plane->indices.begin ()));
            // Extract largest cluster minus the plane
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
                double distance = 1000.0;
                pcl::PointXYZRGBA OXYZ;
                OXYZ.x = 0;
                OXYZ.y = 0;
                OXYZ.z = 0;
                DeterminePosition dp;
                std::stringstream buffer_toado;
                for (int j=0; j<cluster_indices.size(); j++) {
                    int max_x = -100000, min_x = 100000, max_y = -100000, min_y = 100000, max_z = -100000, min_z = 100000;
                    for (int i = 0; i < cluster_indices[j].indices.size(); ++i) {
                        cloud_out->points[cluster_indices[j].indices[i]].r = color_1[j % 6];
                        cloud_out->points[cluster_indices[j].indices[i]].g = color_2[j % 6];
                        cloud_out->points[cluster_indices[j].indices[i]].b = color_3[j % 6];
                        int k = indices_ori[cluster_indices[j].indices[i]];
                        int y = k / 640;
                        int x = k % 640;
                        if (max_x < x)
                            max_x = x;
                        if (min_x > x)
                            min_x = x;
                        if (max_y < y)
                            max_y = y;
                        if (min_y > y)
                            min_y = y;
                        double pointTopoint = dp.DistanceFromPointToPoint(OXYZ, cloud_out->points[cluster_indices[j].indices[i]]);
                        double point1ToPlane = dp.DistanceFromPointToPlance(coefficients, OXYZ);
                        double point2ToPlane = dp.DistanceFromPointToPlance(coefficients,cloud_out->points[cluster_indices[j].indices[i]]);
                        double dis = dp.DistanceHeight(pointTopoint, point1ToPlane, point2ToPlane);
                        if (distance > dis) distance = dis;
                    }
                    buffer_toado << min_x << " " << max_x << " " << min_y << " " << max_y << " " << distance << " ";
                }
                std::vector<uchar> buf;
                cv::imencode(".jpg",frame_rgb,buf);
                reply = (redisReply *)redisCommand(c,"SET %b %b", "image", (size_t) 5, (const char*)buf.data(), (size_t) buf.size());
                freeReplyObject(reply);
                reply = (redisReply *)redisCommand(c,"SET %s %s", "toado", buffer_toado.str().c_str());
                freeReplyObject(reply);
            }
        }
    }

    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud) {
        double st = pcl::getTime ();

        // Convert cloud 2 rgb frame
        cv::Mat frame_rgb(cv::Size(640,480), CV_8UC3);
        cloud2Frame(cloud, frame_rgb);

        // Down sampling
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBA>);
        downsample(cloud, cloud_out);

        applyRANSAC2(cloud, cloud_out, cloud_out, frame_rgb);

        std::cout << "fps of VoxelGrid = " << 1.0/(pcl::getTime() - st) <<std::endl;
        if (!viewer.wasStopped())
            viewer.showCloud(cloud_out);
    }

    void run () {
        struct timeval timeout = { 1, 500000 };
        c = redisConnectWithTimeout("127.0.0.1", 6379, timeout);
        reply = (redisReply *)redisCommand(c,"PING");
        printf("PING: %s\n", reply->str);
        freeReplyObject(reply);

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
    redisContext *c;
    redisReply *reply;
};

int main () {
    SimpleOpenNIViewer v;
    v.run ();
    return 0;
}
