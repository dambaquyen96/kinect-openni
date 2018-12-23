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

    void normalize(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                   pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud_out,
                   pcl::PointIndices::Ptr &indices_origin){
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud_out, indices);
        indices_origin->indices = indices;
    }

    void downsample(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud_out){
        pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.01f, 0.01f, 0.01f);
        sor.filter(*cloud_out);
    }

    void getTable(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                  Eigen::VectorXf &coefficients,
                  pcl::ModelCoefficients::Ptr &table_coefficients,
                  pcl::PointIndices::Ptr &table_indices){
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_down(new pcl::PointCloud<pcl::PointXYZRGBA>);
        downsample(cloud, cloud_down);

        std::vector<int> inliers;
        pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA>::Ptr
                model(new pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA> (cloud_down));
        pcl::RandomSampleConsensus<pcl::PointXYZRGBA> ransac (model);
        ransac.setDistanceThreshold (.01);
        ransac.computeModel();
        ransac.getInliers(inliers);
        ransac.getModelCoefficients(coefficients);
        pcl::PointIndices::Ptr indices(new pcl::PointIndices);
        std::vector<float> coeff;
        for (int i=0; i<coefficients.size(); i++) {
            coeff.push_back(coefficients[i]);
        }
        table_coefficients->values = coeff;

        std::vector<int> tmp;
        pcl::SampleConsensusModelPlane<pcl::PointXYZRGBA> modelPlane (cloud);
        modelPlane.selectWithinDistance(coefficients, .01, tmp);
        table_indices->indices = tmp;
    }

    void removeTable(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                     pcl::ModelCoefficients::Ptr table_coefficients,
                     pcl::PointIndices::Ptr table_indices,
                     pcl::PointIndices::Ptr &object_indices){
        std::vector<int> indices_fullset (cloud->size ());
        for (int p_it = 0; p_it < static_cast<int> (indices_fullset.size ()); ++p_it)
            indices_fullset[p_it] = p_it;
        std::sort (table_indices->indices.begin (), table_indices->indices.end ());
        std::set_difference (indices_fullset.begin (), indices_fullset.end (),
                             table_indices->indices.begin (), table_indices->indices.end (),
                             std::inserter (object_indices->indices, object_indices->indices.begin ()));
    }

    void filterObject(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                     pcl::ModelCoefficients::Ptr table_coefficients,
                     pcl::PointIndices::Ptr table_inliers,
                     pcl::PointIndices::Ptr &object_indices){
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
        pcl::PointIndices indices; // Points lying over the table
        pcl::ExtractPolygonalPrismData<pcl::PointXYZRGBA> prism;
        prism.setHeightLimits(0.01,0.5); // object must lie between 1cm and 50cm
        // over the plane.
        prism.setInputCloud(cloud);
        prism.setInputPlanarHull(table_hull.makeShared());
        prism.segment(indices);
        object_indices->indices = indices.indices;
    }

    void clusterObject(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                       pcl::PointIndices::Ptr object_indices,
                       std::vector<pcl::PointIndices> &cluster_indices){
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
        ec.setClusterTolerance(0.02); // 2cm
        ec.setMinClusterSize(1000);
        ec.setInputCloud(cloud);
        ec.setIndices(object_indices);
        ec.extract(cluster_indices);
    }

    void processData(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                     pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud_out,
                     Eigen::VectorXf coefficients,
                     pcl::PointIndices::Ptr indices_origin,
                     std::vector<pcl::PointIndices> cluster_indices,
                     cv::Mat frame_rgb){
        if (cluster_indices.size() == 0) return;
        pcl::copyPointCloud(*cloud, *cloud_out);
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
                int k = indices_origin->indices[cluster_indices[j].indices[i]];
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

    void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud) {
        double st = pcl::getTime ();

        // Convert cloud 2 rgb frame
        cv::Mat frame_rgb(cv::Size(640,480), CV_8UC3);
        cloud2Frame(cloud, frame_rgb);

        // Normalize data
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_normalize(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PointIndices::Ptr indices_origin (new pcl::PointIndices);
        normalize(cloud, cloud_normalize, indices_origin);

        // Get floor plane
        Eigen::VectorXf coefficients;
        pcl::ModelCoefficients::Ptr table_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr table_indices (new pcl::PointIndices);
        getTable(cloud_normalize, coefficients, table_coefficients, table_indices);

        // Filter object
        pcl::PointIndices::Ptr everything_but_the_plane (new pcl::PointIndices);
        filterObject(cloud_normalize, table_coefficients, table_indices, everything_but_the_plane);

        // Cluster object
        std::vector<pcl::PointIndices> cluster_indices;
        clusterObject(cloud_normalize, everything_but_the_plane, cluster_indices);

        // Final process
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBA>);
        processData(cloud_normalize, cloud_out, coefficients, indices_origin, cluster_indices, frame_rgb);

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
