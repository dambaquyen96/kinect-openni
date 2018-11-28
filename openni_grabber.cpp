#include <pcl/io/openni_grabber.h>
#include <pcl/visualization/cloud_viewer.h>
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
//        // Create a shared plane model pointer directly
//        pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGBA, pcl::Normal>::Ptr
//         model (new pcl::SampleConsensusModelNormalPlane<pcl::PointXYZRGBA, pcl::Normal> (cloud));
//        // Set normals
//        model->setInputNormals(normals_out);
////        model.setInputNormals(normals_out);
//        // Set the normal angular distance weight.
//        model->setNormalDistanceWeight(0.5f);
////        model.setNormalDistanceWeight(0.5f);
//        // Create the RANSAC object
//        pcl::RandomSampleConsensus<pcl::PointXYZRGBA> sac (model, 0.03);
//        // perform the segmenation step
//        bool result = sac.computeModel ();
    }

    void Extract_Object(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                       pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud_objects) {
        // Input
//        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_objects; // points belonging to objects
        // Output vector of objects, one point cloud per object
        std::vector<pcl::PointCloud<pcl::PointXYZRGBA>::Ptr> objects;
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> cluster;
        cluster.setInputCloud(cloud_objects);
        std::vector<pcl::PointIndices> object_clusters;
        cluster.extract(object_clusters);
        pcl::ExtractIndices<pcl::PointXYZRGBA> extract_object_indices;
//        std::cout << "size of object: " << object_clusters.size() << std::endl;

        int color_1[10] = {255,0,0,255,0,255};
        int color_2[10] = {0,255,0,255,255,0};
        int color_3[10] = {0,0,255,0,255,255};
        for(int i=0; i < object_clusters.size(); ++i) {
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
            extract_object_indices.setInputCloud(cloud);
            extract_object_indices.setIndices(
                    boost::make_shared<const pcl::PointIndices>(object_clusters[i]));
            extract_object_indices.filter(*object_cloud);
//            std::cerr << object_cloud->points[0] << std::endl;
//            objects.push_back(object_cloud);
            cloud_objects->points.resize (cloud->size ());
//            std::cout << "size of cluster_indices: " << cluster_indices[j].indices.size () << std::endl;
//                    cloud_out->header   = cloud->header;
//                    cloud_out->width    = cloud->width;
//                    cloud_out->height   = 1;
//                    cloud_out->is_dense = cloud->is_dense;
//                    cloud_out->sensor_orientation_ = cloud->sensor_orientation_;
//                    cloud_out->sensor_origin_ = cloud->sensor_origin_;
            for (int j = 0; j < object_clusters[i].indices.size (); ++j) {
                cloud_objects->points[j] = cloud->points[object_clusters[i].indices[j]];
                cloud_objects->points[j].r = color_1[i%6];
                cloud_objects->points[j].g = color_2[i%6];
                cloud_objects->points[j].b = color_3[i%6];
            }
        }
//        std::cout << "size of object: " << objects[0]->size() << std::endl;
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

    void applyRANSAC2(const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr cloud,
                      pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out, bool negative){
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
        indices->indices = inliers;
        pcl::ModelCoefficients::Ptr table_coefficients(new pcl::ModelCoefficients);
        std::vector<float> coeff;
        for (int i=0; i<coefficients.size(); i++) {
            coeff.push_back(coefficients[i]);
        }
        table_coefficients->values = coeff;
        removeTable(cloud, indices, table_coefficients, *cloud_out);
        Extract_Object(cloud, cloud_out);

        if (negative) {
//             Remove the plane indices from the data
            pcl::PointIndices::Ptr everything_but_the_plane (new pcl::PointIndices);
            std::vector<int> indices_fullset (cloud->size ());
            for (int p_it = 0; p_it < static_cast<int> (indices_fullset.size ()); ++p_it)
                indices_fullset[p_it] = p_it;

            std::sort (inliers.begin (), inliers.end ());
            set_difference (indices_fullset.begin (), indices_fullset.end (),
                            inliers.begin (), inliers.end (),
                            inserter (everything_but_the_plane->indices, everything_but_the_plane->indices.begin ()));

//             Extract largest cluster minus the plane
            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZRGBA> ec;
            ec.setClusterTolerance (0.02); // 2cm
            ec.setMinClusterSize (100);
            ec.setInputCloud (cloud);
            ec.setIndices (everything_but_the_plane);
            ec.extract (cluster_indices);

            // Convert data back
            if (cluster_indices.size() > 0) {
                int color_1[10] = {255,0,0,255,0,255};
                int color_2[10] = {0,255,0,255,255,0};
                int color_3[10] = {0,0,255,0,255,255};
//                std::cout << "size of cluster_indices: " << cluster_indices.size() << std::endl;
                int position = 0;
                for (int j=0; j<cluster_indices.size(); j++) {
                    cloud_out->points.resize (cloud->size ());
//                    std::cout << "size of cluster_indices: " << cluster_indices[j].indices.size () << std::endl;
//                    cloud_out->header   = cloud->header;
//                    cloud_out->width    = cloud->width;
//                    cloud_out->height   = 1;
//                    cloud_out->is_dense = cloud->is_dense;
//                    cloud_out->sensor_orientation_ = cloud->sensor_orientation_;
//                    cloud_out->sensor_origin_ = cloud->sensor_origin_;
                    for (int i = 0; i < cluster_indices[j].indices.size (); ++i) {
                        cloud_out->points[i+position] = cloud->points[cluster_indices[j].indices[i]];
                        cloud_out->points[i+position].r = color_1[j%6];
                        cloud_out->points[i+position].g = color_2[j%6];
                        cloud_out->points[i+position].b = color_3[j%6];
                    }
                    position += cluster_indices[j].indices.size ();
                }
//                pcl::copyPointCloud(*cloud, cluster_indices[1].indices, *cloud_out);
            }
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
	        double now = pcl::getTime ();
	        std::cout << "distance of center pixel :" << cloud->points [(cloud->width >> 1) * (cloud->height + 1)].z << " mm. Average framerate: " << double(count)/double(now - last) << " Hz" <<  std::endl;
	        count = 0;
	        last = now;
	    }

	    // Down sampling
        pcl::VoxelGrid<pcl::PointXYZRGBA> sor;
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBA>);
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.01f, 0.01f, 0.01f);
        sor.filter(*cloud_out);
//        cloud_out->width = 640;
//        cloud_out->height = 480;

        // Filter
//        pcl::FastBilateralFilter<pcl::PointXYZRGBA> filter;
//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBA>);
//        filter.setInputCloud (cloud_out);
//        filter.setSigmaS (5);
//        filter.setSigmaR (0.005f);
//        filter.filter(*cloud_out);

//        pcl::PointCloud<pcl::Normal>::Ptr normals_out(new pcl::PointCloud<pcl::Normal>);
//        applyRANSAC(cloud, normals_out);

//        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZRGBA>);
        applyRANSAC2(cloud_out, cloud_out, true);

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
