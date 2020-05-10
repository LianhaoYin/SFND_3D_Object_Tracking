#include <numeric>
#include "matching2D.hpp"
#include <map>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
        
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "FLANN matching";
    }
    
    
    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        
        /*if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }*/
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
 
        descSource.convertTo(descSource, CV_32FC1);
        descRef.convertTo(descRef, CV_32FC1);

        cout << "szie of descsource" << descSource.size << "szie of descRef" << descRef.size <<endl;
        cout << "szie of keypoint descsource" << kPtsSource.size() << "szie of keypoint descRef" << kPtsRef.size() <<endl;

        if(kPtsSource.size()>=2 && kPtsRef.size()>=2)
        {
            try {
                matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
            } catch (int e) {
                cout<< "exception in knnMatch..........."<< endl;
            }
            
        }
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {

            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
            
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
        // ...
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor BRIEF, ORB, FREAK, AKAZE, SIFT
    // perform feature description
    double t = (double)cv::getTickCount();
    
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
        //cv::Mat descBRISK;
        extractor->compute(img, keypoints, descriptors);
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
        //cv::Mat descORB;
        extractor->compute(img, keypoints, descriptors);
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
        //cv::Mat descFREAK;
        extractor->compute(img, keypoints, descriptors);
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
        //cv::Mat descAKAZE;
        extractor->compute(img, keypoints, descriptors);
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
        //cv::Mat descSIFT;
        extractor->compute(img, keypoints, descriptors);
    }
    else
    {
        cout << "No section of descriptor" << endl;
    }
    

    //extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

//Detect keypoints in image by using one of the detector

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
    
    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;
    
    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);
    
    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {
        
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);
    
    // visualize results
    //string windowName = "Harris Corner Detector Response Matrix";
    //cv::namedWindow(windowName, 4);
    //cv::imshow(windowName, dst_norm_scaled);
    //cv::waitKey(0);
    
    // TODO: Your task is to locate local maxima in the Harris response matrix
    // and perform a non-maximum suppression (NMS) in a local neighborhood around
    // each maximum. The resulting coordinates shall be stored in a list of keypoints
    // of the type `vector<cv::KeyPoint>`.
    
    
    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t j = 0; j < dst_norm.rows; j++)
    {
        for (size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = (int)dst_norm.at<float>(j, i);
            if (response > minResponse)
            { // only store points above a threshold
                
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;
                
                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKeyPoint.response > (*it).response)
                        {                      // if overlap is >t AND response is higher for new kpt
                            *it = newKeyPoint; // replace old key point with new one
                            break;             // quit loop over keypoints
                        }
                    }
                }
                if (!bOverlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
                }
            }
        } // eof loop over cols
    }     // eof loop over rows
    
    std::cout<<"keypoints size "<<keypoints.size()<<endl;
    if (bVis)
    {
        cv::Mat visImage_clone = img.clone();
        cv::drawKeypoints(img, keypoints, visImage_clone, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage_clone);
        cv::waitKey(0);
    }
}

void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &imgGray, bool bVis)
{
    
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(imgGray, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    cv::Mat descBRISK;
    t = (double)cv::getTickCount();
    descriptor->compute(imgGray, keypoints, descBRISK);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = imgGray.clone();
        cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
}
void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Mat descORB;
    t = (double)cv::getTickCount();
    descriptor->compute(img, keypoints, descORB);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB descriptor in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
}

void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::FastFeatureDetector::create();
    cv::Mat descFAST;
    t = (double)cv::getTickCount();
    descriptor->compute(img, keypoints, descFAST);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST descriptor in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
}

void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::AKAZE::create();
    cv::Mat descAKAZE;
    t = (double)cv::getTickCount();
    descriptor->compute(img, keypoints, descAKAZE);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE descriptor in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Results";
        cv::namedWindow(windowName, 1);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &imgGray, bool bVis)
{
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
    
    double t = (double)cv::getTickCount();
    detector->detect(imgGray, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    cv::Mat descSIFT;
    t = (double)cv::getTickCount();
    descriptor->compute(imgGray, keypoints, descSIFT);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT descriptor in " << 1000 * t / 1.0 << " ms" << endl;
    
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = imgGray.clone();
        cv::drawKeypoints(imgGray, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    typedef std::map<std::string, int> detector_struc;
    detector_struc det_str;
    det_str["SHITOMASI"] = 1;
    det_str["HARRIS"] = 2;
    det_str["ORB"] = 3;
    det_str["AKAZE"] = 4;
    det_str["SIFT"] = 5;
    det_str["BRISK"] = 6;
    det_str["FAST"] = 7;
    //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    
    cout << "slected switch" << det_str[detectorType] << endl;
    switch(det_str[detectorType])
    {
        case 1:
            cout << "slected ShiTomasi"<< endl;
            detKeypointsShiTomasi(keypoints, img,bVis);
            break;
        case 2:
            cout << "slected Harris"<< endl;
            detKeypointsHarris(keypoints, img,bVis);
            break;
        case 3:
            cout << "slected ORB"<< endl;
            detKeypointsORB(keypoints, img,bVis);
            break;
        case 4:
            cout << "slected AKAZE"<< endl;
            detKeypointsAKAZE(keypoints, img,bVis);
            break;
        case 5:
            cout << "slected SIFT"<< endl;
            detKeypointsSIFT(keypoints, img,bVis);
            break;
        case 6:
            cout << "slected BRISK"<< endl;
            detKeypointsBRISK(keypoints, img,bVis);
            break;
        case 7:
            cout << "slected BRISK"<< endl;
            // detKeypointsFAST(keypoints, img,bVis);
            detKeypointsBRISK(keypoints, img,bVis);
            break;
            
        default:
            break;
            //break;
    }
}


