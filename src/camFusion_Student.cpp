
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Given a sorted vector, gives the median of the sub-vector in range [start, end)
double findMedian(std::vector<double> data, int start, int end)
{
    int size = end - start;
  	int half = size / 2;
  	double result = 0.0;
  
    if(size % 2 == 0)
    {
		result = (data[start+half] + data[start+(--half)]) / 2;
    }
    else
    {
		result = data[start+half];
    }
  
  	return result;
}

double findOutlierBound(std::vector<double> data)
{
 	double Q2 = findMedian(data, 0, data.size());
  	double Q1 = 0.0;
  	double Q3 = 0.0;
  
  	int size = data.size();
  
  	if(size % 2 == 0)
    {
      	Q1 = findMedian(data, 0, size / 2);
      	Q3 = findMedian(data, size / 2, size);
    }
  	else
    {
     	Q1 = findMedian(data, 0, size / 2);
      	Q3 = findMedian(data, 1 + (size / 2), size);
    }
  
  	double bound = Q1 - (1.5 * (Q3 - Q1));
  
  	return bound;
}

/////////////////////
/// MAIN FUNTIONS ///
/////////////////////

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
  	map<cv::DMatch, double> bbMatches;
  	double meanDist = 0.0;
  
  	for(auto match : kptMatches)
    {
      	auto kptP = kptsPrev[match.queryIdx];
      	auto kptC = kptsCurr[match.trainIdx];
      
     	if(boundingBox.roi.contains(kptC.pt)) 
        { 
          	double distance = cv::norm(kptC.pt - kptP.pt);
          
          	bbMatches[match] = distance;
          	meanDist += distance;
        }
    }
  
  	meanDist /= bbMatches.size();
  
  	for(auto matchDist : bbMatches)
    {
     	if(matchDist.second < 2.5 * meanDist)
        {
         	boundingBox.kptMatches.push_back(matchDist.first); 
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios;
  
  	if(kptMatches.size() == 0) return;
  
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { 
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0;

            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } 
    } 

    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
  	double dT = 1 / frameRate;
  
  	std::vector<double> prevX, currX;
  
    for(auto point : lidarPointsPrev)
    {
      prevX.push_back(point.x);
    }
  
    for(auto point : lidarPointsCurr)
    {
      currX.push_back(point.x);
    }
  
  	std::sort(prevX.begin(), prevX.end());
  	std::sort(currX.begin(), currX.end());
  
  	double prevBound = findOutlierBound(prevX);
  	double currBound = findOutlierBound(currX);

    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    for (auto xCoord : prevX)
    {
     	minXPrev = xCoord > prevBound && minXPrev > xCoord ? xCoord : minXPrev;
    }

    for (auto xCoord : currX)
    {
     	minXCurr = xCoord > currBound && minXCurr > xCoord ? xCoord : minXCurr;
    }

    // compute TTC from both measurements
    TTC = minXCurr * dT / (minXPrev - minXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
  	std::map<int, std::map<int, int>> bbMatches;
  
    for(size_t i = 0; i < matches.size(); i++)
    {
     	cv::KeyPoint prevKpt = prevFrame.keypoints[matches[i].queryIdx];
      	std::vector<BoundingBox> prevBoxes = prevFrame.boundingBoxes;
      	int prevBoxId = -1;
      
      	for(size_t j = 0; j < prevBoxes.size(); j++)
        {
          	if(prevBoxes[j].roi.contains(prevKpt.pt))
            {
             	prevBoxId = prevBoxes[j].boxID;
              	break;
            }
        }
      
      	cv::KeyPoint currKpt = currFrame.keypoints[matches[i].trainIdx];
        std::vector<BoundingBox> currBoxes = currFrame.boundingBoxes;
      	int currBoxId = -1;
      
      	for(size_t j = 0; j < currBoxes.size(); j++)
        {
          	if(currBoxes[j].roi.contains(currKpt.pt))
            {
             	currBoxId = currBoxes[j].boxID;
              	break;
            }
        }
      
      	bbMatches[prevBoxId][currBoxId] += 1;
    } // end of loop over keypoint matches
  
  	for(auto prevBB : bbMatches)
    {
      int matchedBoxId = -1;
     	for(auto currBB : prevBB.second)
        {
         	int assocKpts = 0;
          	if(currBB.second > assocKpts)
            {
             	assocKpts = currBB.second;
              matchedBoxId = currBB.first;
            }
        }
      
      	if(prevBB.first == -1 || matchedBoxId == -1) continue;
      	bbBestMatches[prevBB.first] = matchedBoxId;
    } // end of loop over potential BB matches
}
