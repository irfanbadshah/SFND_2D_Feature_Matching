/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include <numeric>

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{

    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 9;   // last file index to load
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results

  	vector<double> allDetectorAvgTime;
  	vector<int> allDetectorAvgTotal;
    vector<int> allDetectorAvgCar;
  	vector<float> allDetectorStdX;
    vector<float> allDetectorStdY;
    vector<double> allDescriptorAvgTime;
    vector<int> allMatchingAvgCar;
   
    vector<double> eachDetectionTime;
    vector<int> numTotalDetection;
    vector<int> numCarDetection;
    vector<float> stdXDetection;
    vector<float> stdYDetection;
  	vector<double> eachDescriptorTime;
    vector<int> numCarMatching;
  
  	float stdX = 0.0;
  	float stdY = 0.0;
   
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
   		 /* MAIN LOOP OVER ALL IMAGES */
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        //dataBuffer.push_back(frame);
      
      	if(dataBuffer.size() < dataBufferSize)
          dataBuffer.push_back(frame);
      	else 
        {
            dataBuffer.erase (dataBuffer.begin());

            //std::cout<<"size"<<dataBuffer.size()<<std::endl;
            dataBuffer.push_back(frame);
        }
		std::cout<<"size"<<dataBuffer.size()<<std::endl;
        //// EOF STUDENT ASSIGNMENT
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT IMAGE KEYPOINTS        */

 // extract 2D keypoints from current image
      vector<cv::KeyPoint> keypoints; // create empty feature list for current image
       string detectorType = "AKAZE";

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
        //// -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
      
      	double timeDetect = 0.0;

        if (detectorType.compare("SHITOMASI") == 0)
        {
           timeDetect = detKeypointsShiTomasi(keypoints, imgGray, false);
        }
      	else if (detectorType.compare("HARRIS") == 0)
        {
          timeDetect =  detKeypointsHarris(keypoints, imgGray, false);
        }
        else if (detectorType.compare("FAST") == 0)
        {
           timeDetect =  detKeypointsFast(keypoints, imgGray, false);
        }
        else if (detectorType.compare("BRISK") == 0)
        {
           timeDetect = detKeypointsBrisk(keypoints, imgGray, false);
        }
        else if (detectorType.compare("SIFT") == 0)
        {
          timeDetect =  detKeypointsSift(keypoints, imgGray, false);
            
        }
      	else if (detectorType.compare("ORB") == 0)
        {
          timeDetect =  detKeypointsOrb(keypoints, imgGray, false);
        }
      	else if (detectorType.compare("AKAZE") == 0)
        {
          timeDetect =  detKeypointsAkaze(keypoints, imgGray, false);
        }
        else
        {
            //...
        }
      	 
      
      	eachDetectionTime.push_back(timeDetect);
       	numTotalDetection.push_back(keypoints.size());
        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
      	
        if (bFocusOnVehicle)
        {
          std::cout<<" Size before box = "<<keypoints.size()<<std::endl;
          std::vector<cv::KeyPoint> boxPoints;
           for (auto point : keypoints)
           {
              if (vehicleRect.contains(cv::Point2f(point.pt))) 
              { 
                boxPoints.push_back(point); 
              }
            }

            keypoints = boxPoints;
			numCarDetection.push_back(keypoints.size());          
   
          std::cout<<" Size after box = "<<keypoints.size()<<std::endl;
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;
	
        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
        //// -> BRIEF, ORB, FREAK, AKAZE, SIFT

        cv::Mat descriptors;
        string descriptorType = "SIFT"; // BRISK, ORB, FREAK, AKAZE, SIFT
      	
      	double timeDes;
        timeDes = descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);
      	 
      	eachDescriptorTime.push_back(timeDes);
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {

            /* MATCH KEYPOINT DESCRIPTORS */

            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptorType = "DES_HOG"; // DES_BINARY, DES_HOG
            string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType, matcherType, selectorType);

            //// EOF STUDENT ASSIGNMENT
			numCarMatching.push_back(matches.size());
          
            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            // visualize matches between current and previous image
            bVis = false;
            if (bVis)
            {
                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::imshow(windowName, matchImg);
                cout << "Press key to continue to next image" << endl;
                cv::waitKey(0); // wait for key to be pressed
            }
            bVis = false;
        }
        	  		//calc std of detected in x and y of points
  float eX = 0.0;
  float eY = 0.0;
  float sumX = 0.0;
  float sumY = 0.0;

  for (auto kp : keypoints)
  {
    sumX+=kp.pt.x;
    sumY+=kp.pt.y;
  }
  float avgX = sumX/keypoints.size();
  float avgY = sumY/keypoints.size();
  
  for (auto kp : keypoints)
  {
   eX+=(kp.pt.x - avgX)*(kp.pt.x - avgX);
   eY+=(kp.pt.y - avgY)*(kp.pt.y - avgY);
   
   stdX = sqrt(eX/keypoints.size());
   stdY = sqrt(eY/keypoints.size());
   
  } 
  stdXDetection.push_back(stdX);
  stdYDetection.push_back(stdY);

    } // eof loop over all images
  
    	//Calculate Detector Avgs for 10 imgs
  	 double avgDetectorTime = accumulate(eachDetectionTime.begin(), eachDetectionTime.end(), 0.0)/eachDetectionTime.size();
     int avgTotalDetection = accumulate(numTotalDetection.begin(),numTotalDetection.end(), 0)/numTotalDetection.size();
     int avgCarDetection = accumulate(numCarDetection.begin(), numCarDetection.end(), 0)/numCarDetection.size();
     float avgStdX = accumulate(stdXDetection.begin(), stdXDetection.end(), 0)/stdXDetection.size();
  	 float avgStdY = accumulate(stdYDetection.begin(), stdYDetection.end(), 0)/stdYDetection.size();
     int avgCarMatches = accumulate(numCarMatching.begin(), numCarMatching.end(), 0)/numCarMatching.size();
  	double avgDescriptorTime = accumulate(eachDescriptorTime.begin(),eachDescriptorTime.end(),0.0)/eachDescriptorTime.size();
      //push Detector avg into vector for 10 imgs 
      allDetectorAvgTime.push_back(avgDetectorTime);
      allDetectorAvgCar.push_back(avgCarDetection);
      allDetectorAvgTotal.push_back(avgTotalDetection);
  	  allDetectorStdX.push_back(avgStdX);
      allDetectorStdY.push_back(avgStdY);
  	  allMatchingAvgCar.push_back(avgCarMatches);
  	  allDescriptorAvgTime.push_back(avgDescriptorTime);
  
  	  cout<< "Avg for 10 images"<<endl;
 	  cout<< "Avg detection time        = " <<allDetectorAvgTime[0]*1000  / 1.0 << " ms" << endl;
 	  cout<< "Avg Total Detections      = " <<allDetectorAvgTotal[0]  << endl;
 	  cout<< "Avg Car Detections        = " <<allDetectorAvgCar[0]  << endl;
      cout<< "Avg num of Matches        = " <<allMatchingAvgCar[0] << endl;
  	  cout<< "Avg descriptor time       = " <<allDescriptorAvgTime[0]*1000  / 1.0 << " ms" << endl;
  	  cout<< "Detector +Descriptor Time = " <<(allDetectorAvgTime[0]*1000  / 1.0 )+(allDescriptorAvgTime[0]*1000  / 1.0 )<< " ms" << endl;
  	


    return 0;
}
