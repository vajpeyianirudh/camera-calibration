#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/reg/map.hpp>
#include <opencv2/aruco.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <numeric>
#include <filesystem>

using namespace std;
using namespace cv;
using namespace filesystem;

const float calibSquareDim = 0.01908f;
const float arucoSquareDim = 0.02489f;
const Size chessboardDim = Size(9,6);
string directoryName;

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners){
    for(int i = 0; i < boardSize.height; i++){
        for (int j = 0; j < boardSize.width; j++){
            corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength, 0.0f));
        }
    }
}

void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults){
    for (vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++){
        vector<Point2f> pointBuf;
        bool found = findChessboardCorners(*iter, Size(9, 6), pointBuf, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

        if (found){
            allFoundCorners.push_back(pointBuf);
        }
        if (showResults){
            drawChessboardCorners(*iter, Size(9, 6), pointBuf, found);
            imshow("Looking for Corners", *iter);
            waitKey(0);
        }
    }
}

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients){
    vector<vector<Point2f>> chessboardImageSpacePoints;
    getChessboardCorners(calibrationImages, chessboardImageSpacePoints, false);

    vector<vector<Point3f>> worldSpaceCornerPoints(1);

    createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
    worldSpaceCornerPoints.resize(chessboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);

    vector<Mat> rVectors, tVectors;
    distanceCoefficients = Mat::zeros(8, 1, CV_64F);

    calibrateCamera(worldSpaceCornerPoints, chessboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors, tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients){
    ofstream outStream(name);
    if(outStream){
        uint16_t rows = cameraMatrix.rows;
        uint16_t columns = cameraMatrix.cols;

        outStream << rows << endl << columns << endl;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = cameraMatrix.at<double>(r, c);
                outStream << value << endl;
            }
        }

        rows = distanceCoefficients.rows;
        columns = distanceCoefficients.cols;

        outStream << rows << endl << columns << endl;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = distanceCoefficients.at<double>(r, c);
                outStream << value << endl;
            }
        }

        outStream.close();
        return true;
    }
    return false;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distancecoefficients){
    ifstream inStream(name);
    if(inStream){
        uint16_t rows, columns;

        inStream >> rows;
        inStream >> columns;

        cameraMatrix = Mat(Size(rows, columns), CV_64F);

        for(int r = 0; r < rows; r++){
            for (int c = 0; c < columns; c++){
                double read = 0.0f;
                inStream >> read;
                cameraMatrix.at<double>(r, c) = read;
                //cout << cameraMatrix.at<double>(r, c) << "\n";
            }
        }

        inStream >> rows;
        inStream >> columns;

        distancecoefficients = Mat::zeros(rows, columns, CV_64F);

        for(int r = 0; r < rows; r++){
            for (int c = 0; c < columns; c++){
                double read = 0.0f;
                inStream >> read;
                distancecoefficients.at<double>(r, c) = read;
                //cout << distancecoefficients.at<double>(r, c) << "\n";
            }
        }
        inStream.close();
        return true;
    }
    return false;
}

int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDim, bool changeResolution){
    Mat frame;
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;

    Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    VideoCapture vid(0);
    if(changeResolution){
        vid.set(CAP_PROP_FRAME_HEIGHT, 600);
        vid.set(CAP_PROP_FRAME_WIDTH, 800);
    }

    if(!vid.isOpened()){
        return -1;
    }
    namedWindow("Webcam", WINDOW_AUTOSIZE);

    vector<Vec3d> rotationVectors, translationVectors;

    while(true){
        if(!vid.read(frame))
            break;

        aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
        aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDim, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);

        for(int i = 0; i < markerIds.size(); i++){
            // X axis : red, Y axis : green, Z axis : blue
            aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i], translationVectors[i], 0.025f);
        }
        imshow("Webcam", frame);
        if(waitKey(30) >= 0) break;
    }
    return 1;
}

void cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients, bool changeResolution, bool rotateImage){
    Mat frame, drawToFrame;
//    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
//    Mat distanceCoefficients;
    vector<Mat> savedImages;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;
    VideoCapture cap;
    int framesPerSecond = 20;
    cap.open(0);
    if(changeResolution){
        cap.set(CAP_PROP_FRAME_HEIGHT, 600);
        cap.set(CAP_PROP_FRAME_WIDTH, 800);
    }
    if(!cap.isOpened())
     {
       cout << "Error opening video stream" << endl;
     }

    namedWindow("Webcam", WINDOW_AUTOSIZE);

    while(true){
        if (!cap.read(frame))
            break;
        if (rotateImage){
            rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
        }

        //cout << "Trying calibration" << endl;
        vector<Vec2f> foundPoints;
        bool found = false;
        found = findChessboardCorners(frame, chessboardDim, foundPoints, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(drawToFrame);
        drawChessboardCorners(drawToFrame, chessboardDim, foundPoints, found);

        if(found){
            imshow("webcam", drawToFrame);
        }

        else {
            imshow("webcam", frame);
        }

        char character = waitKey(1000/framesPerSecond);

        switch (character) {
        case 32:
            //saving images
            if (found){
                cout << "found image and saving it" << endl;
                Mat temp;
                frame.copyTo(temp);
                savedImages.push_back(temp);
                if(savedImages.size() == 1){
                    auto t = std::time(nullptr);
                    auto tm = *std::localtime(&t);
                    std::cout << std::put_time(&tm, "%b-%d-%Y-%H-%M-%S") << std::endl;
                    ostringstream oss;
                    oss << put_time(&tm, "%b-%d-%Y-%H-%M-%S");
                    directoryName = "../" + oss.str();
                    create_directory(directoryName);
                }
                string filename = directoryName + "/" + to_string(savedImages.size()) + ".png";
                cout << filename << endl;
                imwrite(filename, temp);
            }
            break;
        case 13:
            //start calibration
            cout << "starting and saving calibration" << endl;
            if(savedImages.size() > 10){
                cameraCalibration(savedImages, chessboardDim, calibSquareDim, cameraMatrix, distanceCoefficients);
                saveCameraCalibration(directoryName + "/" + "IntrinsicMatrixOpenCV.txt", cameraMatrix, distanceCoefficients);
            }
            else {cout << "not enough images" << endl;}
            break;
        case 27:
            //exit
            return;
            break;
        }
    }
}

bool solveExtrinsicMatrix(Mat cameraIntrinsics, Mat distanceCoefficients, Mat& cameraExtrinsics){

    vector<Point3d> EMPoints = {{-104.562, -103.861, 86.281}, {76.7444, -98.59, 89.09}, {85.2069, -423.835, 113.24}, {-97.4618, -419.716, 115.433}};
    vector<Point2d> imagePoints = {{112.25, 46.5}, {460.25, 52}, {511.25, 700.75}, {126.5, 715}};
    Mat rotationVectors, translationVectors;
    Mat rotationMatrix;

    cout << "starting PnP" << endl;
    cameraExtrinsics = solvePnPRansac(EMPoints, imagePoints, cameraIntrinsics, distanceCoefficients, rotationVectors, translationVectors, false,\
                                      100, 2.0, 0.99);

    cout << "PnP solved" << endl << "Starting Rodrigues" << endl;
    Rodrigues(rotationVectors, rotationMatrix);
    cout << "Rodrigues solved" << endl;

    hconcat(rotationMatrix, translationVectors, cameraExtrinsics);


    //saving extrinsic Matrix

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::cout << std::put_time(&tm, "%b-%d-%Y-%H-%M-%S") << std::endl;
    ostringstream oss;
    oss << put_time(&tm, "%b-%d-%Y-%H-%M-%S");
    string extrinsicsFileName = "../Extrinsics" + oss.str() + ".csv";

    ofstream extrinsicStream(extrinsicsFileName);
    if(extrinsicStream){
        uint16_t rows = cameraExtrinsics.rows;
        uint16_t columns = cameraExtrinsics.cols;

        extrinsicStream << rows << endl << columns << endl;

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                double value = cameraExtrinsics.at<double>(r, c);
                extrinsicStream << value << endl;
            }
        }
        extrinsicStream.close();
    }

    return true;
}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    bool changeResolution = true;
    bool rotateImage = true;

    Mat cameraMatrix;// = Mat::eye(3, 3, CV_64F);
    Mat distanceCoefficients;
    Mat cameraExtrinsicMatrix = Mat::eye(3, 4, CV_64F);

    cameraCalibrationProcess(cameraMatrix, distanceCoefficients, changeResolution, rotateImage);

//    loadCameraCalibration("IntrinsicMatrixOpenCV.txt", cameraMatrix, distanceCoefficients);

//    solveExtrinsicMatrix(cameraMatrix, distanceCoefficients, cameraExtrinsicMatrix);

//    startWebcamMonitoring(cameraMatrix, distanceCoefficients, arucoSquareDim, changeResolution);

    return a.exec();
}
