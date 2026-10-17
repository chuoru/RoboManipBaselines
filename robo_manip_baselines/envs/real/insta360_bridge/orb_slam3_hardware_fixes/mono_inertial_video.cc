/**
* Offline Monocular-Inertial driver for a raw video file + a long-format IMU
* CSV (as produced by robo_manip_baselines/misc/ConvertInsvToMp4AndImu.py
* from Insta360 .insv recordings), instead of ORB-SLAM3's usual pre-extracted
* image-sequence datasets (EuRoC/TUM-VI style).
*/

#include<cstdlib>
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>

#include<opencv2/opencv.hpp>

#include<System.h>
#include "ImuTypes.h"

using namespace std;

// Loads a long-format IMU CSV with header "timestamp_ms,sensor_type,x,y,z"
// where every pair of adjacent data rows shares the same timestamp_ms in
// the fixed order (gyro, accel). Fails loudly if that invariant doesn't
// hold, rather than silently mispairing gyro/accel.
void LoadIMUCsv(const string &strImuCsvPath, vector<double> &vTimeStamps,
                 vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{
    ifstream fImu(strImuCsvPath.c_str());
    if(!fImu.is_open())
    {
        cerr << "Failed to open IMU csv: " << strImuCsvPath << endl;
        exit(1);
    }

    string header;
    getline(fImu, header);

    vTimeStamps.reserve(50000);
    vAcc.reserve(50000);
    vGyro.reserve(50000);

    while(!fImu.eof())
    {
        string lineGyro, lineAccel;
        if(!getline(fImu, lineGyro) || lineGyro.empty())
            break;
        if(!getline(fImu, lineAccel) || lineAccel.empty())
        {
            cerr << "IMU csv ended with an unpaired row: " << lineGyro << endl;
            exit(1);
        }

        stringstream ssGyro(lineGyro), ssAccel(lineAccel);
        string tsGyroStr, typeGyro, gx, gy, gz;
        string tsAccelStr, typeAccel, ax, ay, az;

        getline(ssGyro, tsGyroStr, ',');
        getline(ssGyro, typeGyro, ',');
        getline(ssGyro, gx, ',');
        getline(ssGyro, gy, ',');
        getline(ssGyro, gz, ',');

        getline(ssAccel, tsAccelStr, ',');
        getline(ssAccel, typeAccel, ',');
        getline(ssAccel, ax, ',');
        getline(ssAccel, ay, ',');
        getline(ssAccel, az, ',');

        if(typeGyro != "gyro" || typeAccel != "accel" || tsGyroStr != tsAccelStr)
        {
            cerr << "Unexpected IMU csv row order/pairing at timestamp "
                 << tsGyroStr << " (" << typeGyro << ") / "
                 << tsAccelStr << " (" << typeAccel << "). "
                 << "Expected an adjacent (gyro, accel) pair sharing the same timestamp_ms."
                 << endl;
            exit(1);
        }

        double t_sec = stod(tsGyroStr) / 1000.0;
        vTimeStamps.push_back(t_sec);
        vGyro.push_back(cv::Point3f(stof(gx), stof(gy), stof(gz)));
        vAcc.push_back(cv::Point3f(stof(ax), stof(ay), stof(az)));
    }
}

int main(int argc, char **argv)
{
    if(argc != 6 && argc != 7)
    {
        cerr << endl << "Usage: ./mono_inertial_video path_to_vocabulary path_to_settings "
                         "path_to_video.mp4 path_to_imu.csv path_to_trajectory_output_basename "
                         "[--viewer]" << endl;
        return 1;
    }

    const string strVoc = argv[1];
    const string strSettings = argv[2];
    const string strVideo = argv[3];
    const string strImuCsv = argv[4];
    const string strOutBase = argv[5];
    const bool bUseViewer = (argc == 7) && (string(argv[6]) == "--viewer");

    vector<double> vTimestampsImu;
    vector<cv::Point3f> vAcc, vGyro;
    cout << "Loading IMU csv..." << endl;
    LoadIMUCsv(strImuCsv, vTimestampsImu, vAcc, vGyro);
    cout << "LOADED! " << vTimestampsImu.size() << " IMU samples." << endl;

    cv::VideoCapture cap(strVideo);
    if(!cap.isOpened())
    {
        cerr << "Failed to open video: " << strVideo << endl;
        return 1;
    }
    const double fps = cap.get(cv::CAP_PROP_FPS);
    const int nFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    cout << "Video: " << nFrames << " frames at " << fps << " fps." << endl;

    // TEMPORARY (debugging why the map never initializes -- 0 KFs): bump
    // from the library's default VERBOSITY_NORMAL so Tracking's internal
    // PrintMess calls (initialization attempts/failures, feature counts,
    // etc.) actually print instead of being filtered out.
    ORB_SLAM3::Verbose::SetTh(ORB_SLAM3::Verbose::VERBOSITY_DEBUG);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // bUseViewer=false (default, no --viewer arg): skips creating the
    // Pangolin window entirely (see System::System in src/System.cc), so
    // this runs fully headless.
    ORB_SLAM3::System SLAM(strVoc, strSettings, ORB_SLAM3::System::IMU_MONOCULAR, bUseViewer);
    const float imageScale = SLAM.GetImageScale();

    // Circular safety mask blacking out pixels beyond the calibration's
    // valid angular range -- see Insta360X3_2880.yaml's Camera1.k1-k4
    // comment: that fit (rescaled from the live-stream calibration) is
    // only monotonic (physically valid) up to ~44 degrees incidence.
    // Feeding ORB_SLAM3 pixels past that segfaulted it (libORB_SLAM3.so)
    // in envs/real/insta360_bridge's live pipeline; same fix applied here.
    // Native-resolution center/radius (2880x2880, r=1125px), scaled by
    // imageScale to match whatever resolution actually reaches
    // TrackMonocular below.
    const cv::Point kMaskCenterNative(1441, 1441);
    const int kMaskRadiusNative = 1125;
    cv::Mat safe_mask;

    size_t firstImu = 0;
    cv::Mat frame;
    for(int ni = 0; ni < nFrames; ni++)
    {
        if(!cap.read(frame))
            break;

        const double tframe = ni / fps;

        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        while(firstImu < vTimestampsImu.size() && vTimestampsImu[firstImu] <= tframe)
        {
            vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                vAcc[firstImu].x, vAcc[firstImu].y, vAcc[firstImu].z,
                vGyro[firstImu].x, vGyro[firstImu].y, vGyro[firstImu].z,
                vTimestampsImu[firstImu]));
            firstImu++;
        }

        if(imageScale != 1.f)
            cv::resize(frame, frame, cv::Size(), imageScale, imageScale);

        // kMaskCenterNative/kMaskRadiusNative above are calibrated for the
        // *native* 2880x2880 on-device recording this mask was originally
        // written for (imageScale<1, downscaled by SLAM to the settings'
        // Camera.width/height). envs/real/insta360_bridge's own --record
        // output is already cropped+masked+resized to the target
        // resolution server-side (imageScale==1 here, frame arrives
        // pre-masked) -- applying this native-2880 mask on top of an
        // already-640x640 frame puts its center (1441,1441) entirely
        // outside the frame (nearest corner is ~1134px away, less than the
        // 1125px radius), so the "circle" never overlaps the visible
        // canvas and every frame comes out pure black. Confirmed: this
        // silently produced zero ORB keypoints on every single frame
        // (0 KFs, no map ever initialized) once the separate
        // Frame::mpMutexImu null-pointer crash on empty keypoints was
        // fixed in libORB_SLAM3.so -- that crash was in fact this same
        // bug manifesting as a crash instead of silent failure. Only
        // apply the legacy mask when SLAM is actually downscaling
        // (imageScale != 1), i.e. the true native-resolution case.
        cv::Mat masked_frame;
        if(imageScale != 1.f)
        {
            if(safe_mask.empty())
            {
                safe_mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
                cv::circle(safe_mask,
                           cv::Point(kMaskCenterNative.x * imageScale, kMaskCenterNative.y * imageScale),
                           static_cast<int>(kMaskRadiusNative * imageScale), cv::Scalar(255), -1);
            }
            masked_frame = cv::Mat::zeros(frame.size(), frame.type());
            frame.copyTo(masked_frame, safe_mask);
        }
        else
        {
            masked_frame = frame;
        }

        SLAM.TrackMonocular(masked_frame, tframe, vImuMeas);
    }

    // Save trajectories BEFORE Shutdown() (reordered from the stock
    // examples' Shutdown()-then-save order): with --viewer, System::
    // Shutdown() (src/System.cc) never synchronizes with the still-running
    // Pangolin viewer thread (a known upstream ORB_SLAM3 issue -- its
    // mpViewer->RequestFinish()/isFinished() wait is commented out), which
    // segfaulted envs/real/insta360_bridge on exit; saving first means
    // that's avoided below by skipping Shutdown() entirely rather than
    // losing the trajectory output too.
    //
    // SaveTrajectoryTUM is not usable here: it explicitly rejects monocular
    // sensors (see src/System.cc), so use the EuRoC-style trajectory savers
    // instead, same as the stock mono_inertial_tum_vi/mono_inertial_euroc examples.
    SLAM.SaveTrajectoryEuRoC(strOutBase + "_CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC(strOutBase + "_KeyFrameTrajectory.txt");

    if(bUseViewer)
    {
        // _Exit() skips the C++ destructor chain (and thus Shutdown()'s
        // risky viewer-thread wait) entirely -- acceptable since --viewer
        // is an interactive/debug aid here, not a headless batch run.
        std::_Exit(0);
    }
    SLAM.Shutdown();

    return 0;
}
