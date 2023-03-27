#pragma once
#ifndef LDSO_TWITCH_UNDISTORT_H_
#define LDSO_TWITCH_UNDISTORT_H_

#include <Eigen/Core>
#include "frontend/ImageAndExposure.h"
#include "NumTypes.h"
#include "MinimalImage.h"

namespace ldso {
    // A photometric undistorter (handles Gamma Response, Vignette)
    class PhotometricUndistorter {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Constructor
        PhotometricUndistorter(std::string file, std::string noiseImage, std::string vignetteImage, int w_, int h_);

        // Destructor
        ~PhotometricUndistorter();

        /*
        Removes readout noise, and converts to irradiance
        Affine normalizes values to 0 <= I < 256.
        raw irradiance = a*I + b.
        output will be written in [output].        
        */
        template<typename T>
        void processFrame(T *image_in, float exposure_time, float factor = 1);

        void unMapFloatImage(float *image);

        ImageAndExposure *output;

        float *getG() {
            if(!valid)
            {
                return 0;
            }
            else
            {
                return G;
            }
        };

        private:

        float G[256 * 256]; // Gamma response
        int GDepth;
        float *vignetteMap;
        float *vignetteMapInv;

        int w, h;
        bool valid;

    };


    // A geometric undistorter (handles Pinhole, KannalaBrandt, FOV, Equidistant camera and distortion models)
    class Undistort {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Destructor:
        virtual ~Undistort();

        // Distortion and Model Type will be implemented by the inherited classes
        virtual void distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const = 0;
        virtual const char* getCameraModelType() const = 0;

        // inline methods:

        // 1.) return camera calibration (3x3 matrix)
        inline const Mat33 getK() const {
            return K;
        };

        // 2.) return cropped image dimensions (2D Vector)
        inline const Eigen::Vector2i getSize() const {
            return Eigen::Vector2i(w, h);
        };

        // 3.) return calibration parameters (depending on the camera model)
        inline const VecX getOriginalParameter() const {
            return parsOrg;
        };

        // 4.) return original image dimensions (2D Vector)
        inline const Eigen::Vector2i getOriginalSize() const {
            return Eigen::Vector2i(wOrg, hOrg);
        };

        // 5.) return if Undistorter is valid
        inline bool isValid() {
            return valid;
        };

        // Prototype methods (will be implemented in the inherited classes)
        // 1.) Undistort method, take image data, exposure and timestamp and create an ImageAndExposure object
        template<typename T>
        ImageAndExposure *
        undistort(const MinimalImage<T> *image_raw, float exposure = 0, double timestamp = 0, float factor = 1) const;

        // 2.) Return Undistorter subclass given a specific geometric and photometric calibration file
        static Undistort *
        getUndistorterForFile(std::string configFilename, std::string gammaFilename, std::string vignetteFilename);

        void loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage);

        PhotometricUndistorter *photometricUndist;


        protected:

        int w, h, wOrg, hOrg, wUp, hUp;
        int upsampleUndistFactor;
        Mat33 K; // Pinhole camera model (intrinsic camera parameters)
        VecX parsOrg; // Contains the parsed values from file "camera.txt"
        
        bool valid;
        bool passthrough;

        float *remapX;
        float *remapY;

        void applyBlurNoise(float *img) const;

        void makeOptimalK_crop();
        void makeOptimalK_full();

        void readFromFile(const char *configFileName, int nPars, std::string prefix="");

    };

    // Subclasses, which implement different camera and distortion models:

    // Field-Of-View Undistorter:
    class UndistortFOV : public Undistort {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Constructor
        UndistortFOV(const char *configFileName, bool noprefix);

        // Destructor
        ~UndistortFOV();

        // Methods for distorting coordinate and getting the CameraModel Type as const char*
        void distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const override;
        inline const char* getCameraModelType() const override { return "FOV"; };

    };
    // Radial and Tangential Undistorter:
    class UndistortRadTan : public Undistort {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Constructor
        UndistortRadTan(const char *configFileName, bool noprefix);

        // Destructor
        ~UndistortRadTan();

        // Methods for distorting coordinate and getting the CameraModel Type as const char*
        void distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const override;
        inline const char* getCameraModelType() const override { return "RadTan"; };

    };
    // Equidistant Undistorter:
    class UndistortEquidistant : public Undistort {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Constructor
        UndistortEquidistant(const char *configFileName, bool noprefix);

        // Destructor
        ~UndistortEquidistant();

        // Methods for distorting coordinate and getting the CameraModel Type as const char*
        void distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const override;
        inline const char* getCameraModelType() const override { return "Equidistant"; };

    };
    // Pinhole Undistorter:
    class UndistortPinhole : public Undistort {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Constructor
        UndistortPinhole(const char *configFileName, bool noprefix);

        // Destructor
        ~UndistortPinhole();

        // Methods for distorting coordinate and getting the CameraModel Type as const char*
        void distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const override;
        inline const char* getCameraModelType() const override { return "Pinhole"; };

        private:
        float inputCalibration[8];

    };
    // Kannala-Brandt Undistorter:
    class UndistortKB : public Undistort {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // Constructor
        UndistortKB(const char *configFileName, bool noprefix);

        // Destructor
        ~UndistortKB();

        // Methods for distorting coordinate and getting the CameraModel Type as const char*
        void distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const override;
        inline const char* getCameraModelType() const override { return "KB"; };

    };
}

#endif // LDSO_TWITCH_UNDISTORT_H_