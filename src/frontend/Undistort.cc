
// Data parsing
#include <sstream>
#include <fstream>
#include <iostream>

// Operations on Vectors and Matrices
#include <Eigen/Core>
#include <iterator>

// Our custom files:
#include "Settings.h"
#include "internal/GlobalFuncs.h"
#include "frontend/Undistort.h"
#include "frontend/ImageRW.h"

using namespace ldso::internal;

namespace ldso {

    /*
        Summary of what we need to implement:
    */

    // Implement PhotometricUndistorter constructor
    // Implement PhotometricUndistorter destructor
    // Implement PhotometricUndistorter methods (unMapFloatImage, 3x"processFrame" due to different image types)

    // Implement Distort destructor
    // Implement Distort method getUndistorterForFile
    // Implement Distort method loadPhotometricCalibration
    // Implement Distort method undistort (3x due to different image types)
    // Implement Distort method applyBlurNoise
    // Implement Distort method makeOptimalK_crop
    // Implement Distort method makeOptimalK_full
    // Implement Distort method readFromFile

    // Implement DistortFOV constructor
    // Implement DistortFOV destructor
    // Implement DistortFOV method distortCoordinates

    // Implement DistortRadTan constructor
    // Implement DistortRadTan destructor
    // Implement DistortRadTan method distortCoordinates

    // Implement DistortEquidistant constructor
    // Implement DistortEquidistant destructor
    // Implement DistortEquidistant method distortCoordinates

    // Implement DistortKB constructor
    // Implement DistortKB destructor
    // Implement DistortKB method distortCoordinates

    // Implement DistortPinhole constructor
    // Implement DistortPinhole destructor
    // Implement DistortPinhole method distortCoordinates

    /// Here is the implementation:
    //  +++++++++++++++++++++++++++++


    // Implement PhotometricUndistorter constructor
    PhotometricUndistorter::PhotometricUndistorter(
        std::string file,
        std::string noiseImage,
        std::string vignetteImage,
        int w_, int h_
    ) {
        valid = false;
        vignetteMap = 0;
        vignetteMapInv = 0;
        w = w_;
        h = h_;
        output = new ImageAndExposure(w, h);
        if(file == "" || vignetteImage == "")
        {
            std::cout << "PhotometricUndistorter: NO PHOTOMETRIC Calibration!" << std::endl;
        }

        // read G (Gamma response)
        std::ifstream f(file.c_str());
        std::cout << "PhotometricUndistorter: Reading Photometric Calibration from file " << file.c_str() << std::endl;
        if(!f.good())
        {
            std::cout << "PhotometricUndistorter: Could not open file!" << std::endl;
            return;
        }

        // Parsing the Gamma Response (pcalib.txt)
        {
            std::string line;
            std::getline(f, line);
            std::istringstream l1i(line);
            std::vector<float> Gvec = std::vector<float>(std::istream_iterator<float>(l1i), 
                                                         std::istream_iterator<float>());

            GDepth = Gvec.size(); // Expected to have size >= 256 (see pcalib.txt)

            if (GDepth < 256)
            {
                std::cout << "PhotometricUndistorter: invalid format! Got " << (int) Gvec.size() << 
                " entries in first line, expected at least 256!" << std::endl;
                return;
            }

            for (int i = 0; i < GDepth; i++)
            {
                // fill in G-Object by parsed values:
                G[i] = Gvec[i];
            }

            for (int i = 0; i < GDepth - 1; i++)
            {
                /* code */
                if(G[i+1] <= G[i])
                {
                    std::cout << "PhotometricUndistorter: G invalid! It has to be strictly increasing, but it isnt!" << std::endl;
                    return;
                }
            }

            float min = G[0];
            float max = G[GDepth - 1];

            for (int i = 0; i < GDepth; i++)
            {
                // Make it to MIN...MAX => 0...255.
                G[i] = 255.0 * (G[i] - min) / (max - min);
            }
        }

        if (setting_photometricCalibration == 0)
        {
            for (int i = 0; i < GDepth; i++)
            {
                // With perfect images, we initialize the Gamma response to a linear curve
                // i.e. G = i / n: 0/255, 1/255, 2/255, 3/255, ... 255/255.
                G[i] = 255.0f * i / (float) (GDepth - 1);
            }
        }


        std::cout << "PhotometricUndistorter: Reading Vignette Image from " << vignetteImage.c_str() << "." << std::endl;

        // read Vignette images into MinimalImage types (both 16 bit and 8 bit)
        MinimalImage<unsigned short> *vm16 = IOWrap::readImageBW_16U(vignetteImage.c_str());
        MinimalImageB *vm8 = IOWrap::readImageBW_8U(vignetteImage.c_str());

        // Initialize vignette data types:
        vignetteMap = new float[w * h];
        vignetteMapInv = new float[w * h];

        // Fill up vignette image data:
        if(vm16 != 0)
        {
            if(vm16->w != w || vm16->h != h)
            {
                std::cout << "PhotometricUndistorter: Invalid vignette image size! Got " 
                << vm16->w << " x " << vm16->h 
                << ", expected " << w << " x " << h << std::endl;
                if(vm16 != 0) delete vm16;
                if(vm8 != 0) delete vm8;
                return;
            }

            float maxV = 0;
            for (int i = 0; i < w * h; i++)
            {
                // Update the maxV value based on the parsed image intensity
                if(vm16->at(i) > maxV)
                {
                    maxV = vm16->at(i);
                }
            }
            for (int i = 0; i < w * h; i++)
            {
                // Normalize vignette intensity values to be in range 0...1:
                vignetteMap[i] = vm16->at(i) / maxV;
            }
        }
        else if(vm8 != 0)
        {
            if(vm8->w != w || vm8->h != h)
            {
                std::cout << "PhotometricUndistorter: Invalid vignette image size! Got " 
                << vm8->w << " x " << vm8->h 
                << ", expected " << w << " x " << h << std::endl;
                if(vm16 != 0) delete vm16;
                if(vm8 != 0) delete vm8;
                return;
            }

            float maxV = 0;
            for (int i = 0; i < w * h; i++)
            {
                // Update the maxV value based on the parsed image intensity
                if(vm8->at(i) > maxV)
                {
                    maxV = vm8->at(i);
                }
            }
            for (int i = 0; i < w * h; i++)
            {
                // Normalize vignette intensity values to be in range 0...1:
                vignetteMap[i] = vm8->at(i) / maxV;
            }
        }
        else 
        {
            std::cout << "PhotometricUndistorter: Invalid vignette image!" << std::endl;
            if(vm16 != 0) delete vm16;
            if(vm8 != 0) delete vm8;
            return;
        }

        if(vm16 != 0) delete vm16;
        if(vm8 != 0) delete vm8;

        // Fill in the inverse vignette Map:
        for (int i = 0; i < w * h; i++)
        {
            vignetteMapInv[i] = 1.0f / vignetteMap[i];
        }

        std::cout << "PhotometricUndistorter: Successfully read photometric calibration!" << std::endl;
        valid = true; // Means we have proper values for Gamma Response and Vignette!
    }


    // Implement PhotometricUndistorter destructor
    PhotometricUndistorter::~PhotometricUndistorter()
    {
        // Cleanup the reserved memory
        if( vignetteMap != 0)
        {
            delete[] vignetteMap;
        }
        if( vignetteMapInv != 0)
        {
            delete[] vignetteMapInv;
        }
        delete output; // ImageAndExposure
    }
    // Implement PhotometricUndistorter methods (unMapFloatImage, 3x"processFrame" due to different image types)
    void PhotometricUndistorter::unMapFloatImage(float *image)
    {
        int wh = w * h;
        for (int i = 0; i < wh; i++)
        {
            float BinvC;
            float color = image[i];

            if(color < 1e-3)
            {
                BinvC = 0.0f;
            }
            else if( color > GDepth - 1.01f)
            {
                BinvC = GDepth - 1.1; /// TODO: For later, check if this can really be correct (why not 1.01f?)
            }
            else
            {
                int c = color;
                float a = color - c;
                // Linear interpolation across neighboring pixels:
                BinvC = G[c] * (1 - a) + G[c + 1] * a;
            }

            float val = BinvC;
            if (val < 0)
            {
                val = 0;
            }
            image[i] = val;
        }
    }

    template<typename T>
    void PhotometricUndistorter::processFrame(T *image_in, float exposure_time, float factor)
    {
        int wh = w * h;
        float *data = output->image; // ImageAndExposure: Float pointer to image data

        assert(output->w == w && output->h == h);
        assert(data != 0);

        if(!valid || exposure_time <= 0 || 
        setting_photometricCalibration == 0 // Full photometric calibration disabled (i.e. perfect images).
        )
        {
            for (int i = 0; i < wh; i++)
            {
                data[i] = factor * image_in[i];
            }
            output->exposure_time = exposure_time;
            output->timestamp = 0;
        }
        else
        {
            for (int i = 0; i < wh; i++)
            {
                data[i] = G[image_in[i]];
            }

            if( setting_photometricCalibration == 2)
            {
                // Remove vignetting using the inverse vignette map (multiply by 1/vignette)
                for (int i = 0; i < wh; i++)
                {
                    if(!std::isinf(vignetteMapInv[i]))
                    {
                        data[i] *= vignetteMapInv[i];
                    }
                    else
                    {
                        data[i] *= vignetteMapInv[i];
                    }
                    
                }
                
            }

            output->exposure_time = exposure_time;
            output->timestamp = 0;
        }

        if (!setting_useExposure)
        {
            output->exposure_time = 1;
        }
        
    }

    /// TODO: Can this be deleted as it is a leftover from previous code (not templated)?
    template
    void PhotometricUndistorter::processFrame<unsigned char>(unsigned char *image_in, float exposure_time, float factor);
    template
    void PhotometricUndistorter::processFrame<unsigned short>(unsigned short *image_in, float exposure_time, float factor);

    // Implement Undistort destructor
    Undistort::~Undistort()
    {
        if(remapX != 0)
        {
            delete[] remapX;
        }
        if(remapY != 0)
        {
            delete[] remapY;
        }
    }
    // Implement Undistort method getUndistorterForFile
    Undistort *Undistort::getUndistorterForFile(std::string configFilename, 
                                                std::string gammaFilename, 
                                                std::string vignetteFilename)
    {
        std::cout << "Reading Calibration from file " << configFilename.c_str() << std::endl;

        std::ifstream f(configFilename.c_str());
        if(!f.good())
        {
            f.close();
            std::cout << " ... not found. Cannot operate without calibration, shutting down." << std::endl;
            f.close(); /// TODO: Why again?
            return 0;
        }

        std::cout << " ... found!" << std::endl;
        std::string l1;
        std::getline(f, l1);
        f.close();


        //std::cout << "l1 variable contains: " << l1 << std::endl;

        float ic[10];
        
        Undistort *u;

        // for backwards-compatibility: Use RadTan model for 8 parameters.
        if(std::sscanf(l1.c_str(), "%f %f %f %f %f %f %f %f", 
                                    &ic[0], &ic[1], &ic[2], &ic[3], 
                                    &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
        {
            std::cout << "Found RadTan (OpenCV) camera model, building rectifier." << std::endl;
            u = new UndistortRadTan(configFilename.c_str(), true);
            if(!u->isValid())
            {
                delete u;
                return 0;
            }
        }
        // for backwards-compatibility: Use Pinhole / FoV model for 5 parameters.
        else if(std::sscanf(l1.c_str(), "%f %f %f %f %f", 
                                    &ic[0], &ic[1], &ic[2], &ic[3], &ic[4]) == 5)
        {
            std::cout << "Scanned 5 values!" << std::endl;
            if(ic[4] == 0)
            {
                std::cout << "Found PINHOLE camera model, building rectifier." << std::endl;
                u = new UndistortPinhole(configFilename.c_str(), true);
                if(!u->isValid())
                {
                    delete u;
                    return 0;
                }
            }
            else
            {
                std::cout << "Found FoV camera model, building rectifier." << std::endl;
                u = new UndistortFOV(configFilename.c_str(), true);
                if(!u->isValid())
                {
                    delete u;
                    return 0;
                }
            }
        }
        // Clean model selection implementation, i.e. camera.txt starts with the camera model type:
        else if(std::sscanf(l1.c_str(), "KannalaBrandt %f %f %f %f %f %f %f %f", 
                                    &ic[0], &ic[1], &ic[2], &ic[3], 
                                    &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
        {
            std::cout << "Found KannalaBrand camera model, building rectifier." << std::endl;
            u = new UndistortKB(configFilename.c_str(), false);

            if(!u->isValid())
            {
                delete u;
                return 0;
            }
            
        }
        else if(std::sscanf(l1.c_str(), "RadTan %f %f %f %f %f %f %f %f", 
                                    &ic[0], &ic[1], &ic[2], &ic[3], 
                                    &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
        {
            std::cout << "Found RadTan camera model, building rectifier." << std::endl;
            u = new UndistortRadTan(configFilename.c_str(), false);

            if(!u->isValid())
            {
                delete u;
                return 0;
            }
            
        }
        else if(std::sscanf(l1.c_str(), "EquiDistant %f %f %f %f %f %f %f %f", 
                                    &ic[0], &ic[1], &ic[2], &ic[3], 
                                    &ic[4], &ic[5], &ic[6], &ic[7]) == 8)
        {
            std::cout << "Found EquiDistant camera model, building rectifier." << std::endl;
            u = new UndistortEquidistant(configFilename.c_str(), false);

            if(!u->isValid())
            {
                delete u;
                return 0;
            }
            
        }
        else if(std::sscanf(l1.c_str(), "FOV %f %f %f %f %f", 
                                    &ic[0], &ic[1], &ic[2], &ic[3], 
                                    &ic[4]) == 5)
        {
            std::cout << "Found FOV camera model, building rectifier." << std::endl;
            u = new UndistortFOV(configFilename.c_str(), false);

            if(!u->isValid())
            {
                delete u;
                return 0;
            }
            
        }
        else if(std::sscanf(l1.c_str(), "Pinhole %f %f %f %f %f", 
                                    &ic[0], &ic[1], &ic[2], &ic[3], 
                                    &ic[4]) == 5)
        {
            std::cout << "Found Pinhole camera model, building rectifier." << std::endl;
            u = new UndistortPinhole(configFilename.c_str(), false);

            if(!u->isValid())
            {
                delete u;
                return 0;
            }
            
        }
        else
        {
            std::cout << "Could not read camera.txt file! Exit." << std::endl;
            exit(1);
        }

        u->loadPhotometricCalibration(gammaFilename, "", vignetteFilename);
        
        return u;
    }


    // Implement Undistort method loadPhotometricCalibration
    void Undistort::loadPhotometricCalibration(std::string file, std::string noiseImage, std::string vignetteImage)
    {
        photometricUndist = new PhotometricUndistorter(file, noiseImage, vignetteImage, 
                                                        getOriginalSize()[0], getOriginalSize()[1]);
    }

    // Implement Undistort method undistort (3x due to different image types)
    template<typename T>
    ImageAndExposure *
    Undistort::undistort(const MinimalImage<T> *image_raw, float exposure, double timestamp, float factor) const {
        if(image_raw->w != wOrg || image_raw->h != hOrg)
        {
            std::cout << "Undistort::undistort: wrong image size (" 
            << image_raw->w << "x" << image_raw->h << " instead of "
            << w << "x" << h << ")" << std::endl;
            exit(1);
        }

        photometricUndist->processFrame<T>(image_raw->data, exposure, factor);
        ImageAndExposure *result = new ImageAndExposure(w, h, timestamp);
        photometricUndist->output->copyMetaTo(*result);

        if(!passthrough)
        {
            float *out_data = result->image;
            float *in_data = photometricUndist->output->image;

            float *noiseMapX = 0;
            float *noiseMapY = 0;

            /// TODO: Maybe leftover from some internal tests by TUM?
            if(benchmark_varNoise > 0)
            {
                int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
                noiseMapX = new float[numnoise];
                noiseMapY = new float[numnoise];

                memset(noiseMapX, 0, sizeof(float) * numnoise);
                memset(noiseMapY, 0, sizeof(float) * numnoise);

                for (int i = 0; i < numnoise; i++)
                {
                    noiseMapX[i] = 2 * benchmark_varNoise * (rand() / (float) RAND_MAX - 0.5f);
                    noiseMapY[i] = 2 * benchmark_varNoise * (rand() / (float) RAND_MAX - 0.5f);
                }
            }

            // Real undistorting begins here:
            for (int idx = w * h - 1; idx >= 0; idx--)
            {
                // get interp. values
                float xx = remapX[idx];
                float yy = remapY[idx];

                if(benchmark_varNoise > 0)
                {
                    float deltax = getInterpolatedElement11BiCub(noiseMapX,
                                                                4 + (xx / (float) wOrg) * benchmark_noiseGridsize,
                                                                4 + (yy / (float) hOrg) * benchmark_noiseGridsize,
                                                                benchmark_noiseGridsize + 8);
                    float deltay = getInterpolatedElement11BiCub(noiseMapY,
                                                                4 + (xx / (float) wOrg) * benchmark_noiseGridsize,
                                                                4 + (yy / (float) hOrg) * benchmark_noiseGridsize,
                                                                benchmark_noiseGridsize + 8);

                    float x = idx % w + deltax;
                    float y = idx / w + deltay;

                    if(x < 0.01) x = 0.01;
                    if(y < 0.01) y = 0.01;
                    if(x > w - 1.01) x = w - 1.01;
                    if(y > h - 1.01) y = h - 1.01;

                    xx = getInterpolatedElement(remapX, x, y, w);
                    yy = getInterpolatedElement(remapY, x, y, w);
                    
                }

                if(xx < 0)
                {
                    out_data[idx] = 0;
                }
                else
                {
                    // get integer and rational parts
                    int xxi = xx;
                    int yyi = yy;
                    xx -= xxi;
                    yy -= yyi;
                    float xxyy = xx * yy;

                    // get offset and check range
                    int src_offset = xxi + yyi * wOrg;
                    if(src_offset < 0 || src_offset > (hOrg-1)*wOrg)
                    {
                        // FIXME: check why the offset is out of range in the first place.
                        // There might be other places which access invalid memory. Fix the source...
                        out_data[idx] = 0;
                    }
                    else
                    {
                        // get array base pointer
                        const float *src = in_data + src_offset;

                        // interpolate (bilinear)
                        out_data[idx] = xxyy * src[1 + wOrg]
                                        + (yy - xxyy) * src[wOrg]
                                        + (xx - xxyy) * src[1]
                                        + (1 - xx - yy + xxyy) * src[0];
                                    
                    }
                }
            }

            if(benchmark_varNoise > 0)
            {
                delete[] noiseMapX;
                delete[] noiseMapY;
            }
            
        }
        else
        {
            memcpy(result->image, photometricUndist->output->image, sizeof(float)* w * h);
        }

        /// TODO: What and why is this?
        applyBlurNoise(result->image);

        return result;
        
    }

    template ImageAndExposure *
    Undistort::undistort<unsigned char>(const MinimalImage<unsigned char> *image_raw, float exposure, double timestamp, float factor) const;

    template ImageAndExposure *
    Undistort::undistort<unsigned short>(const MinimalImage<unsigned short> *image_raw, float exposure, double timestamp, float factor) const;


    // Implement Undistort method applyBlurNoise
    void Undistort::applyBlurNoise(float *img) const 
    {
        if(benchmark_varBlurNoise == 0) return;

        // If benchmark_varBlurNoise is not zero:

        int numnoise = (benchmark_noiseGridsize + 8) * (benchmark_noiseGridsize + 8);
        float *noiseMapX = new float[numnoise];
        float *noiseMapY = new float[numnoise];
        float *blutTmp = new float[w * h];

        if(benchmark_varBlurNoise > 0)
        {
            for (int i = 0; i < numnoise; i++)
            {
                noiseMapX[i] = benchmark_varBlurNoise * (rand() / (float) RAND_MAX);
                noiseMapY[i] = benchmark_varBlurNoise * (rand() / (float) RAND_MAX);
            }
            
        }

        float gaussMap[1000];
        for (int i = 0; i < 1000; i++)
        {
            // Evaluate the Gaussian function (std.dev. = 100)
            gaussMap[i] = expf((float) (-i * i / (100.0 * 100.0)));
        }
        
        // x-blur.
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                float xBlur = getInterpolatedElement11BiCub(noiseMapX,
                                                            4 + (x / (float) w) * benchmark_noiseGridsize,
                                                            4 + (y / (float) h) * benchmark_noiseGridsize,
                                                            benchmark_noiseGridsize + 8);

                // Clamping:
                if(xBlur < 0.01) xBlur = 0.01;

                int kernelSize = 1 + (int) (1.0f + xBlur * 1.5);
                float sumW = 0;
                float sumCW = 0;
                for(int dx = 0; dx <= kernelSize; dx++)
                {
                    int gmid = 100.0f * dx / xBlur + 0.5f;
                    if(gmid > 900)
                    {
                        gmid = 900;
                    }
                    float gw = gaussMap[gmid];

                    if(x + dx > 0 && x + dx < w)
                    {
                        sumW += gw;
                        sumCW += gw * img[x + dx + y * this->w];
                    }

                    if(x - dx > 0 && x - dx < w && dx != 0)
                    {
                        sumW += gw;
                        sumCW += gw * img[x - dx + y * this->w];
                    }

                }

                blutTmp[x + y * this->w] = sumCW / sumW;
            }
        }
        
        // y-blur.
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                float yBlur = getInterpolatedElement11BiCub(noiseMapY,
                                                            4 + (x / (float) w) * benchmark_noiseGridsize,
                                                            4 + (y / (float) h) * benchmark_noiseGridsize,
                                                            benchmark_noiseGridsize + 8);

                // Clamping:
                if(yBlur < 0.01) yBlur = 0.01;

                int kernelSize = 1 + (int) (0.9f + yBlur * 2.5); // Different from xBlur!
                float sumW = 0;
                float sumCW = 0;
                for(int dy = 0; dy <= kernelSize; dy++)
                {
                    int gmid = 100.0f * dy / yBlur + 0.5f;
                    if(gmid > 900)
                    {
                        gmid = 900;
                    }
                    float gw = gaussMap[gmid];

                    if(y + dy > 0 && y + dy < h)
                    {
                        sumW += gw;
                        sumCW += gw * blutTmp[x + (y + dy) * this->w];
                    }

                    if(y - dy > 0 && y - dy < h && dy != 0)
                    {
                        sumW += gw;
                        sumCW += gw * blutTmp[x + (y - dy) * this->w];
                    }

                }

                img[x + y * this->w] = sumCW / sumW;
            }

            delete[] noiseMapX;
            delete[] noiseMapY;
        }

    }
    // Implement Undistort method makeOptimalK_crop
    void Undistort::makeOptimalK_crop()
    {
        std::cout << "Finding CROP optimal new model!" << std::endl;
        K.setIdentity();

        // 1. Stretch the center lines as far as possible, to get initial coarse guess.
        float *tgX = new float[100000];
        float *tgY = new float[100000];
        float minX = 0;
        float maxX = 0;
        float minY = 0;
        float maxY = 0;

        // x-axis:
        for (int x = 0; x < 100000; x++)
        {
            tgX[x] = (x - 50000.0f) / 10000.0f;
            tgY[x] = 0;
        }
        distortCoordinates(tgX, tgY, tgX, tgY, 100000);
        for (int x = 0; x < 100000; x++)
        {
            if(tgX[x] > 0 && tgX[x] < wOrg - 1)
            {
                if (minX == 0)
                {
                    minX = (x - 50000.0f) / 10000.0f;
                }
                maxX = (x - 50000.0f) / 10000.0f;
            }
        }
        // y-axis:
        for (int y = 0; y < 100000; y++)
        {
            tgY[y] = (y - 50000.0f) / 10000.0f;
            tgX[y] = 0;
        }
        distortCoordinates(tgX, tgY, tgX, tgY, 100000);
        for (int y = 0; y < 100000; y++)
        {
            if(tgY[y] > 0 && tgY[y] < hOrg - 1)
            {
                if (minY == 0)
                {
                    minY = (y - 50000.0f) / 10000.0f;
                }
                maxY = (y - 50000.0f) / 10000.0f;
            }
        }
        delete[] tgX;
        delete[] tgY;

        minX *= 1.01;
        maxX *= 1.01;
        minY *= 1.01;
        maxY *= 1.01;

        std::cout << "Initial range: x: " << minX << " - " << maxX << "; y: " << minY << " - " << maxY << "!" << std::endl;

        // 2. while there are invalid pixels at the border: shrink square at the side that has invalid pixels
        // if several to choose from, shrink the wider dimension.

        bool oobLeft = true, oobRight = true, oobTop = true, oobBottom = true;
        int iteration = 0;

        while (oobLeft || oobRight || oobTop || oobBottom)
        {

            oobLeft = oobRight = oobTop = oobBottom = false;

            // Check for left and right bounds:
            for (int y = 0; y < h; y++)
            {
                remapX[y * 2] = minX;
                remapX[y * 2 + 1] = maxX;
                remapY[y * 2] = remapY[y * 2 + 1] = minY + (maxY - minY) * (float) y / ((float) h - 1.0f);
            }
            distortCoordinates(remapX, remapY, remapX, remapY, 2*h);
            for (int y = 0; y < h; y++)
            {
                if(!(remapX[2 * y] > 0 && remapX[2 * y] < wOrg - 1))
                {
                    oobLeft = true;
                }
                if(!(remapX[2 * y + 1] > 0 && remapX[2 * y + 1] < wOrg - 1))
                {
                    oobRight = true;
                }
            }
            // Check for top and bottom bounds:
            for (int x = 0; x < w; x++)
            {
                remapY[x * 2] = minY;
                remapY[x * 2 + 1] = maxY;
                remapX[x * 2] = remapX[x * 2 + 1] = minX + (maxX - minX) * (float) x / ((float) w - 1.0f);
            }
            distortCoordinates(remapX, remapY, remapX, remapY, 2*w);
            for (int x = 0; x < w; x++)
            {
                if(!(remapY[2 * x] > 0 && remapY[2 * x] < hOrg - 1))
                {
                    oobTop = true;
                }
                if(!(remapY[2 * x + 1] > 0 && remapY[2 * x + 1] < hOrg - 1))
                {
                    oobBottom = true;
                }
            }

            if((oobLeft || oobRight) && (oobTop || oobBottom))
            {
                if((maxX - minX) > (maxY - minY))
                {
                    oobBottom = oobTop = false; // only shrink left/right
                }
                else
                {
                    oobLeft = oobRight = false; // only shrink top/bottom
                }
                
            }

            if (oobLeft) minX *= 0.995;
            if (oobRight) maxX *= 0.995;
            if (oobTop) minY *= 0.995;
            if (oobBottom) maxY *= 0.995;

            iteration++;

            std::cout << "iteration "<< iteration << ": range: x: " << minX << " - " << maxX << "; y: " << minY << " - " << maxY << "!" << std::endl;

            if(iteration > 500)
            {
                std::cout << "FAILED TO COMPUTE GOOD CAMERA MATRIX - SOMETHING IS SERIOUSLY WRONG. ABORTING " << std::endl;
                exit(1);
            }
            
        }

        // Use calculations from above to update camera calibration matrix K:
        K(0, 0) = ((float) w - 1.0f) / (maxX - minX);
        K(1, 1) = ((float) h - 1.0f) / (maxY - minY);
        K(0, 2) = -minX * K(0, 0);
        K(1, 2) = -minY * K(1, 1);
        
    }
    // Implement Undistort method makeOptimalK_full
    void Undistort::makeOptimalK_full()
    {
        /// TODO: Why is this even necessary?
        assert(false);
    }

    // Implement Undistort method readFromFile
    void Undistort::readFromFile(const char *configFileName, int nPars, std::string prefix)
    {
        // read the camera.txt (configFile)
        photometricUndist = 0;
        valid = false;
        passthrough = false;
        remapX = 0;
        remapY = 0;

        float outputCalibration[5];

        parsOrg = VecX(nPars);

        // read parameters from camera.txt
        std::ifstream infile(configFileName);
        assert(infile.good());

        // Create strings for each of the four lines:
        std::string l1, l2, l3, l4;

        std::getline(infile, l1); // camera parameters
        std::getline(infile, l2); // original image resolution
        std::getline(infile, l3); // "crop","full", or parameters
        std::getline(infile, l4); // resulting image resolution

        // l1 & l2
        if(nPars == 5) // fov and pinhole model
        {
            char buf[1000];
            snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf", prefix.c_str());

            if(std::sscanf(l1.c_str(), buf, &parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4]) == 5
            && std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2)
            {
                std::cout << "Input resolution: " << wOrg << " " << hOrg << std::endl;
                std::cout << "In: " << parsOrg[0] << " " << parsOrg[1] << " " 
                << parsOrg[2] << " " << parsOrg[3] << " " << parsOrg[4] << std::endl;
            }
            else
            {
                std::cout << "Failed to read camera calibration (invalid format?)" << std::endl 
                << "Calibration file: " << configFileName << std::endl;
                infile.close();
                return;
            }
            
        }
        else if(nPars == 8) // KB, equidistant and radtan camera model
        {
            char buf[1000];
            snprintf(buf, 1000, "%s%%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf %%lf", prefix.c_str());

            if(std::sscanf(l1.c_str(), buf, &parsOrg[0], &parsOrg[1], &parsOrg[2], &parsOrg[3], &parsOrg[4], 
                                            &parsOrg[5], &parsOrg[6], &parsOrg[7]) == 8
            && std::sscanf(l2.c_str(), "%d %d", &wOrg, &hOrg) == 2)
            {
                std::cout << "Input resolution: " << wOrg << " " << hOrg << std::endl;
                std::cout << "In: " << prefix.c_str() << " " << parsOrg[0] << " " << parsOrg[1] << " " 
                << parsOrg[2] << " " << parsOrg[3] << " " << parsOrg[4] << " " << parsOrg[5] << " " 
                << parsOrg[6] << " " << parsOrg[7] << std::endl;
            }
            else
            {
                std::cout << "Failed to read camera calibration (invalid format?)" << std::endl 
                << "Calibration file: " << configFileName << std::endl;
                infile.close();
                return;
            }
        }
        else
        {
            std::cout << "Called with invalid number of parameters.... forgot to implement me?" << std::endl;
            infile.close();
            return;
        }
        
        // Update the parsed parameters based on the image dimension
        if(parsOrg[2] < 1 && parsOrg[3] < 1)
        {
            std::cout << std::endl << std::endl << 
            "Found fx=" << parsOrg[0] << ", fy=" << parsOrg[1] <<  
            ", cx=" << parsOrg[2] <<  ", cy=" << parsOrg[3] << std::endl <<
            "I'm assuming this is the \"relative\" calibration file format, "
            "and will rescale this by image width / height to fx=" << parsOrg[0] * wOrg << 
            ", fy=" << parsOrg[1] * hOrg << 
            ", cx=" << parsOrg[2] * wOrg - 0.5 << ", cy=" << parsOrg[3] * hOrg - 0.5 << "." << 
            std::endl << std::endl;

            // rescale and subtract (offset) 0.5.
            // the 0.5 is because I'm assuming the calibration is given such that the pixel at (0,0)
            // contains the integral over intensity over [0,0]-[1,1], whereas I assume the pixel (0,0)
            // to contain a sample of the intensity at [0,0], which is best approximated by the integral over
            // [-0.5,-0.5]-[0.5,0.5]. Thus the shift by -0.5.

            parsOrg[0] = parsOrg[0] * wOrg;
            parsOrg[1] = parsOrg[1] * hOrg;
            parsOrg[2] = parsOrg[2] * wOrg - 0.5; /// TODO: What does that has to do with intensities?!
            parsOrg[3] = parsOrg[3] * hOrg - 0.5; /// TODO: What does that has to do with intensities?!
        }

        // l3 can contain "crop", "full", "none", or 5 outputCalibration parameters:
        if(l3 == "crop")
        {
            outputCalibration[0] = -1;
            std::cout << "Out: Rectify Crop" << std::endl;
        }
        else if(l3 == "full")
        {
            outputCalibration[0] = -2;
            std::cout << "Out: Rectify Full" << std::endl;
        }
        else if(l3 == "none")
        {
            outputCalibration[0] = -3;
            std::cout << "Out: No Rectification" << std::endl;
        }
        else if(std::sscanf(l3.c_str(), "%f %f %f %f %f", 
        &outputCalibration[0], &outputCalibration[1], &outputCalibration[2], &outputCalibration[3], &outputCalibration[4]) == 5)
        {
            std::cout << "Out: " << outputCalibration[0] << " "
                                 << outputCalibration[1] << " "
                                 << outputCalibration[2] << " "
                                 << outputCalibration[3] << " "
                                 << outputCalibration[4] << std::endl;
        }
        else
        {
            std::cout << "Out: Failed to read output pars... not rectifying." << std::endl;
            infile.close();
            return;
        }
        
        // l4: Output resolution
        if( std::sscanf(l4.c_str(), "%d %d", &w, &h) == 2)
        {
            if(benchmarkSetting_width != 0)
            {
                w = benchmarkSetting_width;
                if(outputCalibration[0] == -3)
                {  
                    // crop instead of none, since probably resolution changed
                    outputCalibration[0] = -1;
                }
            }
            if(benchmarkSetting_height != 0)
            {
                h = benchmarkSetting_height;
                if(outputCalibration[0] == -3)
                {  
                    // crop instead of none, since probably resolution changed
                    outputCalibration[0] = -1;
                }
            }

            std::cout << "Output resolution: " << w << "x" << h << std::endl;
        }
        else
        {
            std::cout << "Out: Failed to read output resolution... not rectifying." << std::endl;
        }

        remapX = new float[w * h];
        remapY = new float[w * h];

        if(outputCalibration[0] == -1)
        {
            makeOptimalK_crop();
        }
        else if (outputCalibration[0] == -2)
        {
            makeOptimalK_full();
        }
        else if (outputCalibration[0] == -3)
        {
            if(w != wOrg || h != hOrg)
            {
                std::cout << "ERROR: rectification mode none requires input and output dimensions to match" << 
                std::endl << std::endl;
                exit(1);
            }
            K.setIdentity();
            K(0, 0) = parsOrg[0];
            K(1, 1) = parsOrg[1];
            K(0, 2) = parsOrg[2];
            K(1, 2) = parsOrg[3];
            passthrough = true;
            
        }
        else
        {
            // Check the cx and cy parts to be <= 1:
            if(outputCalibration[2] > 1 || outputCalibration[3] > 1)
            {
                std::cout  << std::endl << std::endl << std::endl 
                << "WARNING: given output calibration (" << 
                outputCalibration[0] << " " << outputCalibration[1] << " " << 
                outputCalibration[2] << " " << outputCalibration[3] << 
                ") seems wrong. It needs to be relative to image width / height!" << 
                std::endl << std::endl << std::endl;
            }

            K.setIdentity();
            K(0, 0) = outputCalibration[0] * w;
            K(1, 1) = outputCalibration[1] * h;
            K(0, 2) = outputCalibration[2] * w - 0.5;
            K(1, 2) = outputCalibration[3] * h - 0.5;

        }

        if(benchmarkSetting_fxfyfac != 0)
        {
            K(0, 0) = fmax(benchmarkSetting_fxfyfac, (float) K(0, 0));
            K(1, 1) = fmax(benchmarkSetting_fxfyfac, (float) K(1, 1));
            // cannot pass through when fx / fy have been overwritten:
            passthrough = false; 
        }

        // fill the remapX/Y arrays with ascending numbers (from 0 to w/h):
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                remapX[x + y * w] = x;
                remapY[x + y * w] = y;
            }
        }
        
        distortCoordinates(remapX, remapY, remapX, remapY, h * w);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                // make rounding resistant.
                float ix = remapX[x + y * w];
                float iy = remapY[x + y * w];

                if(ix == 0) ix = 0.001;
                if(iy == 0) iy = 0.001;
                if(ix == wOrg - 1) ix = wOrg - 1.001;
                if(iy == hOrg - 1) ix = hOrg - 1.001; /// TODO: Check if there is a typo (ix should be iy?)

                if(ix > 0 && iy > 0 && ix < wOrg - 1 && iy < wOrg - 1) /// TODO: Check if there is a typo ("iy < wOrg - 1" should be "iy < hOrg - 1"?)
                {
                    remapX[x + y * w] = ix;
                    remapY[x + y * w] = iy;
                }
                else
                {
                    remapX[x + y * w] = -1;
                    remapY[x + y * w] = -1;
                }
            }
        }

        valid = true;

        std::cout << std::endl << "Rectified Camera Matrix: " << std::endl 
        << K << std::endl << std::endl;
        
    }

    // Implement UndistortFOV constructor
    UndistortFOV::UndistortFOV(const char *configFileName, bool noprefix)
    {
        std::cout << "Creating FOV undistorter" << std::endl;

        if(noprefix)
        {
            readFromFile(configFileName, 5);
        }
        else
        {
            readFromFile(configFileName, 5, "FOV ");
        }
    }
    // Implement UndistortFOV destructor
    UndistortFOV::~UndistortFOV()
    {

    }
    // Implement UndistortFOV method distortCoordinates
    void UndistortFOV::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
    {
        float dist = parsOrg[4];
        float d2t = 2.0f * tanf(dist / 2.0f);

        // current camera parameters
        float fx = parsOrg[0];
        float fy = parsOrg[1];
        float cx = parsOrg[2];
        float cy = parsOrg[3];

        float ofx = K(0, 0);
        float ofy = K(1, 1);
        float ocx = K(0, 2);
        float ocy = K(1, 2);

        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;

            float r = sqrtf(ix * ix + iy * iy);
            float fac = (r == 0 || dist == 0) ? 1 : atanf(r * d2t) / (dist * r);

            ix = fx * fac * ix + cx;
            iy = fy * fac * iy + cy;

            out_x[i] = ix;
            out_y[i] = iy;
        }
    }

    // Implement UndistortRadTan constructor
    UndistortRadTan::UndistortRadTan(const char *configFileName, bool noprefix)
    {
        std::cout << "Creating RadTan undistorter" << std::endl;

        if(noprefix)
        {
            readFromFile(configFileName, 8);
        }
        else
        {
            readFromFile(configFileName, 8, "RadTan ");
        }
    }
    // Implement UndistortRadTan destructor
    UndistortRadTan::~UndistortRadTan()
    {

    }
    // Implement UndistortRadTan method distortCoordinates
    void UndistortRadTan::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
    {
        // RADTAN (as used in OpenCV)
        float fx = parsOrg[0];
        float fy = parsOrg[1];
        float cx = parsOrg[2];
        float cy = parsOrg[3];
        float k1 = parsOrg[4];
        float k2 = parsOrg[5];
        float r1 = parsOrg[6];
        float r2 = parsOrg[7];

        float ofx = K(0, 0);
        float ofy = K(1, 1);
        float ocx = K(0, 2);
        float ocy = K(1, 2);

        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];

            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;

            float mx2_u = ix * ix;
            float my2_u = iy * iy;
            float mxy_u = ix * iy;

            float rho2_u = mx2_u + my2_u;
            float rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;

            float x_dist = ix + ix * rad_dist_u + 2.0 * r1 * mxy_u + r2 * (rho2_u + 2.0 * mx2_u);
            float y_dist = iy + iy * rad_dist_u + 2.0 * r2 * mxy_u + r1 * (rho2_u + 2.0 * my2_u);

            float ox = fx * x_dist + cx;
            float oy = fy * y_dist + cy;

            out_x[i] = ox;
            out_y[i] = oy;
        }
    }

    // Implement UndistortEquidistant constructor
    UndistortEquidistant::UndistortEquidistant(const char *configFileName, bool noprefix)
    {
        std::cout << "Creating Equidistat undistorter" << std::endl;

        if(noprefix)
        {
            readFromFile(configFileName, 8);
        }
        else
        {
            readFromFile(configFileName, 8, "EquiDistant ");
        }
    }
    // Implement UndistortEquidistant destructor
    UndistortEquidistant::~UndistortEquidistant()
    {

    }
    // Implement UndistortEquidistant method distortCoordinates
    void UndistortEquidistant::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
    {
        // EQUIDISTANT Distortion model
        float fx = parsOrg[0];
        float fy = parsOrg[1];
        float cx = parsOrg[2];
        float cy = parsOrg[3];
        float k1 = parsOrg[4];
        float k2 = parsOrg[5];
        float k3 = parsOrg[6];
        float k4 = parsOrg[7];

        float ofx = K(0, 0);
        float ofy = K(1, 1);
        float ocx = K(0, 2);
        float ocy = K(1, 2);

        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];

            // Equidistant Model
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;

            float r = sqrt(ix * ix + iy * iy);
            float theta = atan(r);
            float theta2 = theta * theta;
            float theta4 = theta2 * theta2;
            float theta6 = theta4 * theta2;
            float theta8 = theta4 * theta4;
            float thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8);

            float scaling = (r > 1e-8) ? thetad / r : 1.0;

            float ox = fx * ix * scaling + cx;
            float oy = fy * iy * scaling + cy;

            out_x[i] = ox;
            out_y[i] = oy;
        }
    }

    // Implement UndistortKB constructor
    UndistortKB::UndistortKB(const char *configFileName, bool noprefix)
    {
        std::cout << "Creating KannalaBrandt undistorter" << std::endl;

        if(noprefix)
        {
            readFromFile(configFileName, 8);
        }
        else
        {
            readFromFile(configFileName, 8, "KannalaBrandt ");
        }
    }
    // Implement UndistortKB destructor
    UndistortKB::~UndistortKB()
    {

    }
    // Implement UndistortKB method distortCoordinates
    void UndistortKB::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
    {
        // KannalaBrandt Distortion model
        const float fx = parsOrg[0];
        const float fy = parsOrg[1];
        const float cx = parsOrg[2];
        const float cy = parsOrg[3];
        const float k0 = parsOrg[4];
        const float k1 = parsOrg[5];
        const float k2 = parsOrg[6];
        const float k3 = parsOrg[7];

        const float ofx = K(0, 0);
        const float ofy = K(1, 1);
        const float ocx = K(0, 2);
        const float ocy = K(1, 2);

        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];

            // Kannala-Brandt Model
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;

            const float Xsq_plus_Ysq = ix * ix + iy * iy;
            const float sqrt_Xsq_Ysq = sqrtf(Xsq_plus_Ysq);
            
            const float theta = atan2f(sqrt_Xsq_Ysq, 1);
            const float theta2 = theta * theta;
            const float theta3 = theta2 * theta;
            const float theta5 = theta3 * theta2;
            const float theta7 = theta5 * theta2;
            const float theta9 = theta7 * theta2;
            const float r = theta + k0 * theta3 + k1 * theta5 + k2 * theta7 + k3 * theta9;

            if(sqrt_Xsq_Ysq < 1e-6)
            {
                out_x[i] = fx * ix + cx;
                out_y[i] = fy * iy + cy;
            }
            else
            {
                out_x[i] = (r / sqrt_Xsq_Ysq) * fx * ix + cx;
                out_y[i] = (r / sqrt_Xsq_Ysq) * fy * iy + cy;
            }
        }
    }

    // Implement UndistortPinhole constructor
    UndistortPinhole::UndistortPinhole(const char *configFileName, bool noprefix)
    {
        std::cout << "Creating Pinhole undistorter" << std::endl;

        if(noprefix)
        {
            readFromFile(configFileName, 5);
        }
        else
        {
            readFromFile(configFileName, 5, "Pinhole ");
        }
    }
    // Implement UndistortPinhole destructor
    UndistortPinhole::~UndistortPinhole()
    {

    }
    // Implement UndistortPinhole method distortCoordinates
    void UndistortPinhole::distortCoordinates(float *in_x, float *in_y, float *out_x, float *out_y, int n) const
    {
        // current camera parameters
        float fx = parsOrg[0];
        float fy = parsOrg[1];
        float cx = parsOrg[2];
        float cy = parsOrg[3];

        float ofx = K(0, 0);
        float ofy = K(1, 1);
        float ocx = K(0, 2);
        float ocy = K(1, 2);

        for (int i = 0; i < n; i++)
        {
            float x = in_x[i];
            float y = in_y[i];
            float ix = (x - ocx) / ofx;
            float iy = (y - ocy) / ofy;

            ix = fx * ix + cx;
            iy = fy * iy + cy;

            out_x[i] = ix;
            out_y[i] = iy;
        }
    }
}