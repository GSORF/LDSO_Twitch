#include <iostream>
#include <thread>

#include <clocale>
#include <csignal>
#include <cstdlib>
#include <cstdio> // TODO: I think that can be removed
#include <cmath> // Note: Not from the original LDSO code, added due to use of "fabs()"
#include <unistd.h>
#include <sys/time.h>

#include <glog/logging.h>

// TODO: add includes ("Fullsystem")
// #include "frontend/FullSystem.h"
#include "DatasetReader.h"



// TODO: create global variables
using namespace std;
using namespace ldso;

std::string vignette = "../VSLAM_Datasets/sequence_45/vignette.png";
std::string gammaCalib = "../VSLAM_Datasets/sequence_45/pcalib.txt"; // photometric calibration
std::string calib = "../VSLAM_Datasets/sequence_45/camera.txt"; // geometric calibration
std::string source = "../VSLAM_Datasets/sequence_45/";

std::string output_file = "./results.txt";
std::string vocPath = "./vocab/orbvoc.dbow3";

double rescale = 1;
bool reversePlay = false;
bool disableROS = false;
int startIdx = 0; // The index of the first image
int endIdx = 100000; // Index of the last image
bool prefetch = false;
float playbackSpeed = 0; // 0 is for play as fast as possible, otherwise factor on timestamps
bool preload = false;
bool useSampleOutput = false;


// Select proper preset (number of feature points, calibration mode, etc.)
void settingsDefault(int preset)
{
    cout << "==== PRESET Settings: ====" << endl;
    if(preset == 0 || preset == 1)
    {
        cout << "DEFAULT settings:" << endl << "- " << (preset==0?"no":"1x") 
        << " real-time enforcing "<< endl
        << "- 2000 active points" << endl 
        << "- 5-7 active frames" << endl 
        << "- 1-6 LM iterations each KF" << endl 
        << "- original image resolution" << endl;

        playbackSpeed = (preset == 0 ? 0 : 1);
        preload = preset == 1;
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_minOptIterations = 1;
        setting_maxOptIterations = 6;

        setting_logStuff = false;
    }

    if(preset == 2 || preset == 3)
    {
        cout << "FAST settings:" << endl << "- " << (preset==2?"no":"5x") 
        << " real-time enforcing "<< endl
        << "- 800 active points" << endl 
        << "- 4-6 active frames" << endl 
        << "- 1-4 LM iterations each KF" << endl 
        << "- 424 x 320 image resolution" << endl;

        playbackSpeed = (preset == 2 ? 0 : 5);
        preload = preset == 3;

        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_minOptIterations = 1;
        setting_maxOptIterations = 4;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }
    cout << "======================" << endl;
}

// TODO: add parsing functions
void parseArgument(char *arg)
{
    int option;
    float foption;
    char buf[1000];

    if( 1 == sscanf(arg, "sampleoutput=%d", &option))
    {
        if(option == 1)
        {
            useSampleOutput = true;
            cout << "USING SAMPLE OUTPUT WRAPPER!" << endl;
        }
        return;
    }

    if( 1 == sscanf(arg, "quiet=%d", &option))
    {
        if(option == 1)
        {
            setting_debugout_runquiet = true;
            cout << "QUIET MODE, I'll shut up!" << endl;
        }
        return;
    }

    if( 1 == sscanf(arg, "preset=%d", &option))
    {
        settingsDefault(option);
        return;
    }

    if( 1 == sscanf(arg, "rec=%d", &option))
    {
        if(option == 0)
        {
            disableReconfigure = true;
            cout << "DISABLE RECONFIGURE!" << endl;
        }
        return;
    }

    if( 1 == sscanf(arg, "noros=%d", &option))
    {
        if(option == 1)
        {
            disableROS = true;
            disableReconfigure = true;
            cout << "DISABLE ROS (AND RECONFIGURE)!" << endl;
        }
        return;
    }

    if( 1 == sscanf(arg, "nolog=%d", &option))
    {
        if(option == 1)
        {
            setting_logStuff = false;
            cout << "DISABLE LOGGING!" << endl;
        }
        return;
    }

    if( 1 == sscanf(arg, "reversePlay=%d", &option))
    {
        if(option == 1)
        {
            reversePlay = true;
            cout << "REVERSE!" << endl;
        }
        return;
    }
    if( 1 == sscanf(arg, "nogui=%d", &option))
    {
        if(option == 1)
        {
            disableAllDisplay = true;
            cout << "REVERSE!" << endl;
        }
        return;
    }
    if( 1 == sscanf(arg, "nomt=%d", &option))
    {
        if(option == 1)
        {
            multiThreading = false;
            cout << "NO MultiThreading!" << endl;
        }
        return;
    }
    if( 1 == sscanf(arg, "prefetch=%d", &option))
    {
        if(option == 1)
        {
            prefetch = true;
            cout << "PREFETCH!" << endl;
        }
        return;
    }
    if( 1 == sscanf(arg, "start=%d", &option))
    {
        startIdx = option;
        cout << "START AT " << startIdx <<"!" << endl;
        return;
    }
    if( 1 == sscanf(arg, "end=%d", &option))
    {
        endIdx = option;
        cout << "END AT " << endIdx <<"!" << endl;
        return;
    }

    if( 1 == sscanf(arg, "loopclosing=%d", &option))
    {
        if(option == 1)
        {
            setting_enableLoopClosing = true;
            
        }
        else
        {
            setting_enableLoopClosing = false;
        }
        cout << "Loopclosing!" << (setting_enableLoopClosing ? "enabled":"disabled") << endl;
        return;
    }

    if( 1 == sscanf(arg, "files=%s", buf))
    {
        source = buf;
        cout << "loading data from " << source.c_str() <<"!" << endl;
        return;
    }

    if( 1 == sscanf(arg, "vocab=%s", buf))
    {
        vocPath = buf;
        cout << "loading vocabulary from " << vocPath.c_str() <<"!" << endl;
        return;
    }
    if( 1 == sscanf(arg, "calib=%s", buf))
    {
        calib = buf;
        cout << "loading calibration from " << calib.c_str() <<"!" << endl;
        return;
    }

    if( 1 == sscanf(arg, "vignette=%s", buf))
    {
        vignette = buf;
        cout << "loading vignette from " << vignette.c_str() <<"!" << endl;
        return;
    }
    if( 1 == sscanf(arg, "gamma=%s", buf))
    {
        gammaCalib = buf;
        cout << "loading gammaCalib from " << gammaCalib.c_str() <<"!" << endl;
        return;
    }

    if( 1 == sscanf(arg, "rescale=%f", &foption))
    {
        playbackSpeed = foption;
        cout << "PLAYBACK SPEED " << playbackSpeed <<"!" << endl;
        return;
    }
    if( 1 == sscanf(arg, "output=%s", buf))
    {
        output_file = buf;
        LOG(INFO) << "Output set to " << output_file <<"!" << endl;
        return;
    }

    if( 1 == sscanf(arg, "save=%d", &option))
    {
        if(option == 1)
        {
            debugSaveImages = true;
            
            // TODO: Do we really need these "interesting" conditional statements?
            if (42 == system("rm -rf images_out"))
            {
                cout << "System call returned 42- what are the odds?. This is only here to shut up the compiler." << endl;
            }
            if (42 == system("mkdir images_out"))
            {
                cout << "System call returned 42- what are the odds?. This is only here to shut up the compiler." << endl;
            }
            if (42 == system("rm -rf images_out"))
            {
                cout << "System call returned 42- what are the odds?. This is only here to shut up the compiler." << endl;
            }
            if (42 == system("mkdir images_out"))
            {
                cout << "System call returned 42- what are the odds?. This is only here to shut up the compiler." << endl;
            }
            cout << "SAVE IMAGES!" << endl;
        }

        return;
    }

    if( 1 == sscanf(arg, "mode=%d", &option))
    {
        if(option == 0)
        {
            cout << "PHOTOMETRIC MODE WITH CALIBRATION!" << endl;

        }
        if(option == 1)
        {
            cout << "PHOTOMETRIC MODE WITHOUT CALIBRATION!" << endl;
            setting_photometricCalibration = 0;
            setting_affineOptModeA = 0; // -1: fix. >=0: optimize (with prior, if > 0)
            setting_affineOptModeB = 0; // -1: fix. >=0: optimize (with prior, if > 0)
        }
        if(option == 2)
        {
            cout << "PHOTOMETRIC MODE WITH PERFECT IMAGES!" << endl;
            setting_photometricCalibration = 0;
            setting_affineOptModeA = -1; // -1: fix. >=0: optimize (with prior, if > 0)
            setting_affineOptModeB = -1; // -1: fix. >=0: optimize (with prior, if > 0)
            setting_minGradHistAdd = 3;
        }
        return;
    }

    cout << "could not parse argument!" << arg << "!!!!" << endl;
}


int main(int argc, char **argv)
{

    FLAGS_colorlogtostderr = true;

    // Parsing the arguments:
    for (int i = 1; i < argc; i++)
    {
        cout << i << ".: Parsing argument = " << argv[i] << endl;

        parseArgument(argv[i]);
    }

    // Check setting conflicts

    if(setting_enableLoopClosing && (setting_pointSelection != 1))
    {
        LOG(ERROR) << "Loop closing is enabled but point selection strategy is not set to LDSO, use setting_pointSelection=1! please!" << endl;
        exit(-1);
    }
    

    if(setting_showLoopClosing == true)
    {
        LOG(WARNING) << "show loop closing results. The program will be paused when any loop is found." << endl;
    }
    

    shared_ptr<ImageFolderReader> reader(new ImageFolderReader(ImageFolderReader::TUM_MONO, source, calib, gammaCalib, vignette));
    
    // TODO: reader->setGlobalCalibration();

    
    if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
    {
        LOG(ERROR) << "ERROR: Don't have photometric calibration. Need to use commandline options mode=1 or mode=2";
        exit(1);
    }
    

    int lstart = startIdx;
    int lend = endIdx;
    int linc = 1;

    if(reversePlay)
    {
        LOG(INFO) << "REVERSE!!!!";
        lstart = endIdx - 1;
        /* TODO: 
        if(lstart >= reader->getNumImages())
        {
            lstart = reader->getNumImages() - 1;
        }
        */
        lend = startIdx;
        linc = -1;
    }

    // Load the ORB-Vocabulary (used for loop closure):
    /* TODO:
    shared_ptr<ORBVocabulary> voc(new ORBVocabulary());
    voc->load(vocPath);
    */
    
    // Initialize the "FullSystem" which is the core of our VSLAM (Visual Simultaneous Localization And Mapping):
    /* TODO:
    shared_ptr<FullSystem> fullSystem(new FullSystem(voc));
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
    fullSystem->linearizeOperation = (playbackSpeed == 0);
    */

    // Initialize the Graphical User Interface (GUI) using the Pangolin library:
    /* TODO:
    shared_ptr<PangolinDSOViewer> viewer = nullptr;
    if(!disableAllDisplay)
    {
        viewer = shared_ptr<PangolinDSOViewer>(new PangolinDSOViewer(wG[0],hG[0], false));
        fullSystem->setViewer(viewer);
    }
    else
    {
        LOG(INFO) << "visualization is disabled!" << endl;
    }
    */

    // This is the main loop which runs on a separate thread:
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;

        // Prepare the correct timestamps for each image:
        /* TODO: 
        for (int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i+= linc)
        {
            idsToPlay.push_back(i);

            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double) 0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
                timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / playbackSpeed);
            }
        }
        */
        
        // Load all the images we need in a data structure in our memory
        /* TODO: 
        std::vector<ImageAndExposure *> preloadedImages;
        if(preload)
        {
            cout << "LOADING ALL IMAGES!" << endl;
            for (int ii = 0; ii < (int) idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }
        */


        // Start a stop watch in order to measure performance on a dataset:
        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset = 0;

        // Here is the main loop of the VSLAM implementation - loop through all images:
        for (int ii = 0; ii < (int) idsToPlay.size(); ii++)
        {
            while (setting_pause == true)
            {
                usleep(5000);
            }
            

            /* TODO: 
            if(!fullSystem->initialized)
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }
            */

            int i = idsToPlay[ii];

            /* TODO:
            ImageAndExposure *img; // The main datatype for an image
            if(preload)
            {
                img = preloadedImages[ii];
            }
            else
            {
                img = reader->getImage(i);
            }
            */

            bool skipFrame = false;
            if(playbackSpeed != 0)
            {
                struct timeval tv_now;
                gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));
                
                if(sSinceStart < timesToPlayAt[ii])
                {
                    // TODO: Can be removed, as comment anyway
                    // usleep((int) ((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
                }
                else if (sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
                {
                    cout << "SKIPFRAME " << ii << "(play at " << timesToPlayAt[ii] << ", now it is "<< sSinceStart << ")!" << endl;
                    skipFrame = true;
                }
            }

            if(!skipFrame)
            {
                // Here the "real nice magic" is happening! 
                /// TODO: fullSystem->addActiveFrame(img, i);
            }
            /// TODO: delete img;

            // Cleaning up
            /* TODO: 
            if(fullSystem->initFailed || setting_fullResetRequested)
            {
                if(ii < 250 || setting_fullResetRequested)
                {
                    LOG(INFO) << "Init failed, RESETTING!";
                    fullSystem = shared_ptr<FullSystem>(new FullSystem(voc));
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);
                    if(viewer)
                    {
                        viewer->reset();
                        sleep(1);
                        fullSystem->setViewer(viewer);
                    }
                    setting_fullResetRequired = false;
                }
            }
            */

            /* TODO
            if(fullSystem->isLost)
            {
                LOG(INFO) << "Lost!";
                break;
            }
            */
            
        }

        /// TODO: fullSystem->blockUntilMappingIsFinished();

        // End the stop watch:
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

        // Statistics: Useful for data analysis
        /// TODO: fullSystem->printResult(output_file, true);
        /// TODO: fullSystem->printResult(output_file + ".noloop", false);


        /* TODO: (SegmentationFault)
        int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
        double numSecondsProcessed = 1234.0; // TODO: fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);

        cout << endl << "=================" 
        << endl << numFramesProcessed << " Frames (" << numFramesProcessed / numSecondsProcessed << " fps)"
        << endl << MilliSecondsTakenSingle / numFramesProcessed << "ms per frame (single core);" 
        << endl << MilliSecondsTakenMT / (float) numFramesProcessed << "ms per frame (multi core);" 
        << endl << 1000 / (MilliSecondsTakenSingle / numFramesProcessed) << "x (single core);" 
        << endl << 1000 / (MilliSecondsTakenMT / numFramesProcessed) << "x (multi core);" 
        << endl << "===============" << endl;
        */

        /* TODO: 
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC * reader->getNumImages()) << " " 
            << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) / (float) reader->getNumImages() << endl;
            tmlog.flush();
            tmlog.close();
        }
        */


    });

    /* TODO: 
    if(viewer)
    {
        viewer->run(); // mac os should keep this in main thread.
    }
    */

    runthread.join(); // This will take a while...

    // TODO: viewer->saveAsPLYFile("./pointcloud.ply");
    LOG(INFO) << "EXIT NOW!";
    
    return 0;
}




