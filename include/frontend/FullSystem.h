#pragma once
#ifndef LDSO_TWITCH_FULL_SYSTEM_H_
#define LDSO_TWITCH_FULL_SYSTEM_H_

// Standard template libraries
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

// Our own data structures
#include "Frame.h"
#include "Point.h"
#include "Feature.h"
#include "Camera.h"
#include "ImageAndExposure.h"
#include "DSOViewer.h"
#include "Map.h"
#include "FeatureDetector.h"
#include "FeatureMatcher.h"
#include "PixelSelector2.h"

#include "internal/IndexThreadReduce.h"
#include "LoopClosing.h"

// Namespace used:
using namespace std;
using namespace ldso;
using namespace ldso::internal;

const int MAX_ACTIVE_FRAMES = 100;

namespace ldso {

    class CoarseTracker;

    class CoarseInitializer;

    class ImageAndExposure;

    class CoarseDistanceMap;

    namespace internal {

        class PointFrameResidual;

        class ImmaturePoint; // We will have immature, active, outofbounds, ... kinds of points

        class EnergyFunctional;

        struct ImmaturePointTemporaryResidual;
    }

    /* 
    
    FullSystem is the top-level interface of DSO system
    call addActiveFrame() to track an image
    
    */

   class FullSystem {
    // Methods:
    public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // Constructor
    FullSystem(shared_ptr<ORBVocabulary> voc);

    // Destructor
    ~FullSystem();

    // Adds a new frame, and creates point & residual structs.
    void addActiveFrame(ImageAndExposure *image, int id);

    // block the tracking until mapping is finished, return when mapping is finished.
    void blockUntilMappingIsFinished();

    /*
    optimize the system, by calling solveSystem()
    @param mnumOptIts number of iterations, will be changed if active frames are less than 4
    @return average energy
    */
    float optimize(int mnumOptIts);

    /*
    Set the gamma function into camera calib hessian
    @param BInv
    */
    void setGammaFunction(float *BInv);

    /*
    Shutdown the system (called when resetting);
    Will stop the loop closing thread (if it is enabled)
    */
    void shutDown();

    // save all information into a binary file.
    bool saveAll(const string &filename);

    // load all information from a binary file.
    bool loadAll(const string &filename);


    // state variables
    bool isLost = false; // if system is lost, note that dso CANNOT recover from lost
    bool initFailed = false; // initialization failed?
    bool initialized = false; // initialized?
    bool linearizeOperation = true; // this is something that controls 
    // if the optimization runs in a single thread
    // but why it is called linearizeOperation...?

    shared_ptr<CoarseDistanceMap> GetDistanceMap()
    {
        return coarseDistanceMap;
    }

    vector<shared_ptr<Frame>> GetActiveFrames()
    {
        unique_lock<mutex> lck(framesMutex);
        return frames;
    }

    void RefreshGUI()
    {
        if(viewer)
        {
            viewer->refreshAll();
        }
    }

    private:

    // mainPipelineFunctions
    // note: track and trace is different.
    // track is used in every frame to estimate its pose
    // and trace is used to record the immature point tracking status

    // marginalizes a frame. Drops / marginalizes points & residuals.
    // residuals will be dropped, but we still have this frame in the memory
    void marginalizeFrame(shared_ptr<Frame> &frame);

    /**
     * @brief 
     * track a new frame and estimate the pose (Special Euclidean Group (SE3) ) 
     * and affine light parameters (a, b)
     * @param fh new frame
     * @return a 4-vector which has some tracking status information,
     * used to judge if we need a keyframe
     */
    Vec4 trackNewCoarse(shared_ptr<FrameHessian> fh);

    /**
     * @brief 
     * trace immature points into new frames, maybe keyframe or not-keyframe, 
     * to update the immature point status.
     * fh's pose should be estimated, otherwise trace does not make sense
     * @param fh
     */
    void traceNewCoarse(shared_ptr<FrameHessian> fh);

    /*
    activate point, turn the immature into real points and insert residuals into backend
    called in making keyframes
    */
    void activatePointsMT();

    /*
    reductor for activating points
    will call optimizeImmaturePoint in a multi-thread way
    */
    void activatePointsMT_Reductor(std::vector<shared_ptr<PointHessian>> *optimized,
                                    std::vector<shared_ptr<ImmaturePoint>> *toOptimize,
                                    int min, int max, Vec10 *stats, int tid);

    /*
    optimize an immature point, if the idepth is good, create a map point from this immature point
    @param point the immature point, must have a host feature
    @param minObs minimal good residual required.
    if the immature point's good residual is less than minObs, it will be marked as an outlier.
    @param[in] residuals the residual of this immature point over other frames, 
    so the size should be frames.size()-1.
    @return nullptr if not converged, or a newly created point hessian if converged.
    */
    shared_ptr<PointHessian> 
    optimizeImmaturePoint(shared_ptr<internal::ImmaturePoint> point, int minObs,
                          vector<shared_ptr<ImmaturePointTemporaryResidual>> &residuals);

    /*
    add new immature points and their residuals
    if loop closing is enabled, use corner detection, otherwise use DSO's keypoint selection method
    @param newFrame
    @param gtDepth
    */
    void makeNewTraces(shared_ptr<FrameHessian> newFrame, float *gtDepth);

    /*
    initialize from the coarse initializer
    @param newFrame
    */
    void initializeFromInitializer(shared_ptr<FrameHessian> newFrame);

    /*
    set the marginalization flag for the frames that need to be marginalized
    @param newFH
    */
    void flagFramesForMarginalization(shared_ptr<FrameHessian> &newFH);

    /*
    Set the map point status according to residuals and host frames.
    The map point status will be set to OUT or MARGINALIZED according to some conditions,
    and if so, they will no longer be used in optimization
    */
    void flagPointsForRemoval();

    /*
    remove the outliers
    */
    void removeOutliers();

    /*
    set precalc values
    */
    void setPrecalcValues();

    /*
    solve. eventually migrate to ef.
    @param iteration number of iterations
    @param lambda lambda of Levenberg-Marquardt (LM) iters
    */
    void solveSystem(int iteration, double lambda);

    /*
    linearize all the residuals
    @param fixLinearization if true, fix the jacobians after this linearization
    @return 3-vector of //TODO
    */
    Vec3 linearizeAll(bool fixLinearization);

    // reducer for multi-threading
    void linearizeAll_Reductor(bool fixLinearization,
                               std::vector<shared_ptr<PointFrameResidual>> *toRemove,
                               int min, int max,
                               Vec10 *stats, int tid);
    
    // step from the backup data
    // called in optimization
    bool doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD);

    // set the current state into backup
    void backupState(bool backupLastStep);

    // load from the backup state
    /// TODO: Maybe typo: "loadStateBackup()"?
    void loadSateBackup();

    // energy computing functions, called in optimization
    double calcLEnergy();

    // energy computing functions, called in optimization
    double calcMEnergy();

    void applyRes_Reductor(bool copyJacobians, int min, int max, Vec10 *stats, int tid);

    std::vector<VecX> getNullspaces(
        std::vector<VecX> &nullspaces_pose,
        std::vector<VecX> &nullspaces_scale,
        std::vector<VecX> &nullspaces_affA,
        std::vector<VecX> &nullspaces_affB
    );

    void setNewFrameEnergyTH();

    // make a new keyframe
    void makeKeyFrame(shared_ptr<FrameHessian> fh);

    // make an ordinary frame
    void makeNonKeyFrame(shared_ptr<FrameHessian> &fh);

    // deliver the tracked frame to makeKeyFrame/makeNonKeyFrame
    // if we do linearization, here we will call makeKeyFrame/makeNonKeyFrame,
    // otherwise, they are called in mapping loop
    void deliverTrackedFrame(shared_ptr<FrameHessian> fh, bool needKF);

    /**
     * @brief mapping loop is running in a single thread
     * 
     */
    void mappingLoop();

    // print the residual in optimization
    void printOptRes(const Vec3 &res, double resL, double resM, double resPrior, 
                     double LExact, float a, float b);

    // Fields / attributes / variables:
    public:
    shared_ptr<Camera> Hcalib = nullptr; // calib information

    private:
    // data
    // ====================== changed by tracker-thread. protected by trackMutex ================
    mutex trackMutex;
    shared_ptr<CoarseInitializer> coarseInitializer = nullptr;
    Vec5 lastCoarseRMSE;
    vector<shared_ptr<Frame>> allFrameHistory; // all recorded frames

    // ====================== changed by mapper-thread. protected by mapMutex ================
    mutex mapMutex;
    
    // ====================== General data ================
    shared_ptr<EnergyFunctional> ef = nullptr;  // optimization
    IndexThreadReduce<Vec10> threadReduce;      // multi thread reducing

    shared_ptr<CoarseDistanceMap> coarseDistanceMap = nullptr;   // coarse distance map
    shared_ptr<PixelSelector> pixelSelector = nullptr;           // pixel selector
    float *selectionMap = nullptr;                               // selection map

    // all frames
    /// TODO: Typo in commend? addFrame == addActiveFrame()?
    std::vector<shared_ptr<Frame>> frames; // all active frames, ONLY changed in marginalizeFrame() and addFrame().
    mutex framesMutex; // mutex to lock frame read and write because other places will use this information

    // active residuals
    std::vector<shared_ptr<PointFrameResidual>> activeResiduals;
    float currentMinActDist = 2; /// TODO: What does this "2" mean? What unit?

    std::vector<float> allResVec; // contains all residuals

    // mutex etc. for tracker exchange.
    mutex coarseTrackerSwapMutex;   // if tracker sees that there is a new reference, tracker locks [coarseTrackerSwapMutex] and swaps the two.
    shared_ptr<CoarseTracker> coarseTracker_forNewKF = nullptr; // set as reference. protectedby [coarseTrackerSwapMutex].
    shared_ptr<CoarseTracker> coarseTracker = nullptr; // always used to track new frames. protected by [trackMutex].

    mutex shellPoseMutex;

    // tracking / mapping synchronization. All protected by [trackMapSyncMutex].
    mutex trackMapSyncMutex;
    condition_variable trackedFrameSignal;
    condition_variable mappedFrameSignal;
    deque<shared_ptr<Frame>> unmappedTrackedFrames;
    int needNewKFAfter = -1; // Otherwise, a new KF is *needed that has ID bigger than [needNewKFAfter]*.


    thread mappingThread;
    bool runMapping = true;
    bool needToKetchupMapping = false; /// TODO: Ketchup == Catchup?

    public:
    shared_ptr<Map> globalMap = nullptr; // global map
    FeatureDetector detector; // feature detector
    // ====================== loop closing ================
    public:
    shared_ptr<ORBVocabulary> vocab = nullptr; // vocabulary
    shared_ptr<LoopClosing> loopClosing = nullptr; // loop closing

    // ====================== visualization ================
    public:
    void setViewer(shared_ptr<PangolinDSOViewer> v)
    {
        viewer = v;
        if(viewer)
        {
            viewer->setMap(globalMap);
        }
    }

    private:
    shared_ptr<PangolinDSOViewer> viewer = nullptr;

    // ====================== debug ================
    public:
    /*
    save trajectory in TUM or EUROC format
    @param filename
    @param printOptimized print the trajectory after loop closure?
    */
    void printResult(const string &filename, bool printOptimized = true);
    
    /*
    save trajectory in Kitti format
    NOTE we only save keyframe poses, not all poses, 
    so they cannot be directly evaluated by Kitti's default program (which also ignore the scale issue)
    @param filename
    @param printOptimized
    */
    void printResultKitti(const string &filename, bool printOptimized = true);
   };

}




#endif // LDSO_TWITCH_FULL_SYSTEM_H_