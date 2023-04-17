#include "Feature.h"
#include "Frame.h"
#include "Point.h"

#include "frontend/FullSystem.h"
#include "frontend/CoarseInitializer.h"
#include "frontend/CoarseTracker.h"
#include "frontend/LoopClosing.h"

#include "internal/ImmaturePoint.h"
#include "internal/GlobalCalib.h"
#include "internal/GlobalFuncs.h"
#include "internal/OptimizationBackend/EnergyFunctional.h"

#include <opencv2/features2d/features2d.hpp>
#include <iomanip>
#include <opencv2/highgui/highgui.hpp>

using namespace ldso;
using namespace ldso::internal;

namespace ldso
{
    // Constructor:
    FullSystem::FullSystem(shared_ptr<ORBVocabulary> voc) :
    coarseDistanceMap(new CoarseDistanceMap(wG[0], hG[0])),
    coarseTracker(new CoarseTracker(wG[0], hG[0])),
    coarseTracker_forNewKF(new CoarseTracker(wG[0], hG[0])),
    coarseInitializer(new CoarseInitializer(wG[0], hG[0])),
    ef(new EnergyFunctional()),
    Hcalib(new Camera(fxG[0], fyG[0], cxG[0], cyG[0])),
    globalMap(new Map(this)),
    vocab(voc)
    {
        LOG(INFO) << 
        "This is Direct Sparse Odometry, a fully direct VO proposed by TUM vision group."
        "For more information about dso, see Direct Sparse Odometry, J. Engel, V. Koltun, D. Cremers,"
        "In arXiv:1607.02565, 2016."
        "For loop closing part, see LDSO: Direct Sparse Odometry with Loop Closure, X. Gao, R. Wang, N. Demmel, D. Cremers, "
        "In International Conference on Intelligent Robots and Systems (IROS), 2018" << endl;

        Hcalib->CreateCH(Hcalib); // Create CalibHessian (CH)
        lastCoarseRMSE.setConstant(100); // Initialize Vec5(100,100,100,100,100)
        ef->red = &this->threadReduce // red = Reduce (used in multithreading)
        mappingThread = thread(&FullSystem::mappingLoop, this);

        pixelSelector = shared_ptr<PixelSelector>(new PixelSelector(wG[0], hG[0]));
        selectionMap = new float[wG[0] * hG[0]];

        if(setting_enableLoopClosing)
        {
            loopClosing = shared_ptr<LoopClosing>(new LoopClosing(this));
            if(setting_fastLoopClosing)
            {
                LOG(INFO) << "Use fast loop closing" << endl;
            }
        }
        else
        {
            LOG(INFO) << "loop closing is disabled" << endl;
        }
        
    }

    // Destructor:
    FullSystem::~FullSystem()
    {
        blockUntilMappingIsFinished();
        // remember to release the inner structure
        this->unmappedTrackedFrames.clear();
        if(setting_enableLoopClosing == false)
        {
            delete[] selectionMap;
        }

    }

    // Very important method: Entry point from main()-loop:
    void FullSystem::addActiveFrame(ImageAndExposure *image, int id)
    {
        if(isLost)
        {
            return;
        }
        unique_lock<mutex> lock(trackMutex);

        LOG(INFO) << "*** taking frame " << id << " ***" << endl;

        // create frame and frame hessian
        shared_ptr<Frame> frame(new Frame(image->timestamp));
        frame->CreateFH(frame); // FH: Frame Hessian
        allFrameHistory.push_back(frame);

        // ====== make images =======
        shared_ptr<FrameHessian> fh = frame->frameHessian;
        fh->ab_exposure = image->exposure_time;
        fh->makeImages(image->image, Hcalib->mpCH);

        if(!initialized)
        {
            LOG(INFO) << "Initializing ..." << endl;
            // use initializer
            if(coarseInitializer->frameID < 0) // first frame not set, set it
            {
                coarseInitializer->setFirst(Hcalib->mpCH, fh);
            }
            else if(coarseInitializer->trackFrame(fh))
            {
                // init succeeded
                initializeFromInitializer(fh);
                lock.unlock();
                deliverTrackedFrame(fh, true); // create a new keyframe
                LOG(INFO) << "init success." << endl;
            }
            else
            {
                // still initializing
                frame->poseValid = false;
                frame->ReleaseAll(); // Don't need this frame, release all the internal
            }
            return;
        }
        else
        {
            // init finished, do tracking
            // ====================================== SWAP coarseTracker based on FrameID ======================
            if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
            {
                unique_lock<mutex> crlock(coarseTrackerSwapMutex);
                LOG(INFO) << "swap coarse tracker to " << coarseTracker_forNewKF->refFrameID << endl;
                auto tmp = coarseTracker;
                coarseTracker = coarseTracker_forNewKF;
                coarseTracker_forNewKF = tmp;
            }

            // track the new frame and get the state
            LOG(INFO) << "tracking new frame" << endl;
            Vec4 tres = trackNewCoarse(fh);

            if(!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1]) ||
               !std::isfinite((double) tres[2]) || !std::isfinite((double) tres[3]))
            {
                // invalid result
                LOG(WARNING) << "Initial Tracking failed: LOST!" << endl;
                isLost = true;
                return;
            }

            bool needToMakeKF = false;
            if(setting_keyframesPerSecond > 0)
            {
                // make key frame by time
                needToMakeKF = allFrameHistory.size() == 1 ||
                (frame->timeStamp - frames.back()->timeStamp) > 
                0.95f / setting_keyframesPerSecond;
            }
            else
            {
                Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                            coarseTracker->lastRef_aff_g2l, fh->aff_g2l());
                float b = setting_kfGlobalWeight * setting_maxShiftWeightT * sqrtf((double) tres[1]) / (wG[0] + hG[0]) +
                          setting_kfGlobalWeight * setting_maxShiftWeightR * sqrtf((double) tres[2]) / (wG[0] + hG[0]) +
                          setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double) tres[3]) / (wG[0] + hG[0]) +
                          setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float) refToFh[0]));

                bool b1 = b > 1;
                bool b2 = 2 * coarseTracker->firstCoarseRMSE < tres[0];

                needToMakeKF = allFrameHistory.size() == 1 || b1 || b2;
                
            }

            if(viewer)
            {
                viewer->publishCamPose(fh->frame, Hcalib->mpCH);
            }

            lock.unlock();
            LOG(INFO) << "deliver frame " << fh->frame->id << endl;
            deliverTrackedFrame(fh, needToMakeKF);
            LOG(INFO) << "add active frame returned" << endl << endl;
            return;
        }
    }

    void FullSystem::deliverTrackedFrame(shared_ptr<FrameHessian> fh, bool needKF)
    {
        if(linearizeOperation)
        {
            if(needKF)
            {
                makeKeyFrame(fh);
            }
            else
            {
                makeNonKeyFrame(fh);
            }
            
        }
        else
        {
            unique_lock<mutex> lock(trackMapSyncMutex);
            unmappedTrackedFrames.push_back(fh->frame);
            trackedFrameSignal.notify_all();
            while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1)
            {
                LOG(INFO) << "wait for mapped frame signal" << endl;
                mappedFrameSignal.wait(lock);
            }
            lock.unlock();
        }
    }
    Vec4 FullSystem::trackNewCoarse(shared_ptr<FrameHessian> fh)
    {
        assert(allFrameHistory.size() > 0);

        shared_ptr<FrameHessian> lastF = coarseTracker->lastRef;
        CHECK(coarseTracker->lastRef->frame != nullptr);

        AffLight aff_last_2_l = AffLight(0, 0); // a=0, b=0 parameters?

        // try a lot of pose values and see which is best
        std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;
        if(allFrameHistory.size() == 2)
        {
            for(unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
            {
                // TODO: maybe wrong, size is obviously zero
                lastF_2_fh_tries.push_back(SE3()); // use identity
            }
        }
        else
        {
            // fill the pose tries ...
            // use the last before last and the last before before last (well my English is really poor...)
            shared_ptr<Frame> slast = allFrameHistory[allFrameHistory.size() - 2];
            shared_ptr<Frame> sprelast = allFrameHistory[allFrameHistory.size() - 3];

            SE3 slast_2_sprelast; // identity
            SE3 lastF_2_slast; // identity

            { // lock on global pose consistency!
                unique_lock<mutex> crlock(shellPoseMutex);
                slast_2_sprelast = sprelast->getPose() * slast->getPose().inverse();
                lastF_2_slast = slast->getPose() * lastF->frame->getPose().inverse();
                aff_last_2_l = slast->aff_g2l;
            }
            SE3 fh_2_slast = slast_2_sprelast; // assumed to be the same as fh_2_slast (Constant Velocity Motion Model).

            // get last delta-movement.
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast); // assume constant motion
            lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast); // assume double motion (frame skipped)
            lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion
            lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
            lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

            // just try a TON of different initializations (all rotations).
            // In the end, if they don't work they will only be tried on the coarsest level, which is super fast anyway
            // also, if tracking fails here, we loose. So we really really want to avoid that.
            for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta += 0.01) // TODO changed this into +=0.01, where DSO writes ++
            {
                // List rotation hypothesis:
                // Positive rotation
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0,0,0))); // assume constant motion.
                // Negative rotation
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0,0,0))); // assume constant motion.

                // Combinations (positive)
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0,0,0))); // assume constant motion.
                // Combinations (positive and negative)
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0,0,0))); // assume constant motion.
                // Combinations (positive and negative, swapped signs)
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0,0,0))); // assume constant motion.
                // Combinations (negative)
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0,0,0))); // assume constant motion.
                // Rotation about all axes (positive and negative)
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0,0,0))); // assume constant motion.

                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0,0,0))); // assume constant motion.
                lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * 
                SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0,0,0))); // assume constant motion.
            }

            if(!slast->poseValid || !sprelast->poseValid || !lastF->frame->poseValid)
            {
                lastF_2_fh_tries.clear(); // delete motion hypotheses from above
                lastF_2_fh_tries.push_back(SE3()); // assuming zero motion
            }
        }

        Vec3 flowVecs = Vec3(100, 100, 100);
        SE3 lastF_2_fh = SE3();
        AffLight aff_g2l = AffLight(0,0); // a=0 and b=0?

        // As long as maxResForImmediateAccept is not reached, I'll continue through the options.
        // I'll keep track of the so-far best achieved residual for each (pyramid) level in achievedRes.
        // If on coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

        Vec5 achievedRes = Vec5::Constant(NAN);
        bool haveOneGood = false;
        int tryIterations = 0;
        for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
        {
            AffLight aff_g2l_this = aff_last_2_l;
            SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

            // use coarse tracker to solve the iteration
            bool trackingIsGood = coarseTracker->trackNewestCoarse(fh, last_F_2_fh_this, aff_g2l_this, pyrLevelsUsed - 1, achievedRes); // Each level has to be at least as good as the last try.
            tryIterations++;

            // do we have a new winner?
            if(trackingIsGood && std::isfinite((float) coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >= achievedRes[0]))
            {
                flowVecs = coarseTracker->lastFlowIndicators;
                aff_g2l = aff_g2l_this;
                lastF_2_fh = lastF_2_fh_this;
                haveOneGood = true;
            }

            // take over achieved res (always).
            if(haveOneGood)
            {
                /// TOOD: Replace "5" with achievedRes.size()
                for (int i = 0; i < 5; i++)
                {
                    if(!std::isfinite((float) achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])
                    {
                        // take over if achievedRes is either bigger or NAN.
                        achievedRes[i] = coarseTracker->lastResiduals[i];
                    }
                }
            }

            if(haveOneGood && achievedRes[0] < lastCoarseRMSE[0] * setting_reTrackThreshold)
            {
                break;
            }

        }

        if(!haveOneGood)
        {
            LOG(WARNING) << "BIG ERROR! Tracking failed entirely. Take predicted pose and hope we may somehow recover." << endl;

            flowVecs = Vec3(0,0,0);
            aff_g2l = aff_last_2_l;
            lastF_2_fh = lastF_2_fh_tries[0];
        }
        
        lastCoarseRMSE = achievedRes;

        // set the pose of new frame
        CHECK(coarseTracker->lastRef->frame != nullptr);
        SE3 camToWorld = lastF->frame->getPose().inverse() * lastF_2_fh.inverse();
        fh->frame->setPose(camToWorld.inverse());
        fh->frame->aff_g2l = aff_g2l;

        if(coarseTracker->firstCoarseRMSE < 0)
        {
            coarseTracker->firstCoarseRMSE = achievedRes[0];
        }

        LOG(INFO) << "Coarse Tracker tracked ab = " << aff_g2l.a << " " << aff_g2l.b <<
        " (exposure " << fh->ab_exposure << " ). Res " << achievedRes[0] << endl;

        return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
    }

    void FullSystem::blockUntilMappingIsFinished()
    {
        {
            unique_lock<mutex> lock(trackMapSyncMutex);
            if(!runMapping)
            {
                // mapping is already finished, no need to finish again
                return;
            }
            runMapping = false;
            trackedFrameSignal.notify_all();
        }

        mappingThread.join();

        if(setting_enableLoopClosing)
        {
            loopClosing->SetFinish(true);
            if(globalMap->NumFrames() > 4)
            {
                globalMap->lastOptimizeAllKFs();
            }
        }

        // Update world points in case optimization hasn't run (with all keyframes)
        // It would maybe be better if the 3d points would always be updated as soon
        // as the poses or depths are updated (no matter if in PoseGraphOptimization(PGO) or in sliding window BundleAdjustment (BA))
        globalMap->UpdateAllWorldPoints();
    }

    void FullSystem::makeKeyFrame(shared_ptr<FrameHessian> fh)
    {
        shared_ptr<Frame> frame = fh->frame;
        auto refFrame = frames.back();

        {
            unique_lock<mutex> crlock(shellPoseMutex);
            fh->setEvalPT_scaled(fh->getPose(), fh->frame->aff_g2l);
            fh->frame->setPoseOpti(Sim3(fh->frame->getPose().matrix));
        }

        LOG(INFO) << "frame " << fh->frame->id << " is marked as key frame, active keyframes: " << frames.size() << endl;

        // trace new keyframe
        traceNewCoarse(fh);

        unique_lock<mutex> lock(mapMutex);

        // == flag frames to be marginalized == //
        flagFramesForMarginalization(fh);

        // add new frame to hessian struct
        {
            unique_lock<mutex> lck(framesMutex);
            fh->idx = frames.size();
            frames.push_back(fh->frame);
            fh->frame->kfId = fh->frameID = globalMap->NumFrames();
        }

        ef->insertFrame(fh, Hcalib->mpCH);
        setPrecalcValues();

        // ======================= add new residuals for old points ==========================
        LOG(INFO) << "adding new residuals" << endl;
        int numFwdResAdde = 0;
        for (auto fht : frames)
        {
            // go through all active frames
            shared_ptr<FrameHessian> &fh1 = fht->frameHessian;
            if(fh1 == fh)
            {
                continue;
            }
            for (auto feat: fht->features)
            {
                if(feat->status == Feature::FeatureStatus::VALID && feat->point->status == Point::PointStatus::ACTIVE)
                {
                    // go through all active feature points
                    shared_ptr<PointHessian> ph = feat->point->mpPH;

                    // add new residuals into this point hessian
                    shared_ptr<PointFrameResidual> r(new PointFrameResidual(ph,fh1, fh)); // residual from fh1 to fh

                    r->setState(ResState::IN);
                    ph->residuals.push_back(r);
                    ef->insertResidual(r);

                    ph->lastResiduals[1] = ph->lastResiduals[0];
                    ph->lastResiduals[0] = pair<shared_ptr<PointFrameResidual>, ResState>(r, ResState::IN);
                    numFwdResAdde++;
                }
            }
        }
        
        // =========== Activate Points (& flag for marginalization). ==============
        activatePointsMT();
        ef->makeIDX();

        // =========== OPTIMIZE ALL ==========================
        fh->frameEnergyTH = frames.back()->frameHessian->frameEnergyTH;
        LOG(INFO) << "call optimize on kf " << frame->kfId << endl;
        float rmse = optimize(setting_maxOptIterations);
        LOG(INFO) << "optimize is done!" << endl;

        // ============ Figure out if INITIALIZATION FAILED =====================
        int numKFs = globalMap->NumFrames();
        if(numKFs <= 4)
        {
            if(numKFs == 2 && rmse > 20 * benchmark_initializerSlackFactor)
            {
                LOG(WARNING) << "I THINK INITIALIZATION FAILED! Resetting." << endl;
                LOG(INFO) << "rmse = " << rmse << endl;
                initFailed = true;
            }
            if(numKFs == 3 && rmse > 13 * benchmark_initializerSlackFactor)
            {
                LOG(WARNING) << "I THINK INITIALIZATION FAILED! Resetting." << endl;
                LOG(INFO) << "rmse = " << rmse << endl;
                initFailed = true;
            }
            if(numKFs == 4 && rmse > 9 * benchmark_initializerSlackFactor)
            {
                LOG(WARNING) << "I THINK INITIALIZATION FAILED! Resetting." << endl;
                LOG(INFO) << "rmse = " << rmse << endl;
                initFailed = true;
            }
        }

        if(isLost)
        {
            return;
        }

        // ============== REMOVE OUTLIER =======================
        removeOutliers();

        // swap the coarse Tracker for new kf
        {
            unique_lock<mutex> crlock(coarseTrackerSwapMutex);
            coarseTracker_forNewKF->makeK(Hcalib->mpCH);
            vector<shared_ptr<FrameHessian>> fhs;
            for (auto &f : frames)
            {
                fhs.push_back(f->frameHessian);
            }
            coarseTracker_forNewKF->setCoarseTrackingRef(fhs);
        }

        // ======================= (Activate-)Marginalize Points ===================
        // traditional bundle adjustment when marginalizing all points
        flagPointsForRemoval();
        ef->dropPointsF();

        getNullspaces(
            ef->lastNullspaces_pose,
            ef->lastNullspaces_scale,
            ef->lastNullspaces_affA,
            ef->lastNullspaces_affB,
        );

        ef->marginalizePointsF();

        // ====================== add new Immature points & new residuals ====================
        makeNewTraces(fh, 0);

        // record the relative poses, note we are building a covisibility graph in fact
        auto minandmax = std::minmax_element(frames.begin(), frames.end(), CmpFrameKFID());
        unsigned long minKFId = (*minandmax.first)->kfId;
        unsigned long maxKFId = (*minandmax.second)->kfId;

        if(setting_fastLoopClosing == false)
        {
            // record all active keyframes
            for (auto &fr : frames)
            {
                auto allKFs = globalMap->GetAllKFs();
                for (auto &f2 : allKFs)
                {
                    if(f2->kfId > minKFId && f2->kfId < maxKFId && f2 != fr)
                    {
                        unique_lock<mutex> lock(fr->mutexPoseRel);
                        unique_lock<mutex> lock2(f2->mutexPoseRel);
                        fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
                        f2->poseRel[fr] = Sim3((f2->getPose() * fr->getPose().inverse()).matrix());
                    }
                }
            }
        }
        else
        {
            // Only record the reference and first frame and also update the keyframe poses in sliding window
            {
                unique_lock<mutex> lock(frame->mutexPoseRel);
                frame->poseRel[refFrame] = Sim3((frame->getPose() * refFrame->getPose().inverse()).matrix());
                auto firstFrame = frames.front();
                frame->poseRel[firstFrame] = Sim3((frame->getPose() * firstFrame->getPose().inverse()).matrix());
            }

            // update the poses in window
            for (auto &fr: frames)
            {
                if(fr == frame)
                {
                    continue;
                }
                for (auto rel: fr->poseRel)
                {
                    auto f2 = rel.first;
                    fr->poseRel[f2] = Sim3((fr->getPose() * f2->getPose().inverse()).matrix());
                }
            }
        }

        // visualization
        if(viewer)
        {
            viewer->publishKeyframes(frames, false, Hcalib->mpCH);
        }

        // ========================= Marginalize Frames ========================
        {
            unique_lock<mutex> lck(framesMutex);
            for (unsigned int i = 0; i < frames.size(); i++)
            {
                if(frames[i]->frameHessian->flaggedForMarginalization)
                {
                    LOG(INFO) << "marg frame " << frames[i]->id << endl;
                    CHECK(frames[i] != coarseTracker->lastRef->frame);
                    marginalizeFrame(frames[i]);
                    i = 0;
                }
            }
        }

        // Add current kf into map and detect loops
        globalMap->AddKeyFrame(fh->frame);
        if(setting_enableLoopClosing)
        {
            loopClosing->InsertKeyFrame(frame);
        }
        LOG(INFO) << "make keyframe done" << endl;
    }

    void FullSystem::makeNonKeyFrame(shared_ptr<FrameHessian> &fh)
    {
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            fh->setEvalPT_scaled(fh->frame->getPose(), fh->frame->aff_g2l);
        }
        traceNewCoarse(fh);
        fh->frame->ReleaseAll(); // no longer needs it
    }

    void FullSystem::marginalizeFrame(shared_ptr<Frame> &frame)
    {
        // marginalize or remove all this frames points
        ef->marginalizeFrame(frame->frameHessian);

        // drop all observations of existing points in that frame
        for (shared_ptr<Frame> &fr: frames)
        {
            if(fr == frame)
            {
                continue;
            }

            for (auto feat: fr->features)
            {
                if(feat->status == Feature::FeatureStatus::VALID && feat->point->status == Point::PointStatus::ACTIVE)
                {
                    // If feature is VALID and Point is ACTIVE:
                    shared_ptr<PointHessian> ph = feat->point->mpPH;

                    // remove the residuals projected into this frame
                    size_t n = ph->residuals.size();
                    for (size_t i = 0; i < n; i++)
                    {
                        auto r = ph->residuals[i]; // residual
                        if(r->target.lock() == frame->frameHessian)
                        {
                            if(ph->lastResiduals[0].first == r)
                            {
                                ph->lastResiduals[0].first = nullptr;
                            }
                            else if(ph->lastResiduals[1].first == r)
                            {
                                ph->lastResiduals[1].first = nullptr;
                            }
                            ef->dropResidual(r);
                            i--; // Current residual id
                            n--; // Total number of residuals
                        }
                    }
                }
            }
        }

        // remove this frame from recorded frames
        frame->ReleaseAll(); // release all things in this frame
        deleteOutOrder<shared_ptr<Frame>>(frames, frame);

        // reset the optimization idx
        for (unsigned int i = 0; i < frames.size(); i++)
        {
            frames[i]->frameHessian->idx = i;
        }

        setPrecalcValues();
        ef->setAdjointsF(Hcalib->mpCH);
    }

    void FullSystem::flagFramesForMarginalization(shared_ptr<FrameHessian> &newFH)
    {
        // Loop through all frames outside of sliding window and mark them as ready for marginalization
        if(setting_minFrameAge > setting_maxFrames)
        {
            for (size_t i = setting_maxFrames; i < frames.size(); i++)
            {
                shared_ptr<FrameHessian> &fh = frames[i - setting_maxFrames]->frameHessian;
                LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << std::endl;
                fh->flaggedForMarginalization = true;
            }
            return;            
        }

        int flagged = 0;

        // marginalize all frames that have not enough points.
        for (int i = 0; i < (int) frames.size(); i++)
        {
            shared_ptr<FrameHessian> &fh = frames[i]->frameHessian;
            int in = 0, out = 0;
            for (auto &feat: frames[i]->features)
            {
                if(feat->status == Feature::FeatureStatus::IMMATURE)
                {
                    in++;
                    continue;
                }

                shared_ptr<Point> p = feat->point;
                if(p && p->status == Point::PointStatus::ACTIVE)
                {
                    in++;
                }
                else
                {
                    out++;
                }
                
            }

            Vec2 refToFh = AffLight::fromToVecExposure(frames.back()->frameHessian->ab_exposure, fh->ab_exposure, frames.back()->frameHessian->aff_g2l(), fh->aff_g2l());
            // Result:
            // refToFh[0] = ab_exposure
            // refToFh[1] = aff_g2l

            // some kind of marginalization conditions
            if( (in < setting_minPointsRemaining * (in + out) || 
                 fabs(logf((float) refToFh[0])) > setting_maxLogAffFacInWindow) &&
                 ((int) frames.size()) - flagged > setting_minFrames)
            {
                LOG(INFO) << "frame " << fh->frame->kfId << " is set as marged" << std::endl;
                fh->flaggedForMarginalization = true;
                flagged++;
            }
        }

        // still too much (frames?), marginalize one
        if((int) frames.size() - flagged >= setting_maxFrames)
        {
            double smallestScore = 1;
            shared_ptr<Frame> toMarginalize = nullptr;
            shared_ptr<Frame> latest = frames.back();

            for (auto &fr: frames)
            {
                // Check for a minimum age of the current frame OR first frame
                if(fr->frameHessian->frameID > latest->frameHessian->frameID - setting_minFrameAge
                || fr->frameHessian->frameID == 0)
                {
                    continue;
                }
                double distScore = 0;
                for (FrameFramePrecalc &ffh : fr->frameHessian->targetPrecalc)
                {
                    // Check for minimum age of current frame OR if target==host
                    if(ffh.target.lock()->frameID > latest->frameHessian->frameID - setting_minFrameAge + 1
                    || ffh.target.lock() == ffh.host.lock())
                    {
                        continue
                    }
                    distScore += 1 / (1e-5 + ffh.distanceLL);
                }
                distScore *= -sqrtf(fr->frameHessian->targetPrecalc.back().distanceLL);

                if(distScore < smallestScore)
                {
                    smallestScore = distScore; // Update the smallest score
                    toMarginalize = fr; // Update frame that needs to be marginalized
                }
            }

            if(toMarginalize)
            {
                toMarginalize->frameHessian->flaggedForMaginalization = true;
                LOG(INFO) << "frame " << toMarginalize->kfId << " is set as marged" << std::endl;
                flagged++;
            }
        }
    }

    float FullSystem::optimize(int mnumOptIts)
    {
        if(frames.size() < 2)
        {
            return 0;
        }
        if(frames.size() < 3)
        {
            mnumOptIts = 20;
        }
        if(frames.size() < 4)
        {
            mnumOptIts = 15;
        }

        // get statistics and active residuals.
        activeResiduals.clear();
        int numPoints = 0;
        int numLRes = 0;

        for (shared_ptr<Frame> &fr : frames)
        {
            for(auto &feat: fr->features)
            {
                shared_ptr<Point> p = feat->point;
                // Check for VALID feature and SET + ACTIVE point:
                if(feat->status == Feature::FeatureStatus::VALID && p && p->status == Point::PointStatus::ACTIVE)
                {
                    auto ph = p->mpPH;
                    for (auto &r : ph->residuals)
                    {
                        if(!r->isLinearized)
                        {
                            activeResiduals.push_back(r);
                            r->resetOOB(); // OOB: Out Of Bounds
                        }
                        else
                        {
                            numLRes++;
                        }
                    }
                }
                numPoints++
            }
        }

        LOG(INFO) << "active residuals: " << activeResiduals.size() << std::endl;

        Vec3 lastEnergy = linearizeAll(false);
        double lastEnergyL = calcLEnergy();
        double lastEnergyM = calcMEnergy();

        // apply res(idual?)
        if(multiThreading)
        {
            // ldso::internal::IndexThreadReduce<Vec10>::reduce(std::function<void (int, int, Vec10 *, int)> callPerIndex, int first, int end, int stepSize)
            // -> first=0, end=activeResiduals.size(), stepSize=50
            threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4),0, activeResiduals.size(), 50);

        }
        else
        {
            applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
        }

        printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frames.back()->frameHessian->aff_g2l().a,frames.back()->frameHessian->aff_g2l().b);

        double lambda = 1e-1;
        float stepsize = 1;
        VecX previousX = VecX::Constant(CPARS + 8 * frames.size(), NAN);

        for (int iteration = 0; iteration < mnumOptIts; iteration++)
        {
            // Solve!
            backupState(iteration != 0);

            solveSystem(iteration, lambda);
            double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
            previousX = ef->lastX;

            // Check for finite direction magnitude and set solver to "Step Momentum":
            if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
            {
                float newStepsize = exp(incDirChange * 1.4);
                if(incDirChange < 0 && stepsize > 1)
                {
                    stepsize = 1;
                }
                stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize ));

                // Check the bounds of stepsize to be in [0.25, ... , 2.0]:
                if(stepsize > 2)
                {
                    stepsize = 2;
                }
                if(stepsize < 0.25)
                {
                    stepsize = 0.25;
                }
            }
            bool canbreak = doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);

            // eval new energy!
            Vec3 newEnergy = linearizeAll(false);
            double newEnergyL = calcLEnergy();
            double newEnergyM = calcMEnergy();

            printOptRes(newEnergy, newEnergyL, newEnergyM, 0, 0, frames.back()->frameHessian->aff_g2l().a,frames.back()->frameHessian->aff_g2l().b);

            // control the lambda in LM (Levenberg-Marquardt)
            if(setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM < lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
            {
                // energy is decreasing
                if(multiThreading)
                {
                    threadReduce.reduce(bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4),0, activeResiduals.size(), 50);
                }
                else
                {
                    applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);
                }

                lastEnergy = newEnergy;
                lastEnergyL = newEnergyL;
                lastEnergyM = newEnergyM;

                lambda *= 0.25;
            }
            else
            {
                // energy increases, reload the backup state and increase lambda
                loadSateBackup();
                lastEnergy = linearizeAll(false);
                lastEnergyL = calcLEnergy();
                lastEnergyM = calcMEnergy();
                lambda *= 1e2; // lambda = lambda * 10*10 = lambda * 100
            }

            if(canbreak && iteration >= setting_minOptIterations)
            {
                break;
            }
        }

        Vec10 newStateZero = Vec10::Zero();
        newStateZero.segment<2>(6) = frames.back()->frameHessian->get_state().segment<2>(6);

        frames.back()->frameHessian->setEvalPT(frames.back()->frameHessian->PRE_worldToCam, newStateZero);

        EFDeltaValid = false;
        EFAdjointsValid = false;
        ef->setAdjointsF(Hcalib->mpCH);
        setPrecalcValues();

        lastEnergy = linearizeAll(true); // fix all the linearizations;

        if(!std::isfinite((double) lastEnergy[0]) 
        || !std::isfinite((double) lastEnergy[1]) 
        || !std::isfinite((double) lastEnergy[2]) )
        {
            LOG(WARNING) << "KF Tracking failed: LOST!";
            isLost = true;
        }

        // set the estimated pose into frame
        {
            unique_lock<mutex> crlock(shellPoseMutex);
            for (auto fr: frames)
            {
                fr->setPose(fr->frameHessian->PRE_camToWorld.inverse());
                if(fr->kfId >= globalMap->getLatestOptimizedKfId())
                {
                    fr->setPoseOpti(Sim3(fr->getPose().matrix()));
                }
                fr->aff_g2l = fr->frameHessian->aff_g2l();
            }
        }

        return sqrtf((float) (lastEnergy[0] / (patternNum * ef->resInA)));
    }

    void FullSystem::setGammaFunction(float *BInv)
    {
        if(BInv == nullptr)
        {
            return;
        }

        // copy BInv
        memcpy(Hcalib->mpCH->Binv, BInv, sizeof(float) * 256);

        // invert
        for (int i = 0; i < 255; i++)
        {
            // find val, such that Binv[val] = i.
            // I dont care about speed for this, so do it the stupid way.

            for (int s = 1; s < 255; s++)
            {
                if(BInv[s] <= i && BInv[s+1] >= i)
                {
                    Hcalib->mpCH->B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
                    break;
                }
            }
        }

        // Update the limits of Hcalib - avoid imprecise values:
        Hcalib->mpCH->B[0] = 0;
        Hcalib->mpCH->B[255] = 255;
    }

    shared_ptr<PointHessian> FullSystem::optimizeImmaturePoint(shared_ptr<internal::ImmaturePoint> point, int minObs, vector<shared_ptr<ImmaturePointTemporaryResidual>> &residuals)
    {
        int nres = 0;
        shared_ptr<Frame> hostFrame = point->feature->host.lock();
        assert(hostFrame); // the feature should have a host frame

        for (auto fr: frames)
        {
            if(fr != hostFrame)
            {
                residuals[nres]->state_NewEnergy = residuals[nres]->state_energy = 0;
                residuals[nres]->state_NewState = ResState::OUTLIER;
                residuals[nres]->state_state = ResState::IN;
                residuals[nres]->target = fr->frameHessian;
                nres++
            }
        }
        assert(nres == frames.size() - 1);

        float lastEnergy = 0;
        float lastHdd = 0;
        float lastbd = 0;
        float currentIdepth = (point->idepth_max + point->idepth_min) * 0.5f;

        for(int i = 0; i < nres; i++)
        {
            lastEnergy += point->linearizeResidual(Hcalib->mpCH, 1000, residuals[i], lastHdd, lastbd, currentIdepth);
            residuals[i]->state_state = residuals[i]->state_NewState;
            residuals[i]->state_energy = residuals[i]->state_NewEnergy;
        }

        if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
        {
            return 0; /// TODO: maybe better return nullptr?
        }

        // do LM iteration for this immature point
        float lambda = 0.1;
        for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++)
        {
            float H = lastHdd;
            H *= 1 + lambda;

            float step = (1.0 / H) * lastbd;
            float newIdepth = currentIdepth - step;

            float newHdd = 0;
            float newbd = 0;
            float newEnergy = 0;

            for (int i = 0; i < nres; i++)
            {
                // compute the energy in other frames
                newEnergy += point->linearizeResidual(Hcalib->mpCH, 1, residuals[i], newHdd, newbd, newIdepth);

            }
            
            /// STD IS FINITE
            if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
            {
                return 0; /// TODO: maybe better return nullptr?
            }


            // Check if the energy is decreasing
            if(newEnergy < lastEnergy)
            {
                // we are improving during our optimization
                currentIdepth = newIdepth;
                lastHdd = newHdd;
                lastbd = newbd;
                lastEnergy = newEnergy;
                for (int i = 0; i < nres; i++)
                {
                    residuals[i]->state_state = residuals[i]->state_NewState;
                    residuals[i]->state_energy = residuals[i]->state_NewEnergy;
                }
                lambda *= 0.5;
            }
            else
            {
                // we are not improving during the optimization
                lambda *= 5;
            }

            if(fabsf(step) < 0.0001 * currentIdepth)
            {
                break;
            }
        }

        if(!std::isfinite(currentIdepth))
        {
            return nullptr;
        }

        int numGoodRes = 0;
        for(int i = 0; i < nres; i++)
        {
            if(residuals[i]->state_state == ResState::IN)
            {
                numGoodRes++;
            }
        }

        if(numGoodRes < minObs)
        {
            // an outlier
            return nullptr;
        }

        point->feature->CreateFromImmature(); // create a point from immature feature
        shared_ptr<PointHessian> p = point->feature->point->mpPH;

        // set residual status in new map point
        p->lastResiduals[0].first = nullptr;
        p->lastResiduals[0].second = ResState::OOB;
        p->lastResiduals[1].first = nullptr;
        p->lastResiduals[1].second = ResState::OOB;
        
        p->setIdepthZero(currentIdepth);
        p->setIdepth(currentIdepth);

        // move the immature point residuals into the new map point
        for (int i = 0; i < nres; i++)
        {
            if(residuals[i]->state_state == ResState::IN)
            {
                shared_ptr<FrameHessian> host = point->feature->host.lock()->frameHessian;
                shared_ptr<FrameHessian> target = residuals[i]->target.lock();
                shared_ptr<PointFrameResidual> r(new PointFrameResidual(p, host, target));
            }
        }
        



        


        
    }

}
