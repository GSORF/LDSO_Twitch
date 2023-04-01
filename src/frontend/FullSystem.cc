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
}
