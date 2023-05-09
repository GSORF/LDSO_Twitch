add_library(ldso_twitch
    # all of the source files
    # ...
    Camera.cc
    Frame.cc
    Feature.cc
    Settings.cc
    Map.cc
    Point.cc
    
    internal/GlobalCalib.cc
    internal/FrameFramePrecalc.cc
    internal/FrameHessian.cc
    internal/ImmaturePoint.cc
    internal/PointHessian.cc
    internal/PR.cc
    internal/Residuals.cc

    internal/OptimizationBackend/AccumulatedSCHessian.cc
    internal/OptimizationBackend/AccumulatedTopHessian.cc
    internal/OptimizationBackend/EnergyFunctional.cc

    frontend/CoarseInitializer.cc
    frontend/CoarseTracker.cc
    frontend/DSOViewer.cc
    frontend/FeatureDetector.cc
    frontend/FeatureMatcher.cc
    frontend/FullSystem.cc
    frontend/LoopClosing.cc
    frontend/PixelSelector2.cc
    frontend/Undistort.cc
    frontend/ImageRW_OpenCV.cc

)

target_link_libraries(
    ldso_twitch 
    ${THIRD_PARTY_LIBS}
)


