add_library(ldso_twitch
    # all of the source files
    # ...
    Feature.cc
    Settings.cc

    internal/GlobalCalib.cc

    frontend/ImageRW_OpenCV.cc
)

target_link_libraries(
    ldso_twitch 
    ${THIRD_PARTY_LIBS}
)


