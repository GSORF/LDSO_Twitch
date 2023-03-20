# We have three types of example binaries. 
# All target different types of datasets, namely TUM_MONO, EUROC, KITTI

add_executable( run_ldso_twitch_tum_mono run_ldso_twitch_tum_mono.cc )
target_link_libraries( run_ldso_twitch_tum_mono
    ldso_twitch ${THIRD_PARTY_LIBS}
)

# ... here follows EUROC and KITTI (we will skip that in this stream)