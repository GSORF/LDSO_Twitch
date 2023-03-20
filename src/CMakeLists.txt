add_library(ldso_twitch
    # all of the source files
    # ...
    Feature.cc
    Settings.cc
)

target_link_libraries(
    ldso_twitch 
    ${THIRD_PARTY_LIBS}
)


