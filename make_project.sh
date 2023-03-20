BUILD_TYPE=Release
NUM_PROC=4

BASEDIR="$PWD"

# Add library (DBoW3)
cd "$BASEDIR/thirdparty/DBoW3"
mkdir build 
cd build 
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

# Add library (g2o)
cd "$BASEDIR/thirdparty/g2o"
mkdir build 
cd build 
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC

# Compile our project
cd "$BASEDIR"
mkdir build 
cd build 
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
make -j$NUM_PROC
