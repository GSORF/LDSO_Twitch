#pragma once
#ifndef LDSDO_TWITCH_IMAGE_RW_H_
#define LDSDO_TWITCH_IMAGE_RW_H_

#include "NumTypes.h"
#include "MinimalImage.h"

namespace ldso
{
    namespace IOWrap
    {
        MinimalImageB *readImageBW_8U(std::string filename);

        MinimalImageB3 *readImageRGB_8U(std::string filename);

        MinimalImage<unsigned short> *readImageBW_16U(std::string filename);

        MinimalImageB *readStreamBW_8U(char *data, int numBytes);

        void writeImage(std::string filename, MinimalImageB *img);

        void writeImage(std::string filename, MinimalImageB3 *img);

        void writeImage(std::string filename, MinimalImageF *img);

        void writeImage(std::string filename, MinimalImageF3 *img);


    } // namespace IOWrap

} // namespace ldso


#endif // LDSDO_TWITCH_IMAGE_RW_H_

