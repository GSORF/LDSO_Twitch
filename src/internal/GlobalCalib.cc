#include "internal/GlobalCalib.h"

namespace ldso
{
    namespace internal
    {
        int wG[PYR_LEVELS], hG[PYR_LEVELS]; // Image dimensions
        float fxG[PYR_LEVELS], fyG[PYR_LEVELS], // Pinhole camera model
                     cxG[PYR_LEVELS], cyG[PYR_LEVELS];

        float fxiG[PYR_LEVELS], fyiG[PYR_LEVELS], // Inverse pinhole camera model
                     cxiG[PYR_LEVELS], cyiG[PYR_LEVELS];

        Eigen::Matrix3f KG[PYR_LEVELS], KiG[PYR_LEVELS];

        float wM3G; // w-3
        float hM3G; // h-3

        void setGlobalCalib(int w, int h, const Eigen::Matrix3f &K)
        {
            int wlvl = w; // 800 -> 400 -> 200 -> 100 -> 50 -> 25 -> !!!12.5
            int hlvl = h; // 600 -> 300 -> 150 -> 75 -> !!! %2 will not be even
            pyrLevelsUsed = 1;

            while (wlvl % 2 == 0 && hlvl % 2 == 0 && wlvl * hlvl > 5000 && pyrLevelsUsed < PYR_LEVELS)
            {
                wlvl /= 2;
                hlvl /= 2;
                pyrLevelsUsed++;
            }
            
            LOG(INFO) << "Using pyramid levels 0 to " << pyrLevelsUsed - 1 
                      << ". Coarsest resolution: " << wlvl << " x " << hlvl << "!" << endl;

            if(wlvl > 100 && hlvl > 100)
            {
                LOG(WARNING) << "================WARNING!==============" << endl <<
                "using not enough pyramid levels." << endl <<
                "Consider scaling to a resolution that is a multiple of a power of 2." << endl;
            }
            if(pyrLevelsUsed < 3)
            {
                LOG(WARNING) << "================WARNING!==============" << endl <<
                "I need higher resolution." << endl <<
                "I will probably segfault." << endl;
            }

            wM3G = w - 3;
            hM3G = h - 3;

            wG[0] = w;
            hG[0] = h;
            KG[0] = K;

            /*
                fx | 0  | cx
            K = 0  | fy | cy
                0  | 0  | 1
            
            */

            // Fill up the non-inverted and inverted camera intrinsics FOR THE FINEST level:
            fxG[0] = K(0, 0);
            fyG[0] = K(1, 1);
            cxG[0] = K(0, 2);
            cyG[0] = K(1, 2);
            KiG[0] = KG[0].inverse();
            fxiG[0] = KiG[0](0, 0);
            fyiG[0] = KiG[0](1, 1);
            cxiG[0] = KiG[0](0, 2);
            cyiG[0] = KiG[0](1, 2);

            // Fill up the non-inverted and inverted camera intrinsics FOR THE REMAINING levels:
            
            for (int level = 1; level < pyrLevelsUsed; ++level)
            {
                // Image dimensions:
                wG[level] = w >> level; // Division of width by 2
                hG[level] = h >> level; // Division of height by 2

                // Camera intrinsics:
                fxG[level] = fxG[level - 1] * 0.5; // Division of fx by 2
                fyG[level] = fyG[level - 1] * 0.5; // Division of fy by 2
                cxG[level] = (cxG[0] + 0.5) / ((int) 1 << level) - 0.5;
                cyG[level] = (cyG[0] + 0.5) / ((int) 1 << level) - 0.5;

                KG[level] << fxG[level], 0.0, cxG[level], 0.0, fyG[level], cyG[level], 0.0, 0.0, 1.0; // synthetic (?)
                KiG[level] = KG[level].inverse();

                fxiG[level] = KiG[level](0, 0);
                fyiG[level] = KiG[level](1, 1);
                cxiG[level] = KiG[level](0, 2);
                cyiG[level] = KiG[level](1, 2);

            }
        }

    }
    
} // namespace internal
