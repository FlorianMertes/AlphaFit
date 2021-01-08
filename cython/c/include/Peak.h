#ifndef PEAK_H
#define PEAK_H

#include "Tail.h"

typedef struct PeakStruct {
    c_Tail** tails;
    int num_tails;
    int* weight_index;
    int area_index;
} c_Peak;