//
// Created by hunglv on 21/12/2018.
//

#ifndef OPENNI_GRABBER_LOCATION_H
#define OPENNI_GRABBER_LOCATION_H

#endif //OPENNI_GRABBER_LOCATION_H

class Location {
public:
    double max_x, max_y, min_x, min_y;
    Location(double max_x, double min_x, double max_y, double min_y) {
        this->max_x = max_x;
        this->min_x = min_x;
        this->max_y = max_y;
        this->min_y = min_y;
    }
};