import numpy as np


def new_fp():
    return np.zeros([8, 8, 5])


def add_wall(loc_x, loc_y, p1_x, p1_y, p2_x, p2_y, npfp=None):

    if npfp == None:
        npfp = new_fp()


    npfp[loc_x, loc_y, 0] = 1
    npfp[loc_x, loc_y, 1] = p1_x
    npfp[loc_x, loc_y, 2] = p1_y
    npfp[loc_x, loc_y, 3] = p2_x
    npfp[loc_x, loc_y, 4] = p2_y

    return npfp


def add_door(loc_x, loc_y, p1_x, p1_y, p2_x, p2_y, npfp=None):

    if not npfp == None:
        npfp = new_fp()

    npfp[loc_x, loc_y, 0] = 2
    npfp[loc_x, loc_y, 1] = p1_x
    npfp[loc_x, loc_y, 2] = p1_y
    npfp[loc_x, loc_y, 3] = p2_x
    npfp[loc_x, loc_y, 4] = p2_y

    return npfp


def get_test_fp():
    fp = add_wall(4, 0, 30, 130, 130, 130)
    fp = add_wall(0, 4, 30, 30, 30, 130, fp)
    fp = add_wall(7, 4, 130, 30, 130, 130, fp)
    fp = add_wall(4, 7, 30, 30, 130, 30, fp)
    return fp
