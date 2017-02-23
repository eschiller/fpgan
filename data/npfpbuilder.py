#code to create a numpy floorplan

def add_wall(npfp, loc_x, loc_y, p1_x, p1_y, p2_x, p2_y):
    npfp[loc_x, loc_y, 0] = 1
    npfp[loc_x, loc_y, 1] = p1_x
    npfp[loc_x, loc_y, 2] = p1_y
    npfp[loc_x, loc_y, 3] = p2_x
    npfp[loc_x, loc_y, 4] = p2_y

    return npfp


def add_door(npfp, loc_x, loc_y, p1_x, p1_y, p2_x, p2_y):
    npfp[loc_x, loc_y, 0] = 2
    npfp[loc_x, loc_y, 1] = p1_x
    npfp[loc_x, loc_y, 2] = p1_y
    npfp[loc_x, loc_y, 3] = p2_x
    npfp[loc_x, loc_y, 4] = p2_y

    return npfp
