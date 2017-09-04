import numpy as np
import math

def mid_x(path):
    return (path["p1x"] + path["p2x"]) / 2.0


def mid_y(path):
    return (path["p1y"] + path["p2y"]) / 2.0


def np_rescale(npdata, multiplier=160, snap=True):
    new_mat = np.zeros(npdata.shape)
    for i in range(npdata.shape[0]):
        for j in range(npdata.shape[1]):
            for k in range(npdata.shape[2]):
                new_mat[i][j][k][0] = npdata[i][j][k][0]
                for l in range(1, npdata.shape[3]):
                    new_mat[i][j][k][l] = npdata[i][j][k][l] * multiplier
    if snap:
        return np.rint(new_mat)
    else:
        return new_mat


class fpdata:
    '''
    Basic data object for managing a single floorplan
    '''
    def __init__(self, np_x_dim=8, np_y_dim=8):
        self.paths = []
        self.np_x_dim = np_x_dim
        self.np_y_dim = np_y_dim


    def __str__(self):
        return str(self.paths)



    def add_path(self, type_value, p1x_value, p1y_value, p2x_value, p2y_value):
        '''
        Add single path (either a wall or a door) to the floor plan

        :param type_value: string value. Currently should be either "wall" or "door"
        :param p1x_value: point 1 x-value for path
        :param p1y_value: point 1 y-value for path
        :param p2x_value: point 2 x-value for path
        :param p2y_value: point 2 y value for path
        '''
        self.paths.append({"pathtype": type_value, "p1x": p1x_value, "p1y": p1y_value,
                           "p2x": p2x_value, "p2y": p2y_value})


    def get_paths(self):
        '''
        returns all paths
        :return: all paths as a list of dicts
        '''
        return self.paths


    def rescale(self, multiplier=160, snapto=False):
        '''
        Rescales all coordinates in the fpdata object by the amount indicated
        by the multiplier parameter. Also has the ability to snap to integer
        flooring all coordinates

        :param multiplier:
        :param floor:
        '''
        for path in self.paths:
            path["p1x"] *= multiplier
            path["p1y"] *= multiplier
            path["p2x"] *= multiplier
            path["p2y"] *= multiplier
            if snapto:
                path["p1x"] = round(path["p1x"])
                path["p1y"] = round(path["p1y"])
                path["p2x"] = round(path["p2x"])
                path["p2y"] = round(path["p2y"])


    def normalize(self):
        '''
        normalizes the floorplan data, both scaling all internal geometry to y value of 0-160
        and snapping all values to the nearest integer value
        '''
        #move all so that lowest y equals 0
        miny = self.paths[0]["p1y"]

        for apath in self.paths:
            if apath["p1y"] < miny:
                miny = apath["p1y"]

            if apath["p2y"] < miny:
                miny = apath["p2y"]

        y_modifier = miny

        for apath in self.paths:
            for key in apath.keys():
                if (key == 'p1y') or (key == 'p2y'):
                    apath[key] -= y_modifier


        #move all so that the lowest x equals 0
        minx = self.paths[0]["p1x"]

        for apath in self.paths:
            if apath["p1x"] < minx:
                minx = apath["p1x"]

            if apath["p2x"] < minx:
                minx = apath["p2x"]

        x_modifier = minx

        for apath in self.paths:
            for key in apath.keys():
                if (key == 'p1x') or (key == 'p2x'):
                    apath[key] -= x_modifier

        #find max value
        maxval = 0

        for apath in self.paths:
            if apath["p1y"] > maxval:
                maxval = apath["p1y"]

            if apath["p2y"] > maxval:
                maxval = apath["p2y"]

            if apath["p1x"] > maxval:
                maxval = apath["p1x"]

            if apath["p2x"] > maxval:
                maxval = apath["p2x"]


        #alter each value by the amount indicated in the multiplier and snap to nearest int
        for apath in self.paths:
            for key in apath.keys():
                if key != 'pathtype':
                    apath[key] = apath[key] / maxval


    def to_numpy_array(self):
        '''
        Converts and returns a numpy array for the data contained in the fpdata object.
        NumPy array will be size (self.np_x_dim, self.np_y_dim, 5). This function
        arranges data as follows:
         * y-index is determined by dividing all possible y coordinates by the np_y_dim size,
         creating a number of y-slices, and putting each path in a slice according to the y-value
         of it's midpoint
         * paths in each slice are sorted according to their relative x-midpoint, then
         are centered and placed according to thier sort location.

        :return: NumPy representation of fpdata object
        '''
        # init y slots
        y_cats = [None] * self.np_y_dim
        for i in range(self.np_y_dim):
            y_cats[i] = []

        #git slot size so that we can put it in the right y slot
        slot_size = 1.0 / self.np_y_dim

        #put each path in the correct category
        for apath in self.paths:
            slot_index, scratch = divmod(mid_y(apath), slot_size)
            y_cats[int(slot_index) - 1].append(apath)


        #sort each category
        for i in range(len(y_cats)):
            for j in range(len(y_cats[i]) - 1):
                for k in range(len(y_cats[i]) - 1):
                    if y_cats[i][k] > y_cats[i][k + 1]:
                        temp = y_cats[i][k]
                        y_cats[i][k] = y_cats[i][k + 1]
                        y_cats[i][k + 1] = temp

        #make the numpy matrix and place everything appropriately
        ret_mat = np.zeros((self.np_x_dim, self.np_y_dim, 5))
        for i in range(len(y_cats)):
            offset, _scratch = divmod((self.np_x_dim - len(y_cats[i])), 2)
            for path in y_cats[i]:
                ret_mat[offset, i, 0] = path['pathtype']
                ret_mat[offset, i, 1] = path['p1x']
                ret_mat[offset, i, 2] = path['p1y']
                ret_mat[offset, i, 3] = path['p2x']
                ret_mat[offset, i, 4] = path['p2y']
                offset += 1

        #finally return
        return ret_mat
