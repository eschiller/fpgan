import numpy as np
import math
import random

def mid_x(path):
    return (path["p1x"] + path["p2x"]) / 2.0


def mid_y(path):
    return (path["p1y"] + path["p2y"]) / 2.0


def np_rescale(npdata, multiplier=32, snap=True):
    new_mat = np.zeros(npdata.shape)
    for i in range(npdata.shape[0]):
        for j in range(npdata.shape[1]):
            for k in range(npdata.shape[2]):
                new_mat[i][j][k][0] = npdata[i][j][k][0] * multiplier
                new_mat[i][j][k][1] = npdata[i][j][k][1] * multiplier
    if snap:
        return np.rint(new_mat)
    else:
        return new_mat


class fpdata:
    '''
    Basic data object for managing a single floorplan
    '''
    def __init__(self, np_x_dim=64, np_y_dim=64):
        self.paths = []
        self.np_x_dim = np_x_dim
        self.np_y_dim = np_y_dim


    def __str__(self):
        return str(self.paths)


    def copy(self, orig):
        '''
        copys an existing fp's paths to an empty fp
        :param orig:
        :return:
        '''
        for path in orig.paths:
            self.add_path(path["pathtype"], path["p1x"], path["p1y"], path["p2x"], path["p2y"])



    def add_path(self, type_value, p1x_value, p1y_value, p2x_value, p2y_value):
        '''
        Add single path (either a wall or a door) to the floor plan

        :param type_value: int value. Currently should be either 1 for wall or 2 for door
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


    def remove_point_paths(self):
        '''
        removes any paths where both points are identical
        '''
        for path in self.paths:
            if path["p1x"] == path["p2x"]:
                if path["p1y"] == path["p2y"]:
                    print("removing path")
                    print(path)
                    self.paths.remove(path)


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
        NumPy array will be size (self.np_x_dim, self.np_y_dim, 2). This function
        arranges data as follows:
         todo - write out algorithm here

        :return: NumPy representation of fpdata object
        '''

        #make the numpy matrix
        ret_mat = np.zeros((self.np_x_dim, self.np_y_dim, 2))

        for path in self.paths:
            #get midpoint. since this is the midpoint on an 2x sized grid, we don't need to divide by 2
            #also, we're storing the values internally as 0-1 floats to be better used by the GAN,
            #but for purposes of the grid placement the values are 0-32 ints. This needs to be
            #rescaled before minding the "midpoint
            #
            #Note, there's a bit of a fencepost problem here. The vector drawings are scales like 0 - 64,
            #which actually contains 65 integer values, but we want to keep indices even (ideally some 2^x)
            #I'm usuing a hacky solution to just combine the first and second indicies by saying if it's
            # < 0, make it 0.
            xmid_rs = round((path['p1x'] + path['p2x']) * (self.np_x_dim / 2)) - 1
            ymid_rs = round((path['p1y'] + path['p2y']) * (self.np_y_dim / 2)) - 1

            if xmid_rs < 0:
                xmid_rs = 0
            if ymid_rs < 0:
                ymid_rs = 0



            #get "lowest point" from path. The rules for lowest are if either
            #point has a lower y, that is the lowest. If their y's are equal,
            #take whichever has the lowest x.
            lowpoint = ""
            if path['p1y'] < path['p2y']:
                lowpoint = "p1"
            elif path['p1y'] > path['p2y']:
                lowpoint = "p2"
            #if we get this far, the y's were equal and we're checking x's
            elif path['p1x'] < path['p2x']:
                lowpoint = "p1"
            elif path['p1x'] > path['p2x']:
                lowpoint = "p2"
            else:
                print "ERROR: two points are equal, can't find the lowest."
                print("p1 = " + str(path['p1x']) + "," + str(path['p1y']) + ", p2 = " + str(path['p2x']) + "," + str(path['p2y']))
                return

            #now we find the points after we've translated the lowest to the
            #origin, and the higher by and equal amount. Of course, the lowest
            #after this translation is just the origin, so really, we just need
            #to find the translated values of the highest
            transhighx = transhighy = 0
            if lowpoint == "p1":
                transhighx = path["p2x"] - path["p1x"]
                transhighy = path["p2y"] - path["p1y"]
            elif lowpoint == "p2":
                transhighx = path["p1x"] - path["p2x"]
                transhighy = path["p1y"] - path["p2y"]

            #finally, if the path is longer than previous path, place the translated high point on
            # the grid at the found midpoint
            if (transhighy + transhighx) > (ret_mat[int(xmid_rs), int(ymid_rs), 0] + ret_mat[int(xmid_rs), int(ymid_rs), 1]):
                ret_mat[int(xmid_rs), int(ymid_rs), 0] = transhighx
                ret_mat[int(xmid_rs), int(ymid_rs), 1] = transhighy

            #print("Converting path " + str(path['p1x']) + "," + str(path['p1y']) + "; " + str(path['p2x']) + "," + str(path['p2y']) + " to a " + str(transhighx) + "," + str(transhighy) + " vec at loc" + str(xmid_rs) + "," + str(ymid_rs))


        #finally return
        return ret_mat

    def rnd_rescale(self):
        '''
        Will rescale down 0 - 10 percent
        :return:
        '''
        adjustment = random.random() / 10.0
        scale = 1 - adjustment
        for path in self.paths:
            path["p1x"] = path["p1x"] * scale
            path["p1y"] = path["p1y"] * scale
            path["p2x"] = path["p2x"] * scale
            path["p2y"] = path["p2y"] * scale