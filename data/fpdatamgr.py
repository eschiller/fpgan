import fpdata
import json
import numpy as np


class fpdatamgr:
    def __init__(self, np_x_dim=8, np_y_dim=8):
        self.np_x_dim = np_x_dim
        self.np_y_dim = np_y_dim
        self.fplist = []


    def __str__(self):
        for fp in self.fplist:
            print(fp)


    def add_fp(self, fp):
        '''
        Adds the passed fpdata object to the end of the fplist list

        :param fp: fpdata object to add
        '''
        self.fplist.append(fp)


    def to_json_file(self, filename="output.json"):
        '''
        Outputs all floorplans in the fplist to a well-formatted JSON file

        :param filename: path of the output file
        '''
        jsonstr =  str(json.dumps(self.fplist, sort_keys=True, indent=4, separators=(',', ': ')))
        with open(filename, 'w') as f:
            f.write(jsonstr)
            f.write("\n")


    def to_numpy_array(self, start_index, end_index):
        '''
        Creates and returns a numpy array dataset containing floorplans from
        fplist from start_index to end_index

        :param start_index: start index to output to numpy array (as int)
        :param end_index: end index to output to numpy array (as int)
        :return: numpy array representation of the indicated floorplans
        '''
        count = end_index - start_index + 1

        ret_mat = np.zeros((count, self.np_x_dim, self.np_y_dim, 5))
        i = 0

        for mgr_index in range(start_index, end_index):
            ret_mat[i] = self.fplist[mgr_index].to_numpy_array()
            i += 1

        return ret_mat


    def generate_test_set(self, size=100):

        #todo - probably need to pass a "size" param and return a set of that size
        fp1 = fpdata.fpdata()
        fp1.add_path(1, 30, 30, 30, 130)
        fp1.add_path(1, 30, 130, 130, 130)
        fp1.add_path(1, 130, 130, 130, 30)
        fp1.add_path(1, 130, 30, 30, 30)
        fp1.normalize()
        for i in range(size):
            self.add_fp(fp1)

        print self.fplist[3]
        return self.to_numpy_array(0, 99)
