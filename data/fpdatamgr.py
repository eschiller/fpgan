import fpdata
import json
import numpy as np
import xml.etree.ElementTree
import re


def get_svg_layer_paths(file_path, layer="layer1"):
    e = xml.etree.ElementTree.parse(file_path).getroot()
    layer_xml = None
    paths = []

    for child in e.iter('{http://www.w3.org/2000/svg}g'):
        if child.attrib['id'] == layer:
            layer_xml = child

    for path in layer_xml:
        paths.append(path.attrib['d'])

    return paths


class fpdatamgr:
    def __init__(self, np_x_dim=8, np_y_dim=8):
        self.np_x_dim = np_x_dim
        self.np_y_dim = np_y_dim
        self.fplist = []


    def __str__(self):
        return_string = ""
        for fp in self.fplist:
            return_string += str(fp)
        return return_string


    def add_fp(self, fp):
        '''
        Adds the passed fpdata object to the end of the fplist list

        :param fp: fpdata object to add
        '''
        self.fplist.append(fp)


    def import_svg_file(self, file_path):
        walls = []
        cursor = {"x": 0.0, "y": 0.0}
        paths = get_svg_layer_paths(file_path)
        path_arrays = []
        for path in paths:
            path_arrays.append(re.split('[\s,]', path))

        for path in path_arrays:
            print("full path is " + str(path))
            print("starting a path")
            #get starting coordinates to handle z command
            start_coord = {"x": 0.0, "y": 0.0}
            first_command = path.pop(0)
            print("first command is " + first_command)
            if first_command == "m":
                cursor["x"] += float(path.pop(0))
                cursor["y"] += float(path.pop(0))
                start_coord["x"] = cursor["x"]
                start_coord["y"] = cursor["y"]
            if first_command == "M":
                cursor["x"] = float(path.pop(0))
                cursor["y"] = float(path.pop(0))
                start_coord["x"] = cursor["x"]
                start_coord["y"] = cursor["y"]

            while len(path) > 0:
                awall = []
                command = path.pop(0)

                #wall segment from current cursor to start
                if (command == "z") or (command == "Z"):
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    awall.append(start_coord["x"])
                    awall.append(start_coord["y"])
                    walls.append(awall)

                #wall segment from current cursor to relative location
                elif command == "l":
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["x"] += float(path.pop(0))
                    cursor["y"] += float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from current cursor to absolute location
                elif command == "L":
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["x"] = float(path.pop(0))
                    cursor["y"] = float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from cursor to relative horizontal loc
                elif command == "h":
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["x"] += float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from cursor to absolut horizontal loc
                elif command == "H":
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["x"] = float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from cursor to relative vertical loc
                elif command == "v":
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["y"] += float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from cursor to absolute vertical loc
                elif command == "V":
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["y"] = float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

        #create a new floorplan and add a path for each wall we've parsed from the svg
        fp1 = fpdata.fpdata()
        for wall in walls:
            print("adding wall with points " + str(wall[0]) + " " + str(wall[1]) + " " + str(wall[2]) + " " + str(wall[3]))
            fp1.add_path(1, wall[0], wall[1], wall[2], wall[3])

        #add the floorplan and return
        self.add_fp(fp1)

        return 0



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

        return self.to_numpy_array(0, (size - 1))
