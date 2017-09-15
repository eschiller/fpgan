import fpdata
import json
import glob
import numpy as np
import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, tostring, XML
import xml.dom.minidom
import re

def fix_svg(path):
    '''

    :param fp:
    :param path:
    :return:
    '''

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = xml.etree.ElementTree.tostring(elem, 'utf-8')
    reparsed = xml.dom.minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="\t")

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


def is_float(s):
    '''
    Tests whether a string is a float value or not

    :param s: string to test
    :return: True if string is a float, else False
    '''
    try:
        float(s)
        return True
    except ValueError:
        return False


class fpdatamgr:
    def __init__(self, real_x_dim=32, real_y_dim=32, np_x_dim=64, np_y_dim=64, debug=False):
        self.debug = debug
        self.real_x_dim = 32
        self.real_y_dim = 32
        self.np_x_dim = np_x_dim
        self.np_y_dim = np_y_dim
        self.fplist = []
        self.samples = []
        self.wall_threshold = .95


    def __str__(self):
        return_string = ""
        for fp in self.fplist:
            return_string += str(fp)
        return return_string


    def get_data_fp(self, index):
        return self.fplist[index]


    def get_sample_fp(self, index):
        return self.samples[index]


    def add_fp(self, fp):
        '''
        Adds the passed fpdata object to the end of the fplist list

        :param fp: fpdata object to add
        '''
        self.fplist.append(fp)


    def add_sample_fp(self, fp):
        '''
        Adds the passed fpdata object to the end of the samples list

        :param fp:  fpdata object to add
        '''
        self.samples.append(fp)


    def import_json_file(self, filepath):
        jsondata = {}
        with open(filepath) as datafile:
            jsondata = json.load(datafile)

            #if self.debug == True:
            #    print("In import_json_file. jsondata is:")
            #    print(jsondata)

            for key, jsonfp in jsondata.iteritems():
                if self.debug == True:
                    print("In import_json_file. jsonfp is:")
                    print(jsonfp)
                fp = fpdata.fpdata(np_x_dim=self.np_x_dim, np_y_dim=self.np_y_dim)
                for path in jsonfp:
                    if self.debug == True:
                        print("In import_json_file. path is:")
                        print(path)
                    fp.add_path(1, path["p1x"], path["p1y"], path["p2x"], path["p2y"])
                fp.normalize()
                self.add_fp(fp)



    def import_svg_file(self, glob_string, target="data"):
        '''
        Will import an existing svg file (or files) to the fpdatamgr. Target
        can either be the fpdatamgr data list or the fpdatamgr sample list.

        :param glob_string:
        :param target:
        :return:
        '''
        files_added = 0

        file_glob = glob.glob(glob_string)

        for file_path in file_glob:
            files_added += 1
            print("importing file " + file_path)
            walls = []
            cursor = {"x": 0.0, "y": 0.0}
            paths = get_svg_layer_paths(file_path)
            path_arrays = []
            for path in paths:
                path_arrays.append(re.split('[\s,]', path))

            for path in path_arrays:
                if self.debug == True:
                    print("full path is " + str(path))
                    print("starting a path")
                #get starting coordinates to handle z command
                start_coord = {"x": 0.0, "y": 0.0}
                first_command = path.pop(0)

                if self.debug == True:
                    print("first command is " + first_command)

                if first_command == "m":
                    if self.debug == True:
                        print("command is m, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))

                    cursor["x"] = float(path.pop(0))
                    cursor["y"] = float(path.pop(0))
                    start_coord["x"] = cursor["x"]
                    start_coord["y"] = cursor["y"]
                if first_command == "M":
                    if self.debug == True:
                        print("command is M, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                    cursor["x"] = float(path.pop(0))
                    cursor["y"] = float(path.pop(0))
                    start_coord["x"] = cursor["x"]
                    start_coord["y"] = cursor["y"]


                while len(path) > 0:
                    awall = []
                    command = path.pop(0)

                    #wall segment from current cursor to start
                    if (command == "z") or (command == "Z"):
                        if self.debug == True:
                            print("command is z, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        awall.append(start_coord["x"])
                        awall.append(start_coord["y"])
                        walls.append(awall)

                    #wall segment from current cursor to relative location
                    elif command == "l":
                        if self.debug == True:
                            print("command is l, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        cursor["x"] += float(path.pop(0))
                        cursor["y"] += float(path.pop(0))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        walls.append(awall)

                    #wall segment from current cursor to absolute location
                    elif command == "L":
                        if self.debug == True:
                            print("command is L, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        cursor["x"] = float(path.pop(0))
                        cursor["y"] = float(path.pop(0))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        walls.append(awall)

                    #wall segment from cursor to relative horizontal loc
                    elif command == "h":
                        if self.debug == True:
                            print("command is h, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        cursor["x"] += float(path.pop(0))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        walls.append(awall)

                    #wall segment from cursor to absolut horizontal loc
                    elif command == "H":
                        if self.debug == True:
                            print("command is H, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        cursor["x"] = float(path.pop(0))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        walls.append(awall)

                    #wall segment from cursor to relative vertical loc
                    elif command == "v":
                        if self.debug == True:
                            print("command is v, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        cursor["y"] += float(path.pop(0))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        walls.append(awall)

                    #wall segment from cursor to absolute vertical loc
                    elif command == "V":
                        if self.debug == True:
                            print("command is V, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        cursor["y"] = float(path.pop(0))
                        awall.append(cursor["x"])
                        awall.append(cursor["y"])
                        walls.append(awall)

                    #just a coordinate indicating continuation of 'm' command
                    #todo extend this to include instances where first command is 'M' and subsequent floats are absolute values
                    elif is_float(command):
                        if first_command == "M":
                            if self.debug == True:
                                print("command is a float value with M, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                            awall.append(cursor["x"])
                            awall.append(cursor["y"])
                            cursor["x"] = float(command)
                            cursor["y"] = float(path.pop(0))
                            awall.append(cursor["x"])
                            awall.append(cursor["y"])
                            walls.append(awall)
                        elif first_command == "m":
                            if self.debug == True:
                                print("command is a float value with m, cursor at " +  str(cursor["x"]) + "," + str(cursor["y"]))
                            awall.append(cursor["x"])
                            awall.append(cursor["y"])
                            cursor["x"] += float(command)
                            cursor["y"] += float(path.pop(0))
                            awall.append(cursor["x"])
                            awall.append(cursor["y"])
                            walls.append(awall)

                    else:
                        print("Error: unknown svg path command: " + command)

            #create a new floorplan and add a path for each wall we've parsed from the svg
            fp1 = fpdata.fpdata(np_x_dim=self.np_x_dim, np_y_dim=self.np_y_dim)
            for wall in walls:
                if self.debug == True:
                    print("adding wall with points " + str(wall[0]) + " " + str(wall[1]) + " " + str(wall[2]) + " " + str(wall[3]))
                fp1.add_path(1, wall[0], wall[1], wall[2], wall[3])

            #add the floorplan and return
            fp1.normalize()
            if target == "data":
                self.add_fp(fp1)
            elif target == "samples":
                self.add_sample_fp(fp1)
            else:
                print("Invalid target. Use either \"data\" or \"samples\"")

        return files_added



    def import_sample_fp(self, np_array):
        '''
        Takes a numpy array of 64, 64, 2 and changes it in to a fpdata object then adds it to the samples
        '''
        fp1 = fpdata.fpdata(np_x_dim=self.np_x_dim, np_y_dim=self.np_y_dim)

        #print("read dim is " + str(self.real_x_dim) + ", np_dim is " + str(self.np_x_dim))

        x_mult = float(self.real_x_dim) / float(self.np_x_dim)
        y_mult = float(self.real_y_dim) / float(self.np_y_dim)

        #print("Multiplier is " + str(x_mult))

        #iterate through the first two dimensions of the matrix looking for values big enough to
        #be considered walls
        for i in range(np.shape(np_array)[0]):
            for j in range(np.shape(np_array)[1]):
                if (np_array[i][j][0] > self.wall_threshold) or (np_array[i][j][1] > self.wall_threshold):
                    #find midpoint
                    midpoint_x = i * x_mult
                    midpoint_y = j * y_mult

                    #print("Midpoint of x is " + str(midpoint_x) + ", midpoint of y is " + str(midpoint_y))

                    #find both x values by  extending past the midpoint one half of the x length, np_array[i][j][0]
                    p1_x = round(midpoint_x - (np_array[i][j][0] / 2))
                    p2_x = round(midpoint_x + (np_array[i][j][0] / 2))
                    p1_y = round(midpoint_y - (np_array[i][j][1] / 2))
                    p2_y = round(midpoint_y + (np_array[i][j][1] / 2))

                    #add the path we've found to the fpdata object
                    fp1.add_path(1, p1_x, p1_y, p2_x, p2_y)

                    #print("Adding index " + str(i) + "," + str(j) + " with values " + str(np_array[i][j][0]) + "," + str(np_array[i][j][1]) + " as path with values " + str(p1_x) + "," + str(p1_y) + "; " + str(p2_x) + "," + str(p2_y))
        self.add_sample_fp(fp1)


    def export_svg(self, index, filename, export_dir="./", source="samples"):
        if source == "data":
            target_fp = self.get_data_fp(index)
        else:
            target_fp = self.get_sample_fp(index)

        fp_paths = target_fp.get_paths()

        top = Element('svg', {"xmlns:dc": "http://purl.org/dc/elements/1.1/",
                              "xmlns:cc": "http://creativecommons.org/ns#",
                              "xmlns:rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                              "xmlns:svg": "http://www.w3.org/2000/svg",
                              "xmlns": "http://www.w3.org/2000/svg",
                              "xmlns:sodipodi": "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
                              "xmlns:inkscape": "http://www.inkscape.org/namespaces/inkscape",
                              "width": "180mm",
                              "height": "180mm",
                              "viewBox": "0 0 180 180",
                              "version": "1.1",
                              "id": "svg8",
                              "inkscape:version": "0.92.0 r",
                              "sodipodi:docname": filename,
                              #"inkscape:export-filename": "./" + filename + ".png",
                              "inkscape:export-xdpi": "96",
                              "inkscape:export-ydpi": "96"
                              })
        defs = SubElement(top, "defs", {"id": "defs2"})
        sodipodi = SubElement(top, "sodipodi:namedview", {"id": "base",
                                                          "pagecolor": "#ffffff",
                                                          "bordercolor": "#666666",
                                                          "borderopacity": "1.0",
                                                          "inkscape:pageopacity": "0.0",
                                                          "inkscape:pageshadow": "2",
                                                          "inkscape:zoom": "1.0",
                                                          "inkscape:cx": "100.0",
                                                          "inkscape:cy": "100.0",
                                                          "inkscape:document-units": "mm",
                                                          "inkscape:current-layer": "layer1",
                                                          "showgrid": "false",
                                                          "inkscape:window-width": "2000",
                                                          "inkscape:window-height": "900",
                                                          "inkscape:window-x": "0",
                                                          "inkscape:window-y": "1",
                                                          "inkscape:window-maximized": "0"
                                                          })

        metadata = SubElement(top, "metadata", {"id": "metadata5"})
        rdf = SubElement(metadata, "rdf:RDF")
        ccWork = SubElement(rdf, "cc:Work", {"rdf:about": ""})
        dcformat = SubElement(ccWork, "dc:format").text = "image/svg+xml"
        dctype = SubElement(ccWork, "dc:type", {"rdf:resource": "http://purl.org/dc/dcmitype/StillImage"})
        dctitle = SubElement(ccWork, "dc:title")

        g1 = SubElement(top, "g", {"inkscape:label": "Layer 1",
                                   "inkscape:groupmode": "layer",
                                   "id": "layer1",
                                   "style": "display:inline"})

        pindex = 3600
        paths = []
        for some_path in fp_paths:
            if some_path['pathtype'] == 1.0:
                paths.append(SubElement(g1, "path", {
                    "style": "fill:none;stroke:#000000;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1",
                    "d": "M " + str(some_path['p1x']) + "," + str(some_path['p1y']) + " L " + str(
                        some_path['p2x']) + "," + str(some_path['p2y']),
                    "id": "path" + str(pindex),
                    "inkscape:connector-curvature": "0"
                    }))
                pindex += 1

        filepath = export_dir + filename
        with open(filepath, 'w') as f:
            f.write(prettify(top))
            f.write("\n")


    def export_json_file(self, filename="output.json"):
        '''
        Outputs all floorplans in the fplist to a well-formatted JSON file

        :param filename: path of the output file
        '''
        index = 0
        fpdict = {}
        for fp in self.fplist:
            fpdict[str(index)] = fp.paths
            index += 1

        jsonstr =  str(json.dumps(fpdict, sort_keys=True, indent=4, separators=(',', ': ')))
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

        ret_mat = np.zeros((count, self.np_x_dim, self.np_y_dim, 2))
        i = 0

        for mgr_index in range(start_index, end_index):
            ret_mat[i] = self.fplist[mgr_index].to_numpy_array()
            i += 1

        return ret_mat



    def generate_data_set(self, size=100):
        '''
        Creates a data set for feeding to a NN (in numpy format) from the
        available floorplans from the data.
        :param size:
        :return:
        '''
        #figure out how many floorplans we have and get them all as numpy mats
        fpcount = len(self.fplist)
        all_fp_as_np = self.to_numpy_array(0, fpcount)

        #get full matrix
        ret_mat = np.zeros((size, self.np_x_dim, self.np_y_dim, 2))

        #just cycle through our available fps until we run out of slots to fill
        for i in range(size):
            ret_mat[i] = all_fp_as_np[i % fpcount]

        return ret_mat



    def generate_test_set(self, size=100):

        fp1 = fpdata.fpdata(np_x_dim=self.np_x_dim, np_y_dim=self.np_y_dim)
        fp1.add_path(1, 6.0, 6.0, 6.0, 26.0)
        fp1.add_path(1, 6.0, 26.0, 26.0, 26.0)
        fp1.add_path(1, 26.0, 26.0, 26.0, 6.0)
        fp1.add_path(1, 26.0, 6.0, 6.0, 6.0)
        fp1.add_path(1, 16.0, 6.0, 16.0, 26.0)
        fp1.normalize()
        for i in range(size):
            self.add_fp(fp1)

        return self.to_numpy_array(0, (size - 1))


    def generate_svg_test_set(self, path_name, size):
        self.import_svg_file(path_name)
        if self.debug == True:
            print(self.get_data_fp(-1))
        fp1 = self.get_data_fp(-1)

        for i in range(size):
            self.add_fp(fp1)

        return self.to_numpy_array(0, (size - 1))


    def fix_and_export_data(self, export_dir="./"):
        index = 0
        for fp in self.fplist:
            fp.normalize()
            fp.rescale(32, True)
            filename = str(index) + ".svg"
            self.export_svg(index, filename, export_dir, "data")
            index += 1
