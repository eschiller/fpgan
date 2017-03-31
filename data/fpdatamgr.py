import fpdata
import json
import numpy as np
import xml.etree.ElementTree
from xml.etree.ElementTree import Element, SubElement, tostring, XML
import xml.dom.minidom
import re



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


class fpdatamgr:
    def __init__(self, np_x_dim=8, np_y_dim=8):
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
        self.samples.append(fp)


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
                print("command is m, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))

                cursor["x"] = float(path.pop(0))
                cursor["y"] = float(path.pop(0))
                start_coord["x"] = cursor["x"]
                start_coord["y"] = cursor["y"]
            if first_command == "M":
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
                    print("command is z, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    awall.append(start_coord["x"])
                    awall.append(start_coord["y"])
                    walls.append(awall)

                #wall segment from current cursor to relative location
                elif command == "l":
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
                    print("command is h, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["x"] += float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from cursor to absolut horizontal loc
                elif command == "H":
                    print("command is H, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["x"] = float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from cursor to relative vertical loc
                elif command == "v":
                    print("command is v, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["y"] += float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                #wall segment from cursor to absolute vertical loc
                elif command == "V":
                    print("command is V, cursor at " + str(cursor["x"]) + "," + str(cursor["y"]))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    cursor["y"] = float(path.pop(0))
                    awall.append(cursor["x"])
                    awall.append(cursor["y"])
                    walls.append(awall)

                else:
                    print("Error: unknown svg path command: " + command)

        #create a new floorplan and add a path for each wall we've parsed from the svg
        fp1 = fpdata.fpdata()
        for wall in walls:
            print("adding wall with points " + str(wall[0]) + " " + str(wall[1]) + " " + str(wall[2]) + " " + str(wall[3]))
            fp1.add_path(1, wall[0], wall[1], wall[2], wall[3])

        #add the floorplan and return
        fp1.normalize()
        self.add_fp(fp1)

        return 0


    def import_sample_fp(self, np_array):
        fp1 = fpdata.fpdata()

        for i in range(np.shape(np_array)[0]):
            for j in range(np.shape(np_array)[1]):
                if np_array[i][j][0] > self.wall_threshold:
                    fp1.add_path(1.0, np_array[i][j][1], np_array[i][j][2], np_array[i][j][3], np_array[i][j][4])


        self.add_sample_fp(fp1)


    def export_svg(self, index, filename, source="samples"):
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
                              "sodipodi:docname": filename + ".svg",
                              "inkscape:export-filename": "./" + filename + ".png",
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

        with open("./" + filename + ".svg", 'w') as f:
            f.write(prettify(top))
            f.write("\n")


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

        fp1 = fpdata.fpdata()
        fp1.add_path(1, 30.0, 30.0, 30.0, 130.0)
        fp1.add_path(1, 30.0, 130.0, 130.0, 130.0)
        fp1.add_path(1, 130.0, 130.0, 130.0, 30.0)
        fp1.add_path(1, 130.0, 30.0, 30.0, 30.0)
        fp1.add_path(1, 80.0, 30.0, 80.0, 130.0)
        fp1.normalize()
        for i in range(size):
            self.add_fp(fp1)

        return self.to_numpy_array(0, (size - 1))


    def generate_svg_test_set(self, path_name, size):
        self.import_svg_file(path_name)
        print(self.get_data_fp(-1))
        fp1 = self.get_data_fp(-1)

        for i in range(size):
            self.add_fp(fp1)

        return self.to_numpy_array(0, (size - 1))