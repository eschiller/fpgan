from xml.etree.ElementTree import Element, SubElement, tostring, XML
import numpy as np

#for pretty xml printing use https://pymotw.com/2/xml/etree/ElementTree/create.html

def buildsvg(name, path_elements):
    top = Element('svg', {  "xmlns:dc":"http://purl.org/dc/elements/1.1/",
                            "xmlns:cc":"http://creativecommons.org/ns#",
                            "xmlns:rdf":"http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                            "xmlns:svg":"http://www.w3.org/2000/svg",
                            "xmlns":"http://www.w3.org/2000/svg",
                            "xmlns:sodipodi":"http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd",
                            "xmlns:inkscape":"http://www.inkscape.org/namespaces/inkscape",
                            "width":"180mm",
                            "height":"180mm",
                            "viewBox":"0 0 180 180",
                            "version":"1.1",
                            "id":"svg8",
                            "inkscape:version":"0.92.0 r",
                            "sodipodi:docname": name + ".svg",
                            "inkscape:export-filename":"./" + name + ".png",
                            "inkscape:export-xdpi":"96",
                            "inkscape:export-ydpi":"96"
                            })
    defs = SubElement(top, "defs", {"id":"defs2"})
    sodipodi = SubElement(top, "sodipodi:namedview", {"id":"base",
                                                      "pagecolor":"#ffffff",
                                                      "bordercolor":"#666666",
                                                      "borderopacity":"1.0",
                                                      "inkscape:pageopacity":"0.0",
                                                      "inkscape:pageshadow":"2",
                                                      "inkscape:zoom":"1.0",
                                                      "inkscape:cx":"100.0",
                                                      "inkscape:cy":"100.0",
                                                      "inkscape:document-units":"mm",
                                                      "inkscape:current-layer":"layer1",
                                                      "showgrid":"false",
                                                      "inkscape:window-width":"2000",
                                                      "inkscape:window-height":"900",
                                                      "inkscape:window-x":"0",
                                                      "inkscape:window-y":"1",
                                                      "inkscape:window-maximized":"0"
                                                      })

    metadata = SubElement(top, "metadata", {"id":"metadata5"})
    rdf = SubElement(metadata, "rdf:RDF")
    ccWork = SubElement(rdf, "cc:Work", {"rdf:about":""})
    dcformat = SubElement(ccWork, "dc:format").text = "image/svg+xml"
    dctype = SubElement(ccWork, "dc:type", {"rdf:resource":"http://purl.org/dc/dcmitype/StillImage"})
    dctitle = SubElement(ccWork, "dc:title")

    g1 = SubElement(top, "g", {"inkscape:label":"Layer 1",
                               "inkscape:groupmode":"layer",
                               "id":"layer1",
                               "style":"display:inline"})

    #create all wall paths
    #for each place in 8x8 or 16x16 grid look for wall
    paths = []
    pindex = 1000
    for i in range(np.shape(path_elements)[0]):
        for j in range(np.shape(path_elements)[1]):
            if path_elements[i][j][0] == 1.0:
                paths.append(SubElement(g1, "path" + str(pindex), {"style":"fill:none;stroke:#000000;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1",
                                    "d":"M " + str(path_elements[i][j][1]) + "," + str(path_elements[i][j][2]) + " L " + str(path_elements[i][j][3]) + "," + str(path_elements[i][j][4]),
                                    "id":"path3799",
                                    "inkscape:connector-curvature":"0"
                                    }))
                pindex += 1



    #TODO ERASE THIS SECTION! THIS IS JUST A TEST!
    '''
    path1 = SubElement(g1, "path", {"style":"fill:none;stroke:#000000;stroke-width:0.26458332px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1",
                                    "d":"M 30,130 L 130,130",
                                    "id":"path3799",
                                    "inkscape:connector-curvature":"0"
                                    })
    '''
    print tostring(top)