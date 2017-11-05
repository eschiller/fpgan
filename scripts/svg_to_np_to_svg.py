#!/usr/bin/python
import sys
sys.path.append("../data")
import fpdatamgr
import fpdata

help_msg = '''
svg_to_np_to_svg takes as arguments the np dimension (assumes x and y are equal)
and the number of the vector drawing to convert. It will then convert it to
np data format, then convert back to svg.

Essentially, it's intended to show data loss between svg and numpy formats.

This script must be run with two arguments.

svg_to_np_to_svg.py 8 4

'''

datamgr = fpdatamgr.fpdatamgr(np_x_dim=int(sys.argv[1]), np_y_dim=int(sys.argv[1]))
datamgr.import_svg_file("../data/vec/" + sys.argv[2] + ".svg")
datamgr.import_svg_file("../data/vec/1.svg")
nparr = datamgr.to_numpy_array(0, 1)
rescaled_samples = fpdata.np_rescale(nparr, snap=False)
datamgr.import_sample_fp(rescaled_samples[0])
datamgr.export_svg(0, sys.argv[2] + "id.svg")
