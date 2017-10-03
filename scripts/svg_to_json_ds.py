#!/usr/bin/python
import sys
sys.path.append("../data")
import fpdatamgr

help_msg = '''
svn_to_json_ds takes a set of svg files and outputs a single json file that
can be used as a dataset..

This script must be run with two arguments. The first argument is the source
svg files, then second is the resulting json file path. Use quotes around the
first arg if using a wild-card.

svg_to_json_ds.py "./src/*.svg" ./output.json

'''

if len(sys.argv) != 3:
    print("Incorrect number of args (2 needed): " + str(len(sys.argv)))
    print help_msg
    exit(1)

mgr = fpdatamgr.fpdatamgr()
mgr.import_svg_file(sys.argv[1])
mgr.export_json_file(sys.argv[2])