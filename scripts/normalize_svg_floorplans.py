#!/usr/bin/python
import sys
sys.path.append("../data")
import fpdatamgr

help_msg = '''
normalize_svg_floorplans takes existing svg floorplans, and converts them to a
common scale, then re-exports them as svgs.

This script must be run with two arguments. The first argument is the source
svg files, then second is the directory to export the resulting svg files to.
File globs are acceptable in the first argument (source files), but of used,
the argument must be in quotes (to avoid interpolation by the shell. So the
following would be acceptable:

normalize_svg_floorplans.py "./src/*.svg" ./dest/

'''

if len(sys.argv) != 3:
    print("Incorrect number of args (2 needed): " + str(len(sys.argv)))
    print help_msg
    exit(1)

mgr = fpdatamgr.fpdatamgr()
mgr.import_svg_file(sys.argv[1])
mgr.fix_and_export_data(sys.argv[2])