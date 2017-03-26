import sys
sys.path.append("../data")
from fpdatamgr import fpdatamgr

print("**********************************")
print("TESTING TEST SET GENERATION")
print("**********************************")

mgr1 = fpdatamgr()
ds1 = mgr1.generate_test_set()
print(ds1[0])
print(ds1[1])

print("**********************************")
print("TESTING SVG IMPORT")
print("**********************************")
mgr1.import_svg_file("./56Leonard_1.svg")
print(mgr1)
mgr1.export_svg(-1, "./id_56Leonard")