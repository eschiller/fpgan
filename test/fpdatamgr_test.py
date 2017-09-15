import sys
sys.path.append("../data")
from fpdatamgr import fpdatamgr
import fpdata
import numpy as np

np.set_printoptions(threshold=np.nan)

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
mgr1.import_svg_file("./56Leonard_1.svg", target="samples")
print(mgr1)
mgr1.get_sample_fp(-1).rescale(multiplier=32)
mgr1.export_svg(-1, "./id_56Leonard.svg")


print("**********************************")
print("TESTING NUMPY FORMAT TO SVG EXPORT")
print("**********************************")
print("importing svg again")
mgr2 = fpdatamgr()
#mgr2.import_svg_file("./56Leonard_1.svg")
ds2 = mgr2.generate_test_set()
#print("coverted fp to numpy:")
#print ds2[0]
ds2_rs = fpdata.np_rescale(ds2, snap=False)
#print("rescaled dataset")
#print ds2_rs[0]
smp = ds2_rs[0]
mgr2.import_sample_fp(smp)
mgr2.export_svg(-1, "./id_56Leonard_from_np.svg")
