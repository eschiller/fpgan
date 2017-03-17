import sys
sys.path.append("../data")
from fpdatamgr import fpdatamgr

mgr1 = fpdatamgr()
ds1 = mgr1.generate_test_set()
print(ds1[0])
print(ds1[1])