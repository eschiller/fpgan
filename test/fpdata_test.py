import sys
sys.path.append("../data")
from fpdata import fpdata

fp1 = fpdata()
fp1.add_path(1, 30, 130, 130, 130)
fp1.add_path(1, 130, 130, 130, 30)
fp1.add_path(1, 130, 30, 30, 30)
fp1.add_path(1, 30, 30, 30, 130)
print(fp1)

fp1.normalize()
print(fp1)

npfp1 = fp1.to_numpy_array()
print(npfp1)
