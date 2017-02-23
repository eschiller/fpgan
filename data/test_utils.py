import numpy as np
import npfpbuilder
import jsonfpbuilder

fp = np.zeros([8, 8, 5])

#square fp in numpy
fp = npfpbuilder.add_wall(fp, 4, 0, 30, 130, 130, 130)
fp = npfpbuilder.add_wall(fp, 0, 4, 30, 30, 30, 130)
fp = npfpbuilder.add_wall(fp, 7, 4, 130, 30, 130, 130)
fp = npfpbuilder.add_wall(fp, 4, 7, 30, 30, 130, 30)

print(fp)


#square fp in json
path1 = {"p1_x": 30, "p1_y":130, "p2_x":130, "p2_y":130}
path2 = {"p1_x": 30, "p1_y":30, "p2_x":30, "p2_y":130}
path3 = {"p1_x": 130, "p1_y":30, "p2_x":130, "p2_y":130}
path4 = {"p1_x": 30, "p1_y":30, "p2_x":130, "p2_y":30}
paths = []
paths.append(path1)
paths.append(path2)
paths.append(path3)
paths.append(path4)

print jsonfpbuilder.get_json_fp(paths)
jsonfpbuilder.write_json_fp_to_file(paths, "test_out.json")