import json
import numpy as np

class metricmgr:
    def __init__(self, filename="./metrics/fpgan_metrics.log"):
        self.log_dict = {}
        self.filename = filename


    def __str(self):
        return_string = ""
        return return_string


    def update_density(self, reps, npsamples):
        if 'density' not in self.log_dict.keys():
            self.log_dict['density'] = {}

        self.log_dict['density'][reps] = str(npsamples.mean())


    def update_quantity(self, reps, npsamples):
        if 'quantity' not in self.log_dict.keys():
            self.log_dict['quantity'] = {}

        # below is a bit much of a one-liner. It's essentailly total walls in the
        # numpy array (making the threshold for a wall .05 normalized), over the
        # number of samples in the array (npsamples.shape[0]) times 2 because you
        # can have an x or y value over .05. Not, this isn't perfect, because it
        # will count diagonal walls twice (x and y above .05), but in practice this
        # doesn't matter much since diagonals are rare.
        self.log_dict['quantity'][reps] = len(npsamples[ np.where(npsamples > .05)]) / (npsamples.shape[0])


    def update_orientation(self, reps, npsamples):
        if 'orientation' not in self.log_dict.keys():
            self.log_dict['orientation'] = {}

        smp_split = np.split(npsamples, [1], 3)
        #xmsk = np.where(smp_split[0] > .05)
        #ymsk = np.where(smp_split[1] > .05)

        walls = 0
        diags = 0

        for i in range(smp_split[0].shape[0]):
            for j in range(smp_split[0].shape[1]):
                for k in range(smp_split[0].shape[2]):
                    if (smp_split[0][i,j,k] > .05) or (smp_split[1][i,j,k] > .05):
                        walls += 1
                        if (smp_split[0][i, j, k] > .05) and (smp_split[1][i, j, k] > .05):
                            diags += 1

        self.log_dict['orientation'][reps] = float(diags) / float(walls)


    def update_all(self, reps, npsamples):
        self.update_density(reps, npsamples)
        self.update_quantity(reps, npsamples)
        self.update_orientation(reps, npsamples)
        self.write_out()


    def write_out(self):
        jsonstr =  str(json.dumps(self.log_dict, sort_keys=True, indent=4, separators=(',', ': ')))
        with open(self.filename, 'w') as f:
            f.write(jsonstr)
            f.write("\n")