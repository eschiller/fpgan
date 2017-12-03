#!/usr/bin/python
import sys
import json
import matplotlib.pyplot as plt

help_msg = '''
First arg is the filepath, second arg is the type of data (key in the json,
like density).
'''

jsondata = {}

filename = sys.argv[1]

with open(filename) as datafile:
    jsondata = json.load(datafile)
    graphdata = jsondata[sys.argv[2]]

    xaxis = []
    yaxis = []

    for key in sorted(graphdata, key=int):
        xaxis.append(key)
        yaxis.append(graphdata[key])


    plt.plot(xaxis, yaxis)
    plt.show()