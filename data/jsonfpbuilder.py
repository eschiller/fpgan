import json


class json_fp:
    def __init__(self):
        paths = []


    def import_npfp(self, np_fp):




def get_json_fp(paths):
    '''
    Takes an array of dicts like "{"p1_x": 30, "p1_y":130, "p2_x":130, "p2_y":130}" and
    converts to a json formatted string

    :param paths: dicts in form "{"p1_x": <value>, "p1_y": <value>, "p2_x": <value>, "p2_y": <value>}
    :return: json formatted string
    '''
    json_fp = { "pathlist": paths }
    return str(json.dumps(json_fp, sort_keys=True, indent=4, separators=(',', ': ')))



def write_json_fp_to_file(paths, filename="output.json"):
    '''

    :param paths:
    :param filename:
    :return:
    '''
    with open(filename, 'w') as f:
        f.write(get_json_fp(paths))
        f.write("\n")
