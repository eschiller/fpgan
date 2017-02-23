import json

def get_json_fp(paths):
    '''

    :param paths:
    :return:
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
