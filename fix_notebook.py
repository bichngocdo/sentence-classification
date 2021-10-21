import json
import sys
from collections import OrderedDict


def fix_notebook(fp_in, fp_out):
    with open(fp_in, 'r') as f:
        json_file = json.load(f, object_hook=OrderedDict)

    count = 0
    for cell in json_file['cells']:
        if 'execution_count' in cell and cell['execution_count'] is not None:
            count += 1
            cell['execution_count'] = count

            for output in cell['outputs']:
                if 'execution_count' in output:
                    output['execution_count'] = count

    with open(fp_out, 'w') as f:
        json.dump(json_file, f, indent=1)


if __name__ == '__main__':
    fix_notebook(sys.argv[1], sys.argv[2])
