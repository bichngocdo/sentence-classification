import argparse

from sentclf.data import read_json_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='prepare_data.py',
        description='Data preparation for scientific paper sentence classification (SPSC)'
    )
    parser.add_argument('input', type=str,
                        help='input data folder of json files')
    parser.add_argument('output', type=str,
                        help='output csv file')

    args = parser.parse_args()
    print(args)

    data = read_json_folder(args.input)
    data.to_csv(args.output, index=False)
