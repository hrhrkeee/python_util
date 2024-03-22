import argparse


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--source', type=str, required=True, help="source")
    parser.add_argument('-t', '--target', type=str, required=True, help="target")

    args = parser.parse_args()

    return args


def main(args):
    
    print(args.source)
    print(args.target)
    
    pass

if __name__ == '__main__':
    args = parse_args()
    main(args)