import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str,
                    default='2024-08-13_20-30-09.xlsx',
                    help='xlsx file to convert')
parser.add_argument('--sheet_name', type=str,
                    default='aclt',
                    help='sheet name')
parser.add_argument('--target', type=str,
                    default='md',
                    choices=['md', 'tex'],
                    help='target format')
args = parser.parse_args()


if __name__ == "__main__":
    sheet = pd.read_excel(args.file_name, sheet_name=args.sheet_name)
    if args.target == 'md':
        message = sheet.to_markdown()
        print(message)
    elif args.target == 'tex':
        message = sheet.to_latex()
        print()
    else:
        print("Unsupported target format")
    