import argparse

from database import functions

parser = argparse.ArgumentParser()

# client port
parser.add_argument('--client', default='mongodb://localhost:27017/', help='client port')
# database name
parser.add_argument('--database', default='mydatabase', help='database name')
# collection name
parser.add_argument('--collection', default='hessian_contour', help='collection name')

args = parser.parse_args()

dropStatus = functions.drop(args.client, args.database, args.collection)
print(dropStatus)