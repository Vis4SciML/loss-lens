import argparse

from database import functions

parser = argparse.ArgumentParser()

# client port
parser.add_argument('--client', default='mongodb://localhost:27017/', help='client port')
# database name
parser.add_argument('--database', default='mydatabase', help='database name')
# collection name
parser.add_argument('--collection', default='hessian_contour', help='collection name')
# query
parser.add_argument('--query', default='{}', help='query')
# delete_option
parser.add_argument('--delete_option', default='one', help='one/many documents to be deleted')

args = parser.parse_args()

if args.delete_option == 'one':
    deleteStatus = functions.deleteOne(args.client, args.database_name, args.collection_name, args.query)
    print(deleteStatus)
elif args.delete_option == 'many':
    deleteStatus = functions.deleteMany(args.client, args.database_name, args.collection_name, args.query)
    print(deleteStatus)
else:
    print('Invalid delete option')