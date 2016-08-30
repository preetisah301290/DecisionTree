__author__ = 'preeti_sah'
import pickle
import csv
import sys
import getopt

from Tree_Model import Tree

def read_file(argv):
       """
       This will read a file and store the data in the list
       :param argv: file name from command line
       :return: input file
       """
       inputfile = ''
       attribute_name=[]
       if not len(argv) > 1:
            print('No user input')
            sys.exit()
       try:
          opts, args = getopt.getopt(argv,"x:",["ifile="])
       except getopt.GetoptError:
            print('build_tree.py -x <inputfile>')
            sys.exit(2)
       for opt, arg in opts:
          if opt in ("-x", "--ifile"):
             inputfile = arg


       return inputfile


if __name__ == '__main__':
    file = read_file(sys.argv[1:])
    tree=Tree(0.80)
    data,column_name = tree.create_list(file)
    tree.train(data)
    with open('ET-Preeti.pkl','wb') as h:
        pickle.dump(tree,h)

