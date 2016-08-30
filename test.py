__author__ = 'preeti_sah'
import sys
import pickle
import getopt

def read_file(argv):
       """
       This will read a file and store the data in the list
       :param argv: file name from command line
       :return: input file
       """
       inputfile = ''
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
    fileName = read_file(sys.argv[1:])
    with open('ET-Preeti.pkl','rb') as h:
        model=pickle.load(h)
    data,column_name = model.create_list(fileName)
    accuracy = model.Accuracy(data)
    print(accuracy)
    model.ConfusionMatrix(data)