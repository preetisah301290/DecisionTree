__author__ = 'vijetasah'
import sys
import getopt
import csv
import math
import copy
import random

import numpy as np
import pandas as pd

# import sklearn.metrics
# from sklearn.metrics import confusion_matrix



def read_file(argv):
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

class Node():

        def __init__(self,val=None):
            self.right=None
            self.left=None
            self.data=val
            self.split=None

class Tree():

    def __init__(self,max_level,attr):
        self.thresold_level=max_level
        self.thresold_attr=attr
        self.root = None


    def create_list(self,inputfile):
        print(inputfile)
        rowdata=[]
        inputF = open(inputfile)
        reader=csv.reader(inputF)
        for row in reader:
            rowdata.append(row)
        column_name=rowdata[0]
        column_name.pop()
        return rowdata,column_name

    def calculate_system_entropy(self,data):
        d=dict()
        for row in data:
            currVal = row[len(row)-1]
            if currVal not in d:
                d[currVal]=1
            else:
                d[currVal]+=1
        row_count=len(data)
        entropy_system=0
        for item in d:
           p=d[item]/row_count
           entropy_system+=-1*p*(math.log(p,2))
        entropy_system=round(entropy_system,2)

    def calc_unique_counts(self,data,atrribute_no):
        col_data=[a[atrribute_no] for a in data]
        unique, count=np.unique(col_data,return_counts=True)
        return dict(zip(unique,count))

    def calc(self, data,column_index):
        main_category=self.calc_unique_counts(data,len(data[0])-1)
        atrr_category=self.calc_unique_counts(data,column_index)
        entropy_atrr_total=0.0
        for k,v in atrr_category.items():

            for k1,v1 in main_category.items():
                count=0
                entropy_atrr=0
                for x in data:
                    if x[column_index]==k and x[len(data[0])-1]==k1:
                        count+=1
                p=count/v
                if p!=0 :
                    entropy_atrr+=-1*p*(math.log(p,2))
                else:
                    entropy_atrr=0
                entropy_atrr_total+=(v/len(data))*entropy_atrr

        return(entropy_atrr_total)

    def find_mean(self,attribute_index,data):

        attribute_in_float=[float(x[attribute_index])for x in data]
        attribute_mean=np.mean(attribute_in_float)
        return attribute_mean

    def split_data(self,attribute_index,attribute_mean,data):
        right_split_data=[]
        left_split_data=[]

        for x in data:
            if(float(x[attribute_index])<= attribute_mean):
                left_split_data.append(x)
            else:
                right_split_data.append(x)

        return left_split_data,right_split_data

    def check_thresold(self,data_left,data_right):

        total_left=len(data_left)
        total_right=len(data_right)
        class_occurence_left=self.calc_unique_counts(data_left,len(data_left[0])-1)
        class_occurence_right=self.calc_unique_counts(data_right,len(data_right[0])-1)
        majorityLeft = max(class_occurence_left, key=class_occurence_left.get)
        majorityRight = max(class_occurence_right, key=class_occurence_right.get)

        leftClassOccurences = class_occurence_left[majorityLeft] / total_left
        rightClassOccurences = class_occurence_right[majorityRight] / total_right

        return leftClassOccurences, rightClassOccurences,majorityLeft,majorityRight

    def train(self,trainData):
        column_name=trainData[0]
        trainData.pop(0)
        self.createTree(column_name,trainData)


    def __create_tree(self,column_name,data, total_no_of_cols, attribute_used,node):
        if len(column_name)!=0:
            gainDict={}
            for i in range (len(data[0]) - 1):
                ent = tree.calc(data,i)
                gainDict[i]=ent
            min_col_index=min(gainDict,key=gainDict.get)
            mean=tree.find_mean(min_col_index,data)
            node.data = column_name[min_col_index]
            attribute_used += 1
            left_data,right_data=tree.split_data(min_col_index,mean,data[1:])
            l,r,lclass,rclass=tree.check_thresold(left_data,right_data)

            if total_no_of_cols - 1 == attribute_used:
                global data_remaining,column_remaining
                data_remaining = copy.deepcopy(data)
                column_remaining=copy.deepcopy(column_name)

            node.split = mean

            if l>=self.thresold_attr:
                node.left=Node(lclass)
                for x in data:
                    if(float(x[min_col_index])<=mean):
                        data.remove(x)
            else:

                if len(column_name)>0 :
                    column_name.pop(min_col_index)
                    for rows in data:
                        del rows[min_col_index]
                node.left=Node()
                data_remaining,column_remaining=self.__create_tree(column_name,data,total_no_of_cols, attribute_used,node.left)

            if r>=self.thresold_attr:
                node.right=Node(rclass)
                for x in data:
                    if(float(x[min_col_index])>mean):
                        data.remove(x)
            else:

                if len(column_name)>0:
                    column_name.pop(min_col_index)
                    for rows in data:
                        del rows[min_col_index]
                node.right=Node()
                data_remaining,column_remaining=self.__create_tree(column_name,data,total_no_of_cols, attribute_used,node.right)

        return data_remaining,column_remaining

    def createTree(self,column_name,data):
        gainDict={}
        for i in range (len(data[0]) - 1):
            ent = tree.calc(data,i)
            gainDict[i]=ent
        min_col_index=min(gainDict,key=gainDict.get)
        self.root=Node(column_name[min_col_index])
        data_remaining=None
        attribute_used=0
        data_remaining,column_remaining=self.__create_tree(column_name,data,len(column_name),attribute_used,self.root)


    def tree_traversal(self):
        leaf_nodes=[]
        leaf_nodes=self.__tree_traversal(self.root,leaf_nodes)
        self.complete_Tree(data_remaining,column_remaining,leaf_nodes,self.root)


    def complete_Tree(self,data_remaining,column_remaining,leaf_nodes,node):
        if node is not None:
            if node.data in leaf_nodes:
                if node.left.data is None:
                    splitValue=node.split
                    attribute_index=column_remaining.index(node.data)
                    left_split,right_split=self.split_data(attribute_index,splitValue,data_remaining)
                    for x in data_remaining:
                        if(float(x[attribute_index])<=splitValue):
                            data_remaining.remove(x)

                    class_category=self.calc_unique_counts(left_split,len(left_split[0])-1)
                    majorityLeft = max(class_category, key=class_category.get)
                    node.left.data=majorityLeft

                if node.right.data is None:
                    splitValue=node.split
                    attribute_index=column_remaining.index(node.data)
                    left_split,right_split=self.split_data(attribute_index,splitValue,data_remaining)
                    for x in data_remaining:
                       if(float(x[attribute_index])>splitValue):
                           data_remaining.remove(x)
                    class_category=self.calc_unique_counts(right_split,len(right_split[0])-1)
                    majorityRight = max(class_category, key=class_category.get)
                    node.right.data=majorityRight


            self.complete_Tree(data_remaining,column_remaining,leaf_nodes,node.left)
            self.complete_Tree(data_remaining,column_remaining,leaf_nodes,node.right)

    def __tree_traversal(self,root,leaf_nodes):
        if root is not None:
            self.__tree_traversal(root.left,leaf_nodes)
            self.__tree_traversal(root.right,leaf_nodes)
            if root.left is not None and root.right is not None:
                if(root.left.data is None or root.right.data is None ):
                        leaf_nodes.append(root.data)
        return leaf_nodes

    def predict(self,testData):
        predicted_list=[]
        predicted_value = None
        column_name=None
        for x in range(len(testData)):
            if x == 0:
                column_name = testData[x]
            else:
                predicted_value=self.__predict_Class(testData[x],column_name,self.root, predicted_list)
        return predicted_value

    def __predict_Class(self,row_data,column_name,node, predClass):
        if node is not None:
            if node.data in column_name:
                pos=column_name.index(node.data)
                if float(row_data[pos])<=node.split:
                    predClass = self.__predict_Class(row_data,column_name,node.left, predClass)
                elif float(row_data[pos]) > node.split:
                    predClass = self.__predict_Class(row_data,column_name,node.right, predClass)
            else:
                predClass.append(node.data)
        return predClass

    def Accuracy(self,testData):
        predicted_value=self.predict(testData)
        testData.pop(0)
        actual_value = [data[len(testData[0]) - 1] for data in testData]
        correctly_predicted_count=len([i for i, j in zip(actual_value, predicted_value) if i == j])
        total_value=len(actual_value)
        accuracy=correctly_predicted_count/total_value * 100
        return actual_value,predicted_value
        # print("Accuracy : ",accuracy)

    def ConfusionMatrix(self,TestData):
        actual,predicted=self.Accuracy(self,TestData)
        confusion_matrix(predicted, actual)
        print(confusion_matrix)


if __name__ == "__main__":
    file = read_file(sys.argv[1:])
    tree=Tree(3,0.80)
    data,column_name = tree.create_list(file)
    random.shuffle(data[1:])
    ratio=int(len(data[1:])*0.75)
    training_data=data[:ratio]
    testing_data=data[ratio:]
    testing_data.insert(0, column_name)
    testing_data=copy.deepcopy(testing_data)
    tree.train(data)
    tree.tree_traversal()
    tree.Accuracy(testing_data)