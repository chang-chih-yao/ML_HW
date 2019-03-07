from os import walk

mypath_train = "CSL/train/"
mypath_test = "CSL/test/"

file_name_train = []
file_name_test = []

abc_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

for root, dirs, files in walk(mypath_train):
    file_name_train = files

f = open('train.txt', 'w')

for i in file_name_train:
	f.write(i + ' ' + str(abc_list.index(i[0])) + '\n')

f.close()


for root, dirs, files in walk(mypath_test):
    file_name_test = files

f = open('test.txt', 'w')

for i in file_name_test:
	f.write(i + ' ' + str(abc_list.index(i[0])) + '\n')

f.close()

print("create train.txt and test.txt files in this folder.")