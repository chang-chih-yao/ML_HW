先執行create_label_file.py會產生兩個txt檔，分別是test.txt跟train.txt
這兩個txt檔分別對應CSL資料夾內的train data跟test data
txt檔格式如下：

a001.jpg 0
a002.jpg 0
b001.jpg 1
b002.jpg 1
c001.jpg 2
c002.jpg 2
.
.
.
z001.jpg 25
z002.jpg 25

接著再執行hog_svm.py