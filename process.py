"""
@credit to https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
@author Nils Tijtgat
"""

"""
April 8th, 2018
modified by Dongjun Seung
dongjun2@illinois.edu
"""


import glob, os

os.chdir("training")

# Current directory
current_dir = (os.path.dirname(os.path.abspath(__file__)))

# Directory where the data will reside, relative to 'darknet.exe'
path_data = 'data/obj/'

# Percentage of images to be used for the test set
percentage_test = 100;

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')  
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt
counter = 1  
index_test = round(100 / percentage_test)  
print (index_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    print (title, ext)

    if counter == index_test:
        counter = 1
        file_test.write(path_data + title + '.jpg' + "\n")
    else:
        file_train.write(path_data + title + '.jpg' + "\n")
        counter = counter + 1

os.chdir("..")