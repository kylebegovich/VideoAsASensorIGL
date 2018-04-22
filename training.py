# Dongjun Seung (dongjun2)
# 3/07/2018
# dongjun2@illinois.edu

import os
import json
import shutil

# The main function calls the generic detect function of darknet using the default yolo
# cfg and weight files to generate all detected people and bicycles in json.
# This json file is then used in the person_on_bike.py script to create a json file
# that contains all 'person and bike' pairs.

def main():
    current_dir = os.getcwd()
    training = os.path.join(current_dir, "training")

    # if the "training" folder does not exist, then let the user know to put images in there
    if not os.path.exists(training):
        os.mkdir("training")
        print ("-------------------------------------------")
        print ("--Put Images in the training folder--")
        print ("-------------------------------------------")
        return

    # create a folder named "training_json," where all the json files will be saved
    training_json = os.path.join(current_dir, "training_json")
    if not os.path.exists(training_json):
        os.mkdir("training_json")

    # create a folder named "training_predictions," where all the predctions.png files will be saved
    training_predictions = os.path.join(current_dir, "training_predictions")
    if not os.path.exists(training_predictions):
        os.mkdir("training_predictions")

    try:
        for image in os.listdir("./training"):
        	image_name = image[:len(image) - 5] if image.endswith("JPEG") else image[:len(image) - 4]
        	print ("image_name: ", image_name)
        	if image_name == ".DS_S":
        		continue

        	os.system("./darknet detect cfg/yolo.cfg yolo.weights training/{}".format(image))

        	shutil.copyfile("detectionfile_negatives.json", os.path.join(training_json, "{}_neg.json".format(image_name)))
                shutil.copyfile("detectionfile_positives.json", os.path.join(training_json, "{}_pos.json".format(image_name)))
        	shutil.copyfile("predictions.png", os.path.join(training_predictions, "{}.jpg".format(image_name)))

        	os.chdir("scripts")
                
        	os.chdir('..')

        # send all json files that have detected 'bike and person' pairs to the "training_json" folder
    	transfer_from = os.path.join(current_dir, "scripts")
    	for json_file in os.listdir("./scripts"):
       		if json_file.endswith(".json"):
       			shutil.move(os.path.join(transfer_from, json_file), training_json)
    
    except shutil.Error:
        print ("-----------------------------------------------")
        print ("This training may have already done in the past")
        print ("   Duplicate files exist, check your folders   ")
        print ("-----------------------------------------------")

    # running darknet and transferring files are done
    print ("---------------------------------------------------------------")
    print ("    Running darknet on images and transferring files are done   ")
    print ("Label images with BBox-Label-Tool, then run 'label_transform.py'")
    print ("---------------------------------------------------------------")


if __name__ == "__main__":
    main()

