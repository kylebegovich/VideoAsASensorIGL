import json
import numpy as np
import os
from sklearn.svm import SVC


# parameters
#   box1: a dictionary with x, y, h, w, confidence, and name
#   box2: a dictionary with x, y, h, w, confidence, and name
# returns a float. The intersection over union of the 2 boxes
def intersection_over_union(box1, box2):
    x1 = box1['x']
    y1 = box1['y']
    w1 = box1['w']
    h1 = box1['h']

    x2 = box2['x']
    y2 = box2['y']
    w2 = box1['w']
    h2 = box1['h']

    box1_left_side = x1 - w1/2.
    box1_right_side = x1 + w1/2.
    box2_left_side = x2 - w2/2.
    box2_right_side = x2 + w2/2.

    intersection_box_width = min(box1_right_side, box2_right_side) - max(box1_left_side, box2_left_side)
    if intersection_box_width < 0:
        intersection_box_width = 0

    box1_top = y1 - h1/2.
    box1_bottom = y1 + h1/2.
    box2_top = y2 - h2/2.
    box2_bottom = y2 + y2/2.

    intersection_box_height = min(box1_bottom, box2_bottom) - max(box1_top, box2_top)
    if intersection_box_height < 0:
        intersection_box_height = 0

    intersection_area = intersection_box_height * intersection_box_width
    union = (w1 * h1) + (w2 * h2) - intersection_area
    return float(intersection_area) / float(union)

# parameters
#   box1: a dictionary with x, y, h, w, confidence, and name
#   box2: a dictionary with x, y, h, w, confidence, and name
# returns a float. The euclidean distance between the centroids of 2 boxes
def distance(box1, box2):
    x1 = box1['x']
    y1 = box1['y']
    x2 = box2['x']
    y2 = box2['y']
    return np.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)

# parameters list of people, list of bicycles
def compare_height(bike, person):
    y1 = bike['y']
    h1 = bike['h']
    y2 = person['y']
    h2 = person['h']
    return (y1 + h1 / 2.0) > (y2 - h2 / 2.0)


# parameters json object
# returns list of tuples
#   each tuple is a (bike,person) pair
#   matches each bike to the nearest person
def closest_person_to_bike(bicycles, people):
    answers = []
    for bike in bicycles:
        nearest_person = people[0]
        for person in people:
            if distance(bike,person) < distance(bike,nearest_person):
                nearest_person = person
        answers.append((bike, nearest_person))
    return answers

# parameters
#   bike - a dictionary with the bounding box coordinates and stats
#   person - a dictionary with the bounding box coordinates and stats
# returns a boolean. true if the person is believed to be on the bike, false otherwise
def check_person_on_bike(bike, person):
    if intersection_over_union(bike,person) > .3 and compare_height(bike,person):
        return True
    return False
    
def get_features_and_labels(dir_name):
    features = []
    labels = []

    
    for filename in os.listdir(dir_name):
        if filename.endswith("_1.json"):
            
            with open(dir_name + "/" + filename) as json_data:
                j = json.load(json_data)
                for obj in j:
                    if obj['type'] == 'person':
                        people = obj['detections']
                    elif obj['type'] == 'bicycle':
                        bicycles = obj['detections']
                for person in people:
                    for bike in bicycles:
                        if (check_person_on_bike(bike, person)):
                            labels.append(1)
                        else:
                            labels.append(0)
                            
                        feature_vec = []
                        for person_attribute in sorted(person.keys()):
                            if person_attribute == 'name':
                                continue
                                
                            feature_vec.append(person[person_attribute])
                            
                        for bike_attribute in sorted(bike.keys()):
                            if bike_attribute == 'name':
                                continue
                            feature_vec.append(bike[bike_attribute])

                    
                        #Can add other attributes such as IOU to feature_vec here
                        feature_vec.append(intersection_over_union(bike, person))
                        
                        features.append(feature_vec)
                        
    return features, labels
    

def main():
    dir_name = "../training_json"
    features, labels = get_features_and_labels(dir_name)
    #Can do test train split here for testing
    
    clf = SVC(kernel='linear')
    clf.fit(features, labels)
    
    #These are the coefficients that would be used in c (note that the order of the features is very important and should be the same as the order in which they are appended to feature_vec)
    print(clf.coef_)
    
    #This is how to predict using the SVM
    print(clf.predict(features))


if __name__ == "__main__":
    main()
