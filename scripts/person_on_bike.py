import json
import numpy as np

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

def main():
    filename = "./../detectionfile.json"
    bicycles = []
    people = []
    try:
        with open(filename) as json_data:
            d = json.load(json_data)
            for obj in d:
                if obj['type'] == 'person':
                    people = obj['detections']
                elif obj['type'] == 'bicycle':
                    bicycles = obj['detections']

            if len(people) == 0:
                print "No person is detected in the image"
            if len(bicycles) == 0:
                print "No bicycle is detected in the image"

    except ValueError:
        print "There was an issue with opening detectionfile.json"

    bikes_with_people = closest_person_to_bike(bicycles,people)
    people_on_bike = []
    for bike, person in bikes_with_people:
        if check_person_on_bike(bike, person):
            people_on_bike.append({'bike': bike, 'person': person, 'IOU': intersection_over_union(bike,person)})

    print "Number of (bike and person) pairs detected: ", repr(len(people_on_bike))

    dump = json.dumps(people_on_bike)
    with open("cyclist.json","w") as output:
        output.write(dump)

if __name__ == "__main__":
    main()
