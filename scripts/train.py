import numpy as np
import json
import person_on_bike as pob
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import os
import re

def main():
  features = []
  labels = []
  for filename in os.listdir("./../training_json"):
    isPos = 0.
    try:
      re.search('.*_pos\.json',filename).group(0)
      print("pos found:")
      print(filename)
      isPos = 1.
    except AttributeError:
      print('not a pos. checking if neg')
    try:
      re.search('.*_neg\.json',filename).group(0)
      print("neg found:")
      print(filename)
      isPos = 0.
    except AttributeError:
      print('not a neg or pos. moving on...')
    try:
      with open("./../training_json/" + filename) as data:
        obj = json.load(data)["detections"]
      for i in range(len(obj)):
        row = np.zeros(11)
        row[0] = obj[i][0]["x"]
        row[1] = obj[i][0]["y"]
        row[2] = obj[i][0]["confidence"]
        row[3] = obj[i][0]["w"]
        row[4] = obj[i][0]["h"]
        row[5] = obj[i][1]["x"]
        row[6] = obj[i][1]["y"]
        row[7] = obj[i][1]["confidence"]
        row[8] = obj[i][1]["w"]
        row[9] = obj[i][1]["h"]
        row[10] = pob.intersection_over_union(obj[i][0], obj[i][1])
        features.append(row)

      for i in range(len(obj)):
        labels.append(isPos)

    except ValueError:
      print("There was an issue opening a file")
      continue
  #train machine here
  features = np.array(features)
  labels = np.array(labels)
  clf = GaussianNB()
  clf.fit(features, labels)

  #print(clf.coef_)

  print(clf.predict(features))

if __name__ == "__main__":
  main()
