import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict
def analyzeResults(filePath):
    for root, dir, file in os.walk(filePath):
        scores = defaultdict(lambda: defaultdict(list))
        for f in file:  
            dashes = f.split('-')
            days, nums = dashes[1], dashes[2]
            if f[-3:] == "csv":
                df = pd.read_csv(os.path.join(filePath, f))
                for i, row in df.iterrows():
                    scores[row["name"][:2]][(days,nums)].append(float(row["marriage_score"]))
        return scores
def printTeam(scores):
    for key, value in scores.items():
        print(key)
        print(value)
        dayList = defaultdict(list)
        suitorList = defaultdict(list)
        for k2, v2 in value.items():
            day, numSuitors = k2
            dayList[day].extend(v2)
            suitorList[numSuitors].extend(v2)
        print('day average analysis')
        for k2, v2 in dayList.items():
            print(k2, np.mean(v2))
            print()
        print('num suitors average analysis')
        for k2, v2 in suitorList.items():
            print(k2, np.mean(v2))
            print()
            

if __name__ == '__main__':
    path = './results'
    if len(sys.argv) > 1:
        print('using', sys.argv[1])
        path = sys.argv[1]
    res = analyzeResults(path)
    printTeam(res)
