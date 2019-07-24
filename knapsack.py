# This file was found online, but I am sorry that I don’t know who the original author is now.
# http://www.geeksforgeeks.org/knapsack-problem/

import numpy as np

def knapsack(v, w, max_weight):
    rows = len(v) + 1
    cols = max_weight + 1

    # adding dummy values as later on we consider these values as indexed from 1 for convinence
    
    v = np.r_[[0], v]
    w = np.r_[[0], w]

    # row : values , #col : weights
    dp_array = [[0 for i in range(cols)] for j in range(rows)]

    # 0th row and 0th column have value 0

    # values
    for i in range(1, rows):
        # weights
        for j in range(1, cols):
            # if this weight exceeds max_weight at that point
            if j - w[i] < 0:
                dp_array[i][j] = dp_array[i - 1][j]

            # max of -> last ele taken | this ele taken + max of previous values possible
            else:
                dp_array[i][j] = max(dp_array[i - 1][j], v[i] + dp_array[i - 1][j - w[i]])

    # return dp_array[rows][cols]  : will have the max value possible for given wieghts

    chosen = []
    i = rows - 1
    j = cols - 1

    # Get the items to be picked
    while i > 0 and j > 0:

        # ith element is added
        if dp_array[i][j] != dp_array[i - 1][j]:
            # add the value
            chosen.append(i-1)
            # decrease the weight possible (j)
            j = j - w[i]
            # go to previous row
            i = i - 1

        else:
            i = i - 1

    return dp_array[rows - 1][cols - 1], chosen

# main
if __name__ == "__main__":
    values = list(map(int, input().split()))
    weights = list(map(int, input().split()))
    max_weight = int(input())

    max_value, chosen = knapsack(values, weights, max_weight)

    print("The max value possible is")
    print(max_value)

    print("The index chosen for these are")
    print(' '.join(str(x) for x in chosen))
