import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # x = np.array([[0,0],[1,1],[1,0],[0,1]])
    # plt.scatter(x[:2,0],x[:2,1], marker='o', color='red', label='Pos')
    # plt.scatter(x[2:,0],x[2:,1], marker='x', color='blue', label='Neg')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.show()
    mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
              {'a': 100, 'b': 200, 'c': 300, 'd': 400},
              {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000}]
    df = pd.DataFrame(mydict)
    print(df)
    # print(df.iloc[0])
    # print(type(df.iloc[0]))
    print(df.iloc[[0,2],[1,3]])
    print(df.iloc[0:2,1:3])
    print(df.iloc[lambda x:x.index%2==0])
