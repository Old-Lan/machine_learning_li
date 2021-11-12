import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x = np.array([[0,0],[1,1],[1,0],[0,1]])
    plt.scatter(x[:2,0],x[:2,1], marker='o', color='red', label='Pos')
    plt.scatter(x[2:,0],x[2:,1], marker='x', color='blue', label='Neg')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
