import numpy as np
import dill as pickle


def hankelize(A):
    import numpy as np
    x = 0
    avrlst = []
# 1. find a list contain the diagonals of the matrix
    matrix = np.array(matrix, dtype=float)
    matrix = np.flipud(matrix)
    a = matrix.shape[0]
    mylist = [np.diag(matrix, k=i).tolist() for i in range(-a+1,a)]

# 2. find the avearage for each item in the list
    for i in mylist:
        x= sum(i) / len(i)
        avrlst.append(x)

# 3. assign the value of each avaerage to the diagonal memebrs
    ROW , COL = np.shape(matrix)
# Note:  It’s a keen observation that the sum of [i+j]
# that is the indexes of the array remains the same throughout the diagonal.
# So we will exploit this property of the matrix to make our code short and simple.
    for i in range(0, ROW):
        for j in range(0, COL):
            matrix[i][j] = avrlst[i+j]
    return matrix
###########################
# Answer:
# Well, here, the point is not to order the cells when you visit them but to get a list of all the diagonals in an existing array.
# To do that, you can use the fact that for each cell in a diagonal, the sum of the column index and the row index is the same.
##########################


# ///////////////////////////////////////////////

# Create A Derivative Calculator in Python
# f^(a) = lim(h-->0) f(a+h) - f(a) / h
# Where f(x) is the function, a is the point to find the slope,
# f’(a) is the slope at the point. Essentially, this limit finds the rate of change between two points
# as those points become increasingly close to each other and converge
# to a point with no distance between each other (h=0)
def gradient(f, x):
    x1 = x[0]
    x2 = x[1]
    fx1 =0
    fx2 =0
    h = 0.00000000001
    top1 = f(x1 + h, x2) - f(x1,x2)
    bottom = h
    slope1 = top1 / bottom
    top2 = f(x1, x2 +h) - f(x1,x2)
    slope2 = top2 / bottom
    return (slope1, slope2)



def minimize(f):
    # run the gradient descent
    import random
    learningRate = 0.001
    x1 = random.randint(0, 9)
    x2 = random.randint(0, 9)
    min1 = 1000000000000
    min2 = 1000000000000
    for i in range(0,1000):
        for j in range(0,1000):
            x = (i,j)
            divx1 , divx2  = gradient(f, x)
            # Updating the parameters
            x1 = x1 - learningRate * divx1  # Update
            x2 = x2 - learningRate * divx2  # Update
            min1 = min(x1,min1)
            min2 = min(x2, min2)

    return (min1, min2)

#######################
# 3. What are the coordinates of x that minimize the function?
#Answer: Sub min1 and min2 in the function to find the coordinate.

# 4. This particular method for minimizing a function has several issues which make it less
# than ideal in practice. Explain what some of these issues might be and possible solutions
# to those problems.

#Answer: If the execution is not done properly while using gradient descent,
# it may lead to problems like vanishing gradient or exploding gradient problems.
# These problems occur when the gradient is too small or too large.
# And because of this problem, the algorithms do not converge.

# 5. How could you optimize this algorithm for efficiency? What would be a more efficient /
# more robust algorithm?

# Answer: The general idea is to initialize the parameters to random values,
# and then take small steps in the direction of the “slope” at each iteration.
# 6. If the function ‘f’ was noisy, how would you adapt your algorithm?

# Answer:  As long as your learning rate is small and you use a nice annealing schedule,
# the more noise you have, the better gradient descent looks when compared to other alternatives.

if __name__ == '__main__':

    #
    # Problem 1
    #

    A = np.array([[2,  6,  -1, 9, -3,  6],
                  [7,  4,   2, 4,  2, 10],
                  [3, -2, -10, 1,  4,  2]])
    print(hankelize(A))

    #
    # Problem 2
    #

    with open('f.pkl', 'rb') as file:
        f = pickle.load(file)

    print(f(x=(0, 0)))
    print(minimize(f))
