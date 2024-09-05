#!/usr/bin/env python
# coding: utf-8

# # Lab 1
# James Miles
#Here is a comment im added
 
# In[20]:


import numpy as np
import matplotlib.pyplot as plt


# In[13]:


x = np.linspace(1,5,5)
y = np.arange(1,6)


# In[14]:


x3 = x[:3]


# In[16]:


print('the first three entries of x are', x[0], x[1], x[2])


# In[26]:


w = 10**(-np.linspace(1,10,10))
x = np.linspace(1,10,10)


# In[21]:


plt.plot(x,w)
plt.show()


# In[24]:


s = 3*w
plt.plot(x,w,label ='w')
plt.plot(x,s,label ='s')
plt.legend()
plt.savefig('plot1.png')
plt.show()


# ## 4.2 exercises
# 

# In[47]:


import numpy as np
import numpy.linalg as la
import math


# In[48]:


def dotProduct(x,y,n):
# Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp


# In[49]:


x = np.array([3,4])
y = np.array([-4,3])
dotProduct(x,y,2)


# In[53]:


def matrix_mult(A, B):
    m = len(A)
    n = len(A[0])
    
    n2 = len(B)
    p = len(B[0])
    
    # Ensure the number of columns in A is equal to the number of rows in B
    if n != n2:
        raise ValueError("Number of columns in A must be equal to the number of rows in B.")
    
    # Initialize the result matrix with zeros
    result = [[0] * p for _ in range(m)]
    
    # Perform the matrix-matrix multiplication
    for i in range(m):
        for j in range(p):
            sum = 0
            for k in range(n):
                sum += A[i][k] * B[k][j]
            result[i][j] = sum
    
    return result


# In[54]:


a = [
    [1, 2],
    [3, 4]
]
b = [
    [5, 6],
    [7, 8]
]

result = matrix_mult(a, b)
print(result)


# In[55]:


dot = np.dot(x,y)
print(dot)


# In[57]:


matrix = np.matmul(a, b)
print(matrix)


# In[58]:


# using np is quicker because it is already optimized


# In[ ]:




