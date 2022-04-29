import time
import sys
import numpy as np
import matplotlib.pyplot as plt

def Plotvec1(u, z, v):
    
    ax = plt.axes() # to generate the full window axes
    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)# Add an arrow to the  U Axes with arrow head width 0.05, color red and arrow head length 0.1
    plt.text(*(u + 0.1), 'u')#Adds the text u to the Axes 
    
    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)# Add an arrow to the  v Axes with arrow head width 0.05, color red and arrow head length 0.1
    plt.text(*(v + 0.1), 'v')#Adds the text v to the Axes 
    
    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')#Adds the text z to the Axes 
    plt.ylim(-2, 2)#set the ylim to bottom(-2), top(2)
    plt.xlim(-2, 2)#set the xlim to left(-2), right(2)
    #plt.show()

def main():
    print("Hi!")
    a = np.array([1, -1, 1, -1, 2])
    print(a.mean())
    print(a.std())
    c = np.array([-10, 201, 43, 94, 502])
    z=c.min()+c.max()
    print(z)
    
    u = np.array([1, 0])
    v = np.array([0, 1])
    z = np.add(u, v)
    
    Plotvec1(u,z,v)
    x = np.linspace(0, 2*np.pi, num=100)
    b= np.linspace(5,4,num=6)
    print(b)
    


if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')