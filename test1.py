import time
import itertools

def a(s):
        
    return 1+int(s)

def main():

    l="abc"
    
    print(l)

    """
    #a.append(7)
    print(a)
    current_pos=0
    c=''
    c='n'
    c='b'
    print(c)
    #sup['straight']=[a[i+current_pos] for i in range(5)]
    #print(a[-2])
    if a[1]:
        print('w')
    else:
        print("l")
    for  i in range(1):
        print(i)"""
    

if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')