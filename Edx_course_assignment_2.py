import time

from random import randint as rnd

memReg = 'members.txt'
exReg = 'inactive.txt'
fee =('yes','no')

def genFiles(current,old):
    
    with open(current,'w+') as writefile: 
        writefile.write('Membership No  Date Joined  Active  \n')
        data = "{:^13}  {:<11}  {:<6}\n"

        for rowno in range(20):
            date = str(rnd(2015,2020))+ '-' + str(rnd(1,12))+'-'+str(rnd(1,25))
            writefile.write(data.format(rnd(10000,99999),date,fee[rnd(0,1)]))


    with open(old,'w+') as writefile: 
        writefile.write('Membership No  Date Joined  Active  \n')
        data = "{:^13}  {:<11}  {:<6}\n"
        for rowno in range(3):
            date = str(rnd(2015,2020))+ '-' + str(rnd(1,12))+'-'+str(rnd(1,25))
            writefile.write(data.format(rnd(10000,99999),date,fee[1]))


def cleanFiles(currentMem, exMem):
    
    with open(currentMem, 'r+') as file1:

        with open(exMem, 'a+') as file2:
    
            Lines = file1.readlines()
            
            innactive_members = []
            
        
            i=1
            while i<len(Lines): 
                #print(Lines[i])
                if "no" in Lines[i]:
                    innactive_members.append(Lines[i])    
                i+=1
        
            i=1
            file1.truncate(0)
            while i<len(Lines):
                if Lines[i] in innactive_members:
                    file2.write(Lines[i])
                else: 
                    file1.write(Lines[i])
                i+=1
        





def main():
    genFiles(memReg,exReg)
    
    with open(memReg,'r') as readFile:
        print("Active Members: \n\n")
        print(readFile.read())

    with open(exReg,'r') as readFile:
        print("Inactive Members: \n\n")
        print(readFile.read())
        
    cleanFiles(memReg,exReg)
    
    headers = "Membership No  Date Joined  Active  \n"
    with open(memReg,'r') as readFile:
        print("Active Members: \n\n")
        print(readFile.read())

    with open(exReg,'r') as readFile:
        print("Inactive Members: \n\n")
        print(readFile.read())
    

    
    

if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')