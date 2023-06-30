#Made by Kiptyk Kirill

import time

very_low=[0,0,0.15,0.25]
low=[0.15,0.25,0.35,0.45]
medium=[0.35,0.45,0.55,0.65]
high=[0.55,0.65,0.75,0.85]
very_high=[0.75,0.85,1,1]

class Tree(dict):
    """A tree implementation using python's autovivification feature."""
    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    #cast a (nested) dict to a (nested) Tree class
    def __init__(self, data={}):
        for k, data in data.items():
            if isinstance(data, dict):
                self[k] = type(self)(data)
            else:
                self[k] = data

def choose_weights(number_of_nodes):
    i=1
    relationships=[]

    while i<number_of_nodes:
        a=str(input("Enter the relationship between "+str(i)+" and "+str(i+1)+" (> or =): "))
        if not(a=='>' or a=='='):
            print("Error!!!")
            break
        relationships.append(a)
        i+=1

    weights=calculate_weights(relationships)
    #print(weights)
    return weights  

def calculate_weights(relationship):
    weights=[]

    if len(relationship)==1:
        if relationship[0]=='>':
            weights.append(2/3)
            weights.append(1/3)
            return weights
        elif relationship[0]=='=':
            weights.append(1/2)
            weights.append(1/2)
            return weights

    elif len(relationship)==2:
        if relationship[0]=='=':
            if relationship[1]=='=':
                weights.append(1/3)
                weights.append(1/3)
                weights.append(1/3)
                return weights
            elif relationship[1]=='>':
                weights.append(2/5)
                weights.append(2/5)
                weights.append(1/5)
                return weights

        elif relationship[0]=='>':
            if relationship[1]=='=':
                weights.append(2/4)
                weights.append(1/4)
                weights.append(1/4)
                return weights
            elif relationship[1]=='>':
                weights.append(3/6)
                weights.append(2/6)
                weights.append(1/6)
                return weights

    elif len(relationship)==3:
        if relationship[0]=='=':
            if relationship[1]=='=':
                if relationship[2]=='=':
                    weights.append(1/4)
                    weights.append(1/4)
                    weights.append(1/4)
                    weights.append(1/4)
                    return weights
                elif relationship[2]=='>':
                    weights.append(2/7)
                    weights.append(2/7)
                    weights.append(2/7)
                    weights.append(1/7)
                    return weights

            elif relationship[1]=='>':
                if relationship[2]=='=':
                    weights.append(2/6)
                    weights.append(2/6)
                    weights.append(1/6)
                    weights.append(1/6)
                    return weights
                elif relationship[2]=='>':
                    weights.append(3/9)
                    weights.append(3/9)
                    weights.append(2/9)
                    weights.append(1/9)
                    return weights

        elif relationship[0]=='>':
            if relationship[1]=='=':
                if relationship[2]=='=':
                    weights.append(2/5)
                    weights.append(1/5)
                    weights.append(1/5)
                    weights.append(1/5)
                    return weights
                elif relationship[2]=='>':
                    weights.append(3/8)
                    weights.append(2/8)
                    weights.append(2/8)
                    weights.append(1/8)
                    return weights

            elif relationship[1]=='>':
                if relationship[2]=='=':
                    weights.append(3/7)
                    weights.append(2/7)
                    weights.append(1/7)
                    weights.append(1/7)
                    return weights
                elif relationship[2]=='>':
                    weights.append(4/10)
                    weights.append(3/10)
                    weights.append(2/10)
                    weights.append(1/10)
                    return weights

def choose_tr_number():
    
    a=int(input("Enter 1-5 to chose between different values (Very Low, Low, Medium, High, Very High): "))
    if a==1:
        f=very_low
        print(f)
        return f
    elif a==2:
        f=low
        print(f)
        return f
    elif a==3:
        f=medium
        print(f)
        return f
    elif a==4:
        f=high
        print(f)
        return f
    elif a==5:
        f=very_high
        print(f)
        return f
    else:
        print("Error!!!")

def input_number():
    tr_number=[]
    i=0
    while(i<4):
        a=float(input("Enter number "+str(i+1)+" element of trapezoidal number: "))
        tr_number.insert(i,a)
        i=i+1
    return tr_number

def func(weight_list, node_list):
    
    node_list = [element for element in node_list if element != {}] 
    #print(node_list)
    
    if len(weight_list)==2:
        first_list=[element * weight_list[0] for element in node_list[0]]
        second_list=[element * weight_list[1] for element in node_list[1]]
        sum_list=[a+b for a,b in zip(first_list,second_list)]
        return sum_list

    elif len(weight_list)==3:
        first_list=[element * weight_list[0] for element in node_list[0]]
        second_list=[element * weight_list[1] for element in node_list[1]]
        third_list=[element * weight_list[2] for element in node_list[2]]
        sum_list=[a+b+c for a,b,c in zip(first_list, second_list, third_list)]
        return sum_list
        
    elif len(weight_list)==4:
        first_list=[element * weight_list[0] for element in node_list[0]]
        second_list=[element * weight_list[1] for element in node_list[1]]
        third_list=[element * weight_list[2] for element in node_list[2]]
        fourth_list=[element * weight_list[3] for element in node_list[3]]
        sum_list=[a+b+c+d for a,b,c,d in zip(first_list, second_list, third_list, fourth_list)]
        return sum_list   

def similarity(f0):
    def calculation(tr_number):
        sub_list=[a-b for a,b in zip(f0,tr_number)]
        abs_list=[abs(a) for a in sub_list]
        return 1-max(abs_list)
    a1=calculation(very_low)
    a2=calculation(low)
    a3=calculation(medium)
    a4=calculation(high)
    a5=calculation(very_high)
    numbers={"Very Low": a1, "Low": a2, "Medium": a3, "High": a4, "Very High": a5}
    max_v = max(a1,a2,a3,a4,a5)
    rate = max(numbers, key=numbers.get)
    return max_v,rate

def default(rate):
    if rate=="Very Low":
        return "Very High"
    elif rate=="Low":
        return "High"
    elif rate=="Medium":
        return "Medium"
    elif rate=="High":
        return "Low"
    elif rate=="Very High":
        return "Very Low"

def main():

    choose_option=int(input("Enter 1 if you want to implement existing model or 2 if you want to build it by yourself: "))
    tree = Tree()
    weights={}

    if choose_option==1:

        choose_model=int(input("Choose exsiting model: \n 1----Ukraine(DRM-1) \n 2----Ukraine(DRM-2) \n 3----Poland(DRM-1)\n 4----Poland(DRM-2) \n 5----Greece \n 6----Germany \n"))    
        try:

            if choose_model==1:

                tree['F0']['F1']
                tree['F0']['F2']
                tree['F0']['F3']=very_low
                tree['F0']['F1']['F1.1']
                tree['F0']['F1']['F1.2']=medium
                tree['F0']['F2']['F2.1']=very_low
                tree['F0']['F2']['F2.2']=medium
                tree['F0']['F1']['F1.1']['F1.1.1']=low
                tree['F0']['F1']['F1.1']['F1.1.2']=low

                weights["F1"]=3/6
                weights["F2"]=2/6
                weights["F3"]=1/6
                weights["F1.1"]=2/3
                weights["F1.2"]=1/3
                weights["F2.1"]=2/3
                weights["F2.2"]=1/3
                weights["F1.1.1"]=1/2
                weights["F1.1.2"]=1/2
                
            elif choose_model==3:
                
                tree['F0']['F1']
                tree['F0']['F2']
                tree['F0']['F3']=medium
                tree['F0']['F1']['F1.1']
                tree['F0']['F1']['F1.2']=medium
                tree['F0']['F2']['F2.1']=medium
                tree['F0']['F2']['F2.2']=very_high
                tree['F0']['F1']['F1.1']['F1.1.1']=very_high
                tree['F0']['F1']['F1.1']['F1.1.2']=high

                weights["F1"]=3/6
                weights["F2"]=2/6
                weights["F3"]=1/6
                weights["F1.1"]=2/3
                weights["F1.2"]=1/3
                weights["F2.1"]=2/3
                weights["F2.2"]=1/3
                weights["F1.1.1"]=1/2
                weights["F1.1.2"]=1/2
            
            elif choose_model==4:

                tree['F0']['F1']
                tree['F0']['F2']
                tree['F0']['F3']
                tree['F0']['F4']=medium
                tree['F0']['F1']['F1.1']
                tree['F0']['F1']['F1.2']=medium
                tree['F0']['F2']['F2.1']=medium
                tree['F0']['F2']['F2.2']=very_high
                tree['F0']['F1']['F1.1']['F1.1.1']=very_high
                tree['F0']['F1']['F1.1']['F1.1.2']=high
                tree['F0']['F3']['F3.1']=high
                tree['F0']['F3']['F3.2']
                tree['F0']['F3']['F3.3']=very_high
                tree['F0']['F3']['F3.4']=medium
                tree['F0']['F3']['F3.2']['F3.2.1']=very_high
                tree['F0']['F3']['F3.2']['F3.2.2']=high

                weights["F1"]=3/7
                weights["F2"]=2/7
                weights["F3"]=1/7
                weights["F4"]=1/7
                weights["F1.1"]=2/3
                weights["F1.2"]=1/3
                weights["F2.1"]=2/3
                weights["F2.2"]=1/3
                weights["F1.1.1"]=1/2
                weights["F1.1.2"]=1/2
                weights["F3.1"]=2/7
                weights["F3.2"]=2/7
                weights["F3.3"]=2/7
                weights["F3.4"]=1/7
                weights["F3.2.1"]=2/3
                weights["F3.2.2"]=1/3
                
            elif choose_model==5:

                tree['F0']['F1']
                tree['F0']['F2']
                tree['F0']['F3']
                tree['F0']['F4']=medium
                tree['F0']['F1']['F1.1']
                tree['F0']['F1']['F1.2']=high
                tree['F0']['F2']['F2.1']=low
                tree['F0']['F2']['F2.2']=high
                tree['F0']['F1']['F1.1']['F1.1.1']=low
                tree['F0']['F1']['F1.1']['F1.1.2']=high
                tree['F0']['F3']['F3.1']=medium
                tree['F0']['F3']['F3.2']
                tree['F0']['F3']['F3.3']=very_high
                tree['F0']['F3']['F3.4']=medium
                tree['F0']['F3']['F3.2']['F3.2.1']=medium
                tree['F0']['F3']['F3.2']['F3.2.2']=high

                weights["F1"]=3/7
                weights["F2"]=2/7
                weights["F3"]=1/7
                weights["F4"]=1/7
                weights["F1.1"]=2/3
                weights["F1.2"]=1/3
                weights["F2.1"]=2/3
                weights["F2.2"]=1/3
                weights["F1.1.1"]=1/2
                weights["F1.1.2"]=1/2
                weights["F3.1"]=2/7
                weights["F3.2"]=2/7
                weights["F3.3"]=2/7
                weights["F3.4"]=1/7
                weights["F3.2.1"]=2/3
                weights["F3.2.2"]=1/3

            elif choose_model==6:

                tree['F0']['F1']
                tree['F0']['F2']
                tree['F0']['F3']
                tree['F0']['F4']=very_high
                tree['F0']['F1']['F1.1']
                tree['F0']['F1']['F1.2']=very_high
                tree['F0']['F2']['F2.1']=medium
                tree['F0']['F2']['F2.2']=very_high
                tree['F0']['F1']['F1.1']['F1.1.1']=very_high
                tree['F0']['F1']['F1.1']['F1.1.2']=very_high
                tree['F0']['F3']['F3.1']=medium
                tree['F0']['F3']['F3.2']
                tree['F0']['F3']['F3.3']=very_high
                tree['F0']['F3']['F3.4']=very_high
                tree['F0']['F3']['F3.2']['F3.2.1']=very_high
                tree['F0']['F3']['F3.2']['F3.2.2']=very_high

                weights["F1"]=3/7
                weights["F2"]=2/7
                weights["F3"]=1/7
                weights["F4"]=1/7
                weights["F1.1"]=2/3
                weights["F1.2"]=1/3
                weights["F2.1"]=2/3
                weights["F2.2"]=1/3
                weights["F1.1.1"]=1/2
                weights["F1.1.2"]=1/2
                weights["F3.1"]=2/7
                weights["F3.2"]=2/7
                weights["F3.3"]=2/7
                weights["F3.4"]=1/7
                weights["F3.2.1"]=2/3
                weights["F3.2.2"]=1/3
            
            elif choose_model==2:

                tree['F0']['F1']
                tree['F0']['F2']
                tree['F0']['F3']
                tree['F0']['F4']=very_low
                tree['F0']['F1']['F1.1']
                tree['F0']['F1']['F1.2']=medium
                tree['F0']['F2']['F2.1']=very_low
                tree['F0']['F2']['F2.2']=medium
                tree['F0']['F1']['F1.1']['F1.1.1']=low
                tree['F0']['F1']['F1.1']['F1.1.2']=low
                tree['F0']['F3']['F3.1']=high
                tree['F0']['F3']['F3.2']
                tree['F0']['F3']['F3.3']=low
                tree['F0']['F3']['F3.4']=low
                tree['F0']['F3']['F3.2']['F3.2.1']=very_low
                tree['F0']['F3']['F3.2']['F3.2.2']=medium

                weights["F1"]=3/7
                weights["F2"]=2/7
                weights["F3"]=1/7
                weights["F4"]=1/7
                weights["F1.1"]=2/3
                weights["F1.2"]=1/3
                weights["F2.1"]=2/3
                weights["F2.2"]=1/3
                weights["F1.1.1"]=1/2
                weights["F1.1.2"]=1/2
                weights["F3.1"]=2/7
                weights["F3.2"]=2/7
                weights["F3.3"]=2/7
                weights["F3.4"]=1/7
                weights["F3.2.1"]=2/3
                weights["F3.2.2"]=1/3

        except:
            pass

    elif choose_option==2:
        
        w=[]

        second_layer_nodes=int(input("Enter number of nods in the second layer (2-4): "))
        w=choose_weights(second_layer_nodes)

        i=1
        while i<=second_layer_nodes:
            weights["F"+str(i)]=w[i-1]
            i+=1

        i1=1

        while i1<=second_layer_nodes:

            tree['F0']['F'+str(i1)]
            a=input("Do you want to add child(s) to "+'F'+str(i1)+" node (0 -> No; 1 -> Yes): ")

            if a=='0':
                print("Choose "+'F'+str(i1)+" node value:")

                b=int(input("Enter 1 to chose between preset trapezoidal numbers or 2 to input it by yourself: "))
                if b==1:
                    tree['F0']['F'+str(i1)]=choose_tr_number()
                elif b==2:
                    tree['F0']['F'+str(i1)]=input_number()

                i1+=1
            
            if a=='1':

                third_layer_nodes=int(input("Enter number of child(s) of the "+'F'+str(i1)+" node (2-4): "))
                w=choose_weights(third_layer_nodes)

                i=1
                while i<=third_layer_nodes:
                    weights["F"+str(i1)+"."+str(i)]=w[i-1]
                    i+=1

                i2=1
                
                while i2<=third_layer_nodes:

                    tree['F0']['F'+str(i1)]['F'+str(i1)+"."+str(i2)]
                    b=input("Do you want to add child(s) to "+'F'+str(i1)+"."+str(i2)+" node (0 -> No; 1 -> Yes): ")

                    if b=='0':
                        print("Choose "+'F'+str(i1)+"."+str(i2)+" node value:")

                        b=int(input("Enter 1 to chose between preset trapezoidal numbers or 2 to input it by yourself: "))
                        if b==1:
                            tree['F0']['F'+str(i1)]['F'+str(i1)+"."+str(i2)]=choose_tr_number()
                        elif b==2:
                            tree['F0']['F'+str(i1)]['F'+str(i1)+"."+str(i2)]=input_number()

                        i2+=1

                    if b=='1':
                        fourth_layer_nodes=int(input("Enter number of child(s) of the "+'F'+str(i1)+"."+str(i2)+" node (2-4): "))
                        w=choose_weights(fourth_layer_nodes)

                        i=1
                        while i<=fourth_layer_nodes:
                            weights["F"+str(i1)+"."+str(i2)+"."+str(i)]=w[i-1]
                            i+=1

                        i3=1
                        while i3<=fourth_layer_nodes:
                            print("Choose "+'F'+str(i1)+"."+str(i2)+"."+str(i3)+" node value:")

                            b=int(input("Enter 1 to chose between preset trapezoidal numbers or 2 to input it by yourself: "))
                            if b==1:
                                tree['F0']['F'+str(i1)]['F'+str(i1)+"."+str(i2)]['F'+str(i1)+"."+str(i2)+"."+str(i3)]=choose_tr_number()
                            if b==2:
                                tree['F0']['F'+str(i1)]['F'+str(i1)+"."+str(i2)]['F'+str(i1)+"."+str(i2)+"."+str(i3)]=input_number()

                            i3+=1
                        
                        i2+=1

                i1+=1

    print("-------------------------------------------------------------------------------")
    print("Given Tree structure")
    print(tree)
    print("Weights of every node in a tree")
    print(weights)
    print("-------------------------------------------------------------------------------")

    i1=1
    while i1<=4:
        i2=1
        while i2<=4:

            try:
                if tree['F0']['F'+str(i1)]['F'+str(i1)+'.'+str(i2)]['F'+str(i1)+'.'+str(i2)+'.1']:

                    counter=1

                    node_value=[]
                    weights_value=[]

                    while counter<=4:
                        try:
                            node_value.append(tree['F0']['F'+str(i1)]['F'+str(i1)+'.'+str(i2)]['F'+str(i1)+'.'+str(i2)+'.'+str(counter)])
                            a='F'+str(i1)+'.'+str(i2)+'.'+str(counter)
                            if a in weights.keys():
                                weights_value.append(weights[a])

                        except: 
                            print("Weird")

                        counter+=1

                    k=func(weights_value,node_value)
                    if k!=[]:
                        tree['F0']['F'+str(i1)]['F'+str(i1)+'.'+str(i2)]=k
                        print("The value of a factor "+'F'+str(i1)+'.'+str(i2)+" is: ", k)
                        max_v,rate=similarity(k)
                        print("The similarity rate V of a factor "+'F'+str(i1)+'.'+str(i2)+" and linguistic estimation \""+rate+"\" is: "+str(max_v))
                        print("-------------------------------------------------------------------------------")
                    
            except:

                pass

            i2+=1
        i1+=1

    i1=1
    while i1<=4:
        try:
            if tree['F0']['F'+str(i1)]['F'+str(i1)+'.1']:
                counter=1
                node_value=[]
                weights_value=[]
                
                while counter<=4:
                    try:
                        node_value.append(tree['F0']['F'+str(i1)]['F'+str(i1)+'.'+str(counter)])
                        a='F'+str(i1)+'.'+str(counter)
                        if a in weights.keys():
                            weights_value.append(weights[a])

                    except: 
                        print("Weird")

                    counter+=1

                k=func(weights_value,node_value)
                if k!=[] and k!=None:
                    tree['F0']['F'+str(i1)]=k
                    print("The value of a factor "+'F'+str(i1)+" is: ", k)
                    max_v,rate=similarity(k)
                    print("The similarity rate V of a factor "+'F'+str(i1)+" and linguistic estimation \""+rate+"\" is: "+str(max_v))
                    print("-------------------------------------------------------------------------------")

        except:
            pass

        i1+=1

    node_value=[]
    weights_value=[]

    i1=1
    while i1<=4:
        node_value.append(tree['F0']['F'+str(i1)])
        a='F'+str(i1)
        if a in weights.keys():
            weights_value.append(weights[a])

        i1+=1

    tree['F0']=func(weights_value,node_value)
    print("The value of a factor F0 is: ", tree['F0'])

    max_v,rate=similarity(tree['F0'])
    print("The similarity rate V of a factor F0 and linguistic estimation \""+rate+"\" is: "+str(max_v))

    default_rate=default(rate)
    print("Default probability: "+default_rate)

if __name__ == "__main__": 
    t1=time.perf_counter()
    main()
    t2=time.perf_counter()
    print(f'Finished in {t2-t1} seconds')
