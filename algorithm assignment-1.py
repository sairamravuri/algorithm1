#!/usr/bin/env python
# coding: utf-8

# In[9]:


#uniform distribution
import numpy as np
import csv
UD=[]
for i in range(17):
    UD.append(np.random.uniform(low=0,high=100000,size=2**i))
with open("ud.txt","w") as ud:
    csv.writer(ud,delimiter=' ').writerows(UD)


# In[11]:


import numpy as np
import csv
ND=[]
for i in range(17):
    ND.append(np.random.normal(size=2**i))
with open("nd.txt","w") as nd:
    csv.writer(nd,delimiter=' ').writerows(ND)


# In[9]:


#MERGESORT
def MergeSort(arr):
    if len(arr)>1:
        mid=len(arr)//2
        L=arr[:mid]
        R=arr[mid:]
        MergeSort(L)
        MergeSort(R)
        i=j=k=0
        while i<len(L) and j<len(R):
            if L[i]<R[j]:
                arr[k]=L[i]
                i+=1
            else:
                arr[k]=R[j]
                j+=1
            k+=1
        while i<len(L):
            arr[k]=L[i]
            i+=1
            k+=1
        while j<len(R):
            arr[k]=R[j]
            j+=1
            k+=1

            
            
a=[10,1,3,8,2,12]
MergeSort(a)
print(a)


# In[16]:


#QUICKSORT
def partition(arr,start,end):
    pivot=arr[start]
    i,j=start+1,end
    while i<=j:
        while i<=end and arr[i]<=pivot:
            i+=1
        while j>=start and arr[j]>pivot:
            j-=1
        if i<j:
            arr[i],arr[j]=arr[j],arr[i]
    arr[start],arr[j]=arr[j],arr[start]
    return j
def QuickSort(a,l,h):
    if l==h:
        return a
    if l<h:
        m=partition(a,l,h)
        QuickSort(a,l,m-1)
        QuickSort(a,m+1,h)
        
        
b=[1,5,3,10,2,4]
QuickSort(b,0,len(b)-1)
print(b)   


# In[17]:


comparisions=0
swaps=0
def compare(x,y):
    global comparisions
    comparisions+=1
    if x>y: return -1
    if x<y: return 1
    return 0


# In[18]:


#MODIFIED MERGESORT
def MergeSort1(arr):
    global swaps
    if compare(len(arr), 1) == -1 :
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]
        MergeSort1(L)
        MergeSort1(R)
        i = j = k = 0
        while compare(i, len(L)) == 1 and compare(j, len(R)) == 1:
            if compare(L[i], R[j]) == 1:
                arr[k] = L[i]; swaps += 1
                i += 1
            else:
                arr[k] = R[j]; swaps += 1
                j += 1
            k += 1
        while compare(i, len(L)) == 1:
            arr[k] = L[i]; swaps += 1
            i += 1
            k += 1
        while compare(j, len(R)) == 1:
            arr[k] = R[j]; swaps += 1
            j += 1
            k += 1


# In[19]:


ud = open("ud.txt","r")
nd = open("nd.txt","r")
uniform_data = [line.rstrip() for line in ud]
normal_data = [line.rstrip() for line in nd]
comp_uniform = [[], []]
comp_normal = [[],[]]
swaps_uniform = [[],[]]
swaps_normal = [[],[]] 


# In[20]:


for x in uniform_data:
    arr = list(map(float, x.split()))
    MergeSort1(arr)
    comp_uniform[0].append(len(arr))
    comp_uniform[1].append(comparisions)
    comparisions = 0
    swaps_uniform[0].append(len(arr))
    swaps_uniform[1].append(swaps)
    swaps = 0
    
print(comp_uniform[1])
print(swaps_uniform[1])


# In[21]:


for x in normal_data:
    arr = list(map(float, x.split()))
    MergeSort1(arr)
    comp_normal[0].append(len(arr) )
    comp_normal[1].append(comparisions)
    comparisions = 0
    swaps_normal[0].append(len(arr))
    swaps_normal[1].append(swaps)
    swaps = 0
print(comp_normal[1])
print(swaps_normal[1]) 


# In[22]:


from matplotlib import pyplot as plt 
def plot_fig(data, attr):
    plt.plot(data[0], data[1])
    plt.xlabel('size')
    plt.ylabel(attr)
plt.show() 


# In[23]:


plot_fig(comp_uniform, "comparisions uniform")
comparisions, swaps = 0, 0


# In[24]:


plot_fig(comp_normal, "comparisions normal")


# In[25]:


plot_fig(swaps_uniform, "copies uniform") 


# In[26]:


plot_fig(swaps_normal, "copies normal")


# In[35]:


comparisions, swaps = 0, 0
def partition1(a, p, r):
    global swaps
    x = a[p]
    i, j = p+1, r
    while i <= j:
        while i < len(a) and compare(a[i], x) >= 0: i += 1;
        while j > -1 and compare(a[j], x) == -1: j -= 1;
        if i < j: a[i], a[j] = a[j], a[i]; swaps +=1
    a[p], a[j] = a[j], a[p]
    swaps += 1
    return j 


# In[39]:


def QuickSort1(a, p, r):
    if compare(p, r)==0: return a
    if compare(p, r)==1:
        q = partition1(a, p, r)
        QuickSort1(a, p, q-1)
        QuickSort1(a, q+1, r) 


# In[40]:


comp_uniform = [[], []]
comp_normal = [[], []]
swaps_uniform = [[], []]
swaps_normal = [[], []] 


# In[41]:


for x in uniform_data:
    arr = list(map(float, x.split()))
    QuickSort1(arr, 0, len(arr)-1)
    comp_uniform[0].append(len(arr) )
    comp_uniform[1].append(comparisions)
    comparisions=0
    swaps_uniform[0].append(len(arr) )
    swaps_uniform[1].append(swaps)
    swaps = 0 


# In[42]:


for x in normal_data:
    arr = list(map(float, x.split()))
    QuickSort1(arr, 0 , len(arr)-1)
    comp_normal[0].append(len(arr) )
    comp_normal[1].append(comparisions)
    comparisions = 0
    swaps_normal[0].append(len(arr) )
    swaps_normal[1].append(swaps)
    swaps = 0 


# In[43]:


plt.plot(comp_uniform[0], comp_uniform[1], color = '#58b970', label = "Uniform")
plt.xlabel("size")
plt.ylabel("comparisons")
plt.show()


# In[45]:


plt.plot(comp_normal[0], comp_normal[1], color = "#ef5423", label = "Normal")
plt.xlabel('size')
plt.ylabel('comparisons' )
plt.show() 


# In[46]:


print(swaps_uniform[1])
plt.plot(swaps_uniform[0], swaps_uniform[1], color = "#58b970", label = "Uniform")
plt.xlabel('size')
plt.ylabel('Swaps')
plt.show()


# In[47]:


print(swaps_normal[1])
plt.plot(swaps_normal[0], swaps_normal[1], color = "#58b970", label = "Normal")
plt.xlabel('size')
plt.ylabel('Swaps')
plt.show() 


# In[79]:


import sys 
sys.setrecursionlimit(1500)
ud = open("ud.txt","r")
nd = open("nd.txt","r")
uniform_data = [line.rstrip() for line in ud]
normal_data = [line.rstrip() for line in nd]
comp_uniform = [[], []]
comp_normal = [[],[]]
swaps_uniform = [[],[]]
swaps_normal = [[],[]] 
comparisions=0
swaps=0


# In[80]:


from random import randint, randrange 
def quicksort(arr, start , stop):  
    if(compare(start, stop) == 1): 
        pivotindex = partitionrand(arr, start, stop) 
        quicksort(arr , start , pivotindex-1) 
        quicksort(arr, pivotindex + 1, stop) 
def partitionrand(arr , start, stop): 
    global swaps 
    randpivot = randrange(start, stop) 
    arr[start], arr[randpivot] = arr[randpivot], arr[start]; swaps += 1 
    return partition(arr, start, stop) 
def partition(arr,start,stop): 
    global swaps 
    pivot = start 
    i = start +1 
    for j in range(start + 1, stop + 1): 
        if compare(arr[j], arr[pivot]) >= 0: 
            arr[i] , arr[j] = arr[j] , arr[i]; swaps += 1 
            i=i+1 
    arr[pivot] , arr[i - 1] = arr[i - 1] , arr[pivot]; swaps += 1 
    pivot =i-1 
    return (pivot) 


# In[81]:


for x in uniform_data: 
    arr = list(map(float, x.split()))
    quicksort(arr, 0,len(arr)-1) 
    comp_uniform[0].append(len(arr)) 
    comp_uniform[1].append(comparisions) 
    comparisions=0 
    swaps_uniform[0].append(len(arr)) 
    swaps_uniform[1].append(swaps) 
    swaps = 0


# In[82]:


for x in normal_data: 
    arr = list(map(float, x.split())) 
    quicksort(arr, 0, len(arr)-1) 
    comp_normal[0].append(len(arr)) 
    comp_normal[1].append(comparisions) 
    comparisions = 0 
    swaps_normal[0].append(len(arr)) 
    swaps_normal[1].append(swaps) 
    swaps=0


# In[83]:


plt.plot(comp_uniform[0], comp_uniform[1], color = "#58b970", label = "Uniform") 
plt.xlabel("size") 
plt.ylabel("comparisons") 
plt.show() 


# In[84]:


plt.plot(comp_normal[0], comp_normal[1], color = "#ef5423", label = "Normal") 
plt.xlabel("size") 
plt.ylabel("comparisons") 
plt.show()


# In[86]:


print(swaps_uniform[1]) 
plt.plot(swaps_uniform[0], swaps_uniform[1], color = "#58b970", label = "Uniform") 
plt.xlabel("size") 
plt.ylabel("Swaps") 
plt.show() 


# In[88]:


print(swaps_normal[1]) 
plt.plot(swaps_normal[0], swaps_normal[1], color = "#ef5423", label = "Normal") 
plt.xlabel("size") 
plt.ylabel("Swaps") 
plt.show()


# In[93]:


import numpy as np
import csv 
UD = []
for i in range(17):
    UD. append(np.random.uniform(low=0, high = 1, size= 2**i))
with open("ud1.txt", "w") as ud:
    csv.writer(ud, delimiter=" ").writerows(UD) 


# In[94]:


ND = []
for i in range(17):
    ND. append(np. random. random(2**i))
with open("nd1.txt", "w") as nd:
    csv.writer(nd, delimiter=" ").writerows(ND) 


# In[3]:


#BUCKETSORT
def insertionSort(b):
    for i in range(1, len(b)):
        up = b[i]
        j = i - 1
        while j >= 0 and b[j] > up:
            b[j + 1] = b[j]
            j -= 1
        b[j + 1] = up    
    return b    
             
def bucketSort(x):
    arr = []
    slot_num = 10
    for i in range(slot_num):
        arr.append([])
    for j in x:
        index_b = int(slot_num * j)
        arr[index_b].append(j)
    for i in range(slot_num):
        arr[i] = insertionSort(arr[i])
    k = 0
    for i in range(slot_num):
        for j in range(len(arr[i])):
            x[k] = arr[i][j]
            k += 1
    return x
 

x = [0.897, 0.565, 0.656,0.1234, 0.665, 0.3434]
print("Sorted Array is")
print(bucketSort(x))


# In[4]:


ud = open("ud1.txt", "r")
nd = open("nd1.txt", "r")
uniform_data = [line.rstrip() for line in ud]
normal_data = [line.rstrip() for line in nd]
time_uniform = [[], []]
time_normal = [[], []] 


# In[5]:


import time 
for x in uniform_data:
    arr = list(map(float, x.split()))
    begin = time.time()
    bucketSort(arr)
    end = time.time()
    time_uniform[0].append(len(arr))
    time_uniform[1].append(end-begin) 


# In[5]:


from matplotlib import pyplot as plt 
plt.plot(time_uniform[0], time_uniform[1], color = "#58b970", label = "Time taken Bucket Sort")
plt.xlabel("size")
plt.ylabel("Time")
plt.show()          


# In[8]:


for x in normal_data: 
    arr = list(map(float, x.split())) 
    begin = time.time() 
    bucketSort(arr) 
    end = time.time() 
    time_normal[0].append(len(arr)) 
    time_normal[1].append(end-begin) 


# In[10]:


from matplotlib import pyplot as plt 
plt.plot(time_normal[0], time_normal[1], color = "#ef5423", label = "Time taken Bucket Sort") 
plt.xlabel("size") 
plt.ylabel("Time") 
plt.show() 


# In[15]:


def findMedian(a, p, r): 
    L =[]
    for i in range(p, r+1): 
        L.append(a[i]) 
    L.sort() 
    return L[(len(L)//2)] 
def KthSmallest(a, p, r, k): 
    n=r-p+1 
    median = []
    i=0 
    while i < n//5: 
        median.append(findMedian(a, p+5*i, p+5*i+4)) 
        i+=1 
    if i*5 <n: 
        median.append(findMedian(a, p+5*i, p+5*i+(n%5-1))) 
        i+=1 
    if i ==1: 
        medofmed = median[i-1] 
    else: 
        medofmed = KthSmallest(median, 0, i-1, i//2) 
    q = partition(a, p, r, medofmed) 
    i=q-p+1 
    if i == k: 
        return a[q] 
    elif i> k: 
        return KthSmallest(a, p, q-1, k) 
    else: 
        return KthSmallest(a, q+1, r, k-i) 


# In[16]:


def partition(a, p, r, x): 
    for i in range(p, r+1): 
        if a[i] == x: 
            a[i], a[r] = a[r], a[i] 
            break 
    i=p-1 
    for j in range(p, r): 
        if a[j] <= a[r]: 
            i+=1 
            a[i], a[j] = a[j], a[i] 
    a[i+1], a[r] = a[r], a[i+1] 
    return i+1 


# In[17]:


def QuickSort(a, p, r): 
    if p>=r: 
        return 
    med = KthSmallest(a, p, r, (r-p+1)//2) 
    q = partition(a, p, r, med) 
    QuickSort(a, p, q-1) 
    QuickSort(a, q+1, r) 

s= [1,4,8,6,12,2,3] 
QuickSort(s, 0, len(s)-1) 
print(s) 


# In[18]:


def KthSmallest_new(a, p, r, k, s): 
    n=r-p+1 
    median = [] 
    i=0 
    while i < n//s: 
        median. append(findMedian(a, p+s*i, p+s*i+s-1)) 
        i+=1 
    if i*s <n: 
        median.append(findMedian(a, p+s*i, p+s*i+(n%s-1))) 
        i+=1 
    if i == 1:
        medofmedian=median[i-1]
    else: 
        medofmedian = KthSmallest_new(median, 0, i-1, i//2, s) 
    q = partition(a, p, r, medofmedian) 
    i=q-p+1
    if i == k:
        return a[q] 
    elif i > k: 
        return KthSmallest_new(a, p, q-1, k, s) 
    else: 
        return KthSmallest_new(a, q+1, r, k-i, s) 

def Quick_Sort(a, p, r, s): 
    if p >= r: 
        return 
    med = KthSmallest_new(a, p, r, (r-p+1)//2, s) 
    q = partition(a, p, r, med) 
    Quick_Sort(a, p, q-1, s) 
    Quick_Sort(a, q+1, r, s) 


# In[19]:


ud =open("ud.txt","r")
nd = open("nd.txt","r") 
uniform_data = [line.rstrip() for line in ud] 
normal_data = [line.rstrip() for line in nd] 
time_uniform_3 = [[], []] 
time_normal_3 = [[], []] 
time_uniform_5 = [[], []] 
time_normal_5 = [[], []] 
time_uniform_7 = [[], []] 
time_normal_7 = [[], []] 


# In[20]:


import time
for i in range(len(uniform_data)): 
    arr = list(map(float, uniform_data[i].split())) 
    begin = time.time() 
    Quick_Sort(arr, 0, len(arr)-1, 3) 
    end = time.time() 
    time_uniform_3[0].append(len(arr)) 
    time_uniform_3[1].append(end-begin) 


# In[31]:


for i in range(len(uniform_data)): 
    arr = list(map(float, uniform_data[i].split())) 
    begin = time.time() 
    Quick_Sort(arr, 0, len(arr)-1, 5) 
    end = time.time() 
    time_uniform_5[0].append(len(arr) ) 
    time_uniform_5[1].append(end-begin) 


# In[12]:


for i in range(len(uniform_data)): 
    arr = list(map(float, uniform_data[i].split())) 
    begin = time.time() 
    Quick_Sort(arr, 0, len(arr)-1, 7) 
    end = time.time() 
    time_uniform_7[0].append(len(arr)) 
    time_uniform_7[1].append(end-begin) 


# In[34]:


from matplotlib import pyplot as plt
plt.plot(time_uniform_3[0], time_uniform_3[1], color = "r",label="size=3")
plt.plot(time_uniform_5[0], time_uniform_5[1], color = "g",label="size=5")
plt.plot(time_uniform_7[0], time_uniform_7[1], color = "b",label ="size=7") 
plt.legend() 
plt.show() 


# In[21]:


for i in range(len(normal_data)): 
    arr = list(map(float, normal_data[i].split())) 
    begin = time.time() 
    Quick_Sort(arr, 0, len(arr)-1, 3) 
    end = time.time() 
    time_normal_3[0].append(len(arr)) 
    time_normal_3[1].append(end-begin) 


# In[22]:


for i in range(len(normal_data)): 
    arr = list(map(float, normal_data[i].split())) 
    begin = time.time() 
    Quick_Sort(arr, 0, len(arr)-1, 5) 
    end = time.time() 
    time_normal_5[0].append(len(arr)) 
    time_normal_5[1].append(end-begin) 


# In[23]:


for i in range(len(normal_data)): 
    arr = list(map(float, normal_data[i].split())) 
    begin = time.time() 
    Quick_Sort(arr, 0, len(arr)-1, 7) 
    end = time.time() 
    time_normal_7[0].append(len(arr)) 
    time_normal_7[1].append(end-begin) 


# In[25]:


from matplotlib import pyplot as plt
plt.plot(time_normal_3[0], time_normal_3[1], color = "r", label = "size = 3") 
plt.plot(time_normal_5[0], time_normal_5[1], color = "g", label = "size = 5") 
plt.plot(time_normal_7[0], time_normal_7[1], color = "b", label = "size = 7")
plt.legend()
plt.show()


# In[1]:


ud = open("ud.txt","r") 
nd = open("nd.txt","r") 
uniform_data = [line.rstrip() for line in ud] 
normal_data = [line.rstrip() for line in nd] 
udata, ndata = [], [] 


# In[2]:


import sys 
sys.setrecursionlimit(2500)
def select(L): 
    if len(L) < 10: 
        L.sort() 
        return L[int(len(L)/2)] 
    S=[] 
    lIndex = 0 
    while lIndex+5 < len(L)-1: 
        S.append(L[lIndex: lIndex+5]) 
        lIndex += 5 
    S.append(L[lIndex: ]) 
    Meds = [] 
    for subList in S: 
        Meds.append(select(subList))
    L2 = select(Meds) 
    L1 = L3 = [] 
    for i in L: 
        if i< L2: 
            L1.append(i) 
        if i> L2: 
            L3.append(i) 
    if len(L) < len(L1): 
        return select(L1) 
    elif len(L) > len(L1) + 1: 
        return select(L3) 
    else: 
        return L2 


# In[3]:


import numpy as np
for x in uniform_data: 
    arr = list(map(float, x.split())) 
    np.random.shuffle(arr) 
    MoM = select(arr) 
    arr.sort() 
    ind = arr.index(MoM) 
    udata.append(abs((len(arr))//2 - ind-1))
print(udata) 


# In[4]:


for x in normal_data: 
    arr = list(map(float, x.split())) 
    np.random.shuffle(arr) 
    MoM = select(arr) 
    arr.sort() 
    ind = arr.index(MoM) 
    ndata.append(abs((len(arr))//2 - ind-1)) 
print(ndata)


# In[6]:


from matplotlib import pyplot as plt
plt.plot(udata) 
plt.show()


# In[7]:


plt.plot(ndata) 
plt.show() 

