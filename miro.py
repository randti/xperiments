
"""
Сортировка пузурьком

def buble_sort(array):
    n=len(array)
    for i in range(1,n):
        for j in range(n-i):
            if array[j] > array[j+1]:
                                array[j],array[j+1]=array[j+1],array[j]
n=int(input())
array = list(map(int, input().split()))
buble_sort(array)
print(*array)
"""
"""
Сортировка максимума
def sort_array(array):
    my_list=[]
    for i in range 6(len(array)):
        minn=min(array)
        my_list.append(minn)
        array.remove(minn)
    for i in range (len(my_list)):
        array.append(my_list[i])
n=int(input())
array = list(map(int, input().split()))
sort_array(array)
print(*array)
""" 
"""   
сортировка вставками
def insertsort(array):
    a=len(array)
    for i in range(1,a):
        z=i
        b = array[i]
        while (array[z-1] > b) and (z > 0):
            array[z] = array[z-1]
            z = z - 1
        array[z] = b
       
n=int(input())
array = list(map(int, input().split()))
insertsort(array)
print(*array)
"""
"""
быстрая сортировка
import random
from random import randint
 
def qSort ( A, nStart, nEnd ):
                if nStart >= nEnd: return
                L = nStart; R = nEnd
                X = A[(L+R)//2]
                while L <= R:
                                while A[L] < X: L += 1
                                while A[R] > X: R -= 1
                                if L <= R:
                                                A[L], A[R] = A[R], A[L]
                                                L += 1; R -= 1
                qSort ( A, nStart, R )
                qSort ( A, L, nEnd )
n=int(input())
array = list(map(int, input().split()))
qSort(array,0,n-1)
print(*array)
 
"""
"""
сортировка слиянием
def merge(a,b):
  n=len(a)
  m=len(b)
  res=[0]*(n+m)
  a.append(10000000)
  b.append(10000000)
  i=0
  j=0
  for k in range(n+m):
    if a[i]<b[j]:
      res[k]=a[i]
      i+=1
    else:
      res[k]=b[j]
      j+=1
  return res
 
def mergesort(lst):
   
    length = len(lst)
    if length >= 2:
        mid = int(length / 2)
        lst = merge(mergesort(lst[:mid]), mergesort(lst[mid:]))
    return lst
       
n=int(input())
array = list(map(int, input().split()))
print(*mergesort(array))
"""




"""
import time
print(time.ctime()) #время 
перевод числа
def dvoich(a):
  x=[]
  if a==0:
   x.append(a)
     
  else:
    while (a!=1):
      p=a%2
      x.append(p)
      a=a//2
    x.append(a)
    x.reverse()
  return(x)
a=int(input())
for k in range(a):

  x=dvoich(k)
  for i in range (len(x)):
    print(x[i],end="") #без комментариев
  print('')
"""
"""
array = list( input())
arrat = list( input())
h=len(array)
b=len(arrat)
if (h!=b):
    print('NO')
p=0
j=0


for z in range(h):
    u=array[z]
    n=array.count(u)
    w=arrat.count(u)
    if (n!=w):
        p=p+1
if (p==0):
    print('YES')
else:
    print('NO')
            
"""  
"""
корята
a=list(map(int,input().split()))
coords=list(map(int,input().split()))
k=a[1]
def is_correct(x):
    cows=1
    cow=coords[0]
    for c in coords:
        if (c-cow)>=x:
            cows +=1
            cow=c
    return (cows>=k)
left=0
right=max(coords)-min(coords)+1
while (right-left)!=1:
    middle=(left+right)//2
    if is_correct(middle):
        left=middle
    else:
        right=middle
print (left)        

"""



"""  
количество в различных записях
def poisk(a,b,c):
  x=[]
  if a==0:
     x.append(a)
     
  else:
    while (a!=1):
      p=a%b
      x.append(p)
      a=a//b
    x.append(a)
    x.reverse()
  d=x.count(c)

  return(d)

a=49**7+7**20-28
b=7
c=6
print(poisk(a,b,c))



количество делителей
def is_prime(b):
    if (b==1):
        return False
    else:
        i=2
        while i*i<=b:
            if (b%i==0):
                return False
            i=i+1
        return True    
 
def de_cay(num):
    if is_prime(num)==True:
        u=2
    else:
        m=0
        u=1    
        i=2
        
        while (i)<=(num):
            m=0
            while (num%i==0):
                num=num//i
                m=m+1
 
            u=u*(m+1)
            i=i+1
   
    print(u)
   
       
num=int(input())
 
de_cay(num)
"""
"""
кубическое уравнение бинпоиском
p=list(map(int,input().split()))
a=p[0]
b=p[1]
c=p[2]
d=p[3]
left=-1000000.0
right=1000000.0
if a<0:
    a=-a
    b=-b
    c=-c
    d=-d
 
for i in range(100):
    middle=(left+right)/2
    x=middle
    if ((a*x*x*x)+(b*x*x)+(c*x)+d) <0:
        left=middle
    else:
        right=middle
print(left)
"""
"""

словарь синонимов
n = int(input())
d = {}
for i in range(n):
    a, b = input().split()
    d[a] = b
    d[b] = a
print(d[input()])
"""

"""
чо-то со словарями
n = int(input())
d = []
for i in range(n):
    a,n, b = input().split()
    d.append(b)
print(max(d, key=d.count))

d = dict()
if 1 not in d:
    d[1] = list()

d[1].append(1)
d[1].append(2)
d[1].append(3)

if 2 not in d:
    d[2] = list()

d[2].append(1)
d[2].append(1)
d[2].append(1)

print (d[1])
print (d[2])
"""
"""

Бинпоиск массива
import random
from random import randint
 
def qSort ( A, nStart, nEnd ):
                if nStart >= nEnd: return
                L = nStart; R = nEnd
                X = A[(L+R)//2]
                while L <= R:
                                while A[L] < X: L += 1
                                while A[R] > X: R -= 1
                                if L <= R:
                                                A[L], A[R] = A[R], A[L]
                                                L += 1; R -= 1
                qSort ( A, nStart, R )
                qSort ( A, L, nEnd )
 
 
 
 
 
 
 
def sebch(a,x):
 
    n=len(a)
 
    left=-1
    right=n
    while (right-left)>1:
        middle=(left+right)//2
        if a[middle]>=x:
            right=middle
        else:
            left=middle
    if (right!=n) and (a[right]==x):
        return True
    else:
        return False
 
n=0
w=int(input())
 
a = list(map(int, input().split()))
q=int(input())
b = list(map(int, input().split()))
 
qSort ( a, 0, w-1 )
 
for u in range(len(b)):
    z=1
   
    while z>0:
        x=b[u]
        k=sebch(a,x)
        if k==True:
            j=x
            a.remove(j)
           
 
            n=n+1
           
        else:
            z=z-1
        v=n    
    if (b[u]==b[u-1]):
        print(v,end = " " )
    else:
        print(n, end = " ")
        n=0
"""

"""

кинотеатр
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
a=list(map(int,input().split()))
b=a[0]
c=a[1]
if (b==1) or (c==1):
    print(b*c)
else:
    b=gcd(b-1,c-1)
    print(b+1)
"""




"""
import sys
print (sys.version)
версия языка

гипотеза гольдбаха
def isprime(b):
    if (b==1):
        return 0
    else:
        i=2
        while(i*i<=b):
            if (b%i==0):
                return 0
            i=i+1
        return 1
g=int(input())
i=2
 
while (isprime(g-i)+isprime(i))!=2:
   
    i=i+1
print(g-i,i)
"""



"""
Шестеренки
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
a=list(map(int,input().split()))
 
n=a[0]
j=a[1]
b=n*j
print(b//gcd(n,j))
"""

"""
алгоритм евклида в рекурсии
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
n=int(input())
j=int(input())
print(gcd(n,j))
"""

"""
Решето Эратосфена
N = 50

is_prime = [True] * (N + 1)  # просто заполняем массив
is_prime[0], is_prime[1] = False, False
for i in range(2, N + 1):  # можно и до sqrt(N)
    if is_prime[i] is True:
        for j in range(2 * i, N + 1, i):  # идем с шагом i, можно начиная с i * i
            is_prime[j] = False

for i in range(1, N + 1):
    if is_prime[i]:
        print(i)

for i in range(1, N + 1):
    print(i, '\t', is_prime[i])

N = 30
primes = []
min_d = [0] * (N + 1)

for i in range(2, N + 1):
    if min_d[i] == 0:
        min_d[i] = i
        primes.append(i)
    for p in primes:
        if p > min_d[i] or i * p > N:
            break
        min_d[i * p] = p
    print(i, min_d)
print(min_d)
print(primes)
"""

"""
Различные возведения в степень
def expt(b, n):
    if n == 0:
        return 1
    return b * expt(b, n - 1)


def expt_iter(b, n):
    def exptIter(counter, product):
        if counter == 0:
            return product
        return exptIter(counter - 1, b * product)

    return exptIter(n, 1)

def even(n):
    if n % 2 == 0:
        return 1
    return 0

def fast_exp(b, n):
    if n == 0:
        return 1
    if even(n):
        return fast_exp(b, n / 2) ** 2
    return b * fast_exp(b, n - 1)


def fast_pow(a, k):  # возведем а в степень k
    accum = 1
    while k:
        if k % 2:
            accum *= a
        a *= a
        k //= 2
    return accum

"""
"""
задача про отрезки
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
a=list(map(int,input().split()))
b=abs(a[0]-a[2])
c=abs(a[1]-a[3])
print(b+c-gcd(b,c))
"""


"""
Мое решето 
def resheto(a):
    
    q =[x for x in range(a[0],a[1]+1)]
    i=2
    m=0
    while (i*i)<=a[1]:
        k=0
        for t in range(len(q)):
            
            
            if (q[t]% i==0) and (q[t]!=i):
                q[t]=0
                k=k+1
                
                 
        i=i+1
    q.sort()

        

  
    for j in range(k,len(q)):
        print(q[j])        

b=list(map(int, input().split()))
t=time.time()
resheto(b)
print(time.time()-t)
"""

"""
меганод
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def mego(a):
    s=a[0]  
    for i in range(1,len(a)):
        s=gcd(a[i],s)

        
    return s
n=int(input())
b=list(map(int, input().split()))
print(mego(b))
"""



"""
Проверка линейного
уравнения
def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)
 
def diof(a, b,c):
   
    if (c%gcd(a,b)!=0):
        print('Impossible')
    else:
        
        x=0
        y=c-a*x
        while (y%b!=0):
            y=c-(a*x)
            
            x=x+1
            
        y=y//b
        if x==0:
            print(gcd(a,b),x,y)
        else:
            print(gcd(a,b),x-1,y)
   
 
a=list(map(int,input().split()))
 
diof(a[0],a[1],a[2])
"""



"""
Решение окнов с помощью кучи
import heapq
n,k=map(int,input().split())
a=list(map(int,input().split()))
h=list(zip(a[:k],range(k)))
heapq.heapify(h)
print(h[0][0])
for i in range(k,n):
  heapq.heappush(h,(a[i],i));
  while h[0][1]<=i-k:
    heapq.heappop(h)
  print(h[0][0])
"""




"""
факторизация со знаками
def factorize(num):
   
    i = 2
   
    while i * i <= num:  
        u=0
        q=0
        while num % i == 0:  
            num //= i  
            u=u+1
            q=1
        if u>1:
            print(i,u,sep="^",end="")
        else:
            if u==1:
                print(i,end="")            
        if (q==1) and (num>1):
            print("*",end="")
        i += 1
 
   
    if num > 1:
        print(num,end="")
       
a=int(input())
factorize(a)
"""


"""
выбор заявок 
from typing import List, Tuple,Any
def meets(input_array:List[Tuple[Any,Any]])-> List[Tuple[Any,Any]]:
  result=[]
  sorted_array=sorted(input_array,key=lambda x :x[1])
  is_pol=False
  while True:
    if not sorted_array:
      break
    item=sorted_array.pop(0)
    left,right=item
    sorted_array=list(filter(lambda x:x[0]>=right, sorted_array))
    result.append(item)
  return result
am=int(input())
r=[]
for i in range (am):
  a,b=map(int,input().split())
  r.append((a,b))
result=meets(r)
print(len(result))
"""

"""
сапожник

a=list(map(int,input().split()))
b=list(map(int, input().split()))
b=sorted(b)

k=a[0]
i=0
z=0
p=sum(b)

while (z<=k) and (i<len(b)):
  z=z+b[i]
  i=i+1
if p<=k:
  print(i)

else:
  a=i-1
  print(a)
"""

"""
решение с помощью кучи про паспорта
import heapq
n=int(input())
a=list(map(int,input().split()))
k=len(a)
h=list(zip(a[:k],range(k)))
heapq.heapify(h)
m=[]
y=0
f=0
r=0
while (len(h)>1):
  y=h[0][0]
  heapq.heappop(h)
  f=h[0][0]+y
  r=r+f
  m.append(f)
  heapq.heappop(h)
  heapq.heappush(h,(m[0],0))
  m.clear()
print(r)
"""
"""
города
n=int(input())
m=[]
y=0
for a in range (n):
  z=list(input())
  p=z.count('C')
  y=y+p
  m.append(z)

w=y//2

h=0

for i in range (n):
  for b in range(n):
    if (m[i][b]=='D' and(h==0))  :
      m[i][b]=1
for i in range (n):
  for b in range(n):
    if (m[i][b]=='C' and  (y!=w)) :
      m[i][b]=1
      y=y-1
    else:
      if (m[i][b]=='C' and  (y<=w)):
        m[i][b]=2
k=[]
for i in range (n):
  for b in range(n):
    j=m[i][b]
    k.append(j)

i=0

while k[i]!=2:
  i=i+1
for b in range(i,len(k)):
  k[b]=2

for a in range (len(k)):
  print(k[a],end="")
  if (a+1)%n==0 :
    print()
  

"""
"""
pars={a:0 for a in range(1,11)}
print(pars)
n=int(input())
for i in range(n):
  l=int(input())
  pars[l]=pars[l]+1
pars=(sorted(pars.items(), key=lambda t:t[1]))
for i in pars:
    if i[1]!=0:
        print(*i)
"""
"""
import tensorflow as tf
import timeit

# See https://www.tensorflow.org/tutorials/using_gpu#allowing_gpu_memory_growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.device('/cpu:0'):
  random_image_cpu = tf.random_normal((100, 100, 100, 3))
  net_cpu = tf.layers.conv2d(random_image_cpu, 32, 7)
  net_cpu = tf.reduce_sum(net_cpu)

with tf.device('/gpu:0'):
  random_image_gpu = tf.random_normal((100, 100, 100, 3))
  net_gpu = tf.layers.conv2d(random_image_gpu, 32, 7)
  net_gpu = tf.reduce_sum(net_gpu)

sess = tf.Session(config=config)

# Test execution once to detect errors early.
try:
  sess.run(tf.global_variables_initializer())
except tf.errors.InvalidArgumentError:
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise

def cpu():
  sess.run(net_cpu)
  
def gpu():
  sess.run(net_gpu)
  
# Runs the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

sess.close()
"""
"""
круглый стол 
a,b=map(int,input().split())

for i in range(a+b):
  if (i%3!=2) and (b!=0):
    print('G',end="")
    b=b-1
  else:
    if (a!=0):
      print('B',end="")
      a=a-1
    else:
      print('G',end="")
"""
"""
def binop(a,b):
    
      n=len(a)
      left=-1
      right=n

      x=b
       
      while (right-left)>1:
        middle=(left+right)//2
        if a[middle]>=x:
          right=middle
        else:
          left=middle
      r=right
      if r==n:
        r=r-1
 
      k=(a[r-1]-b)
      p=(a[r]-b)
           
      if p<0:
        p=p*(-1)
 
      if k<0:
        k=k*(-1)
           
       
      if (k<=p) and (a[r-1]<a[r]):
        return(a[r-1])
      else:
        return(a[r])
               

a,b=map(int,input().split())
k=list(map(int,input().split()))
k.pop(0)

z=a-b
r=0
y=0
if (a<=b):
  print(0)
else:
  while (y<=a) and (r!=len(k)):
    y=binop(k,b)+y+b
    r=r+1
    
  if (y<a) or b<k[0]):
    print(-1)
  else:
    print(r)
"""





"""
import time
n=int(input())
ti=time.time()
t=[]
for i in range(n):
  for j in range(i+1):
    if i%2==0:
      z=1/((i-j+1)/(j+1))
    else:
      z=((i-j+1)/(j+1))
    t.append(z)
l =list(dict([(item, None) for item in t]).keys())
print(*l)

print('Время работы:',time.time()-ti)

"""



"""
b=set()
for i in range(len(t)):
  z=len(b)
  b.add(t[i])
  if len(b)==z:
    t[i]=None
k=filter(lambda a: a !=None , t)
print(*k)
print('Время работы:',time.time()-ti)
"""






"""
b=int(input())
a=list(map(int,input().split()))
c=a.copy()
a=sorted(a)
a.reverse()



d=[2**i for i in range(31)]
j=[]
g=0
while g!=(len(a)):
  if b>=a[g]:
    b=b-a[g]
    j.append(a[g])
  g=g+1

r=0
for g in range(len(j)):
  z=d[c.index(j[g])]
  r=r+z
print(r)
"""

"""
сортировка кучей
def hsort(a):
  import heapq
  k=len(a)
  h=list(zip(a[:k],range(k)))
  heapq.heapify(h)
  p=[]
  for i in range(k):
    z=h[0][0]
    heapq.heappop(h)
    p.append(z)
  return p
a=list(map(int,input().split()))
print (*hsort(a))
"""


"""
f=open("costumes.in.txt", "r")
n=f.readline()
a=[]
a=f.readline()
print(*a)
c=[]
for i in range(len(a)):
    if a[i]!=' ':
        b=a[i]
        c.append(b)
h=int(f.readline())
d=[]
for i in range(h):
    a=f.readline()
print(*c)
"""


"""
кузнечик
n=20 
badg=[2,3,6,12]
dp=[0]*(n+1)
is_bad=[0]*(n+1)
for cell in bad_cells:
  is_bad[cell]=1
dp[1]=1
dp[2]=1
dp[3]=2


if  not is_bad[1]:
  dp[1]=1



for i in range(2,n+1):
  if not is_badg[i]:
    for k in range(1,4):
      if i-k>=1:
        dp[i] += dp[i-k]
"""



"""
фибоначчи
h=int(input())
N=50
a=[0]*N
a[0]=0
a[1]=1

for i in range(2,h+1):
  a[i]=a[i-1]+a[i-2]
print(a[h])
"""



"""
черепашка
N,M=map(int,input().split())
z=[]
COINS=[]
for a in range (N):
  z=list(map(int,input().split()))
  COINS.append(z)
dp = [[None] * M for i in range(N)]
prev = [[None] * M for i in range(N)]

for i in range(N):
    for j in range(M):
        if i == 0 and j == 0:
            dp[0][0] = COINS[0][0]
            prev[0][0] = -1
        elif i == 0:
            dp[0][j] = dp[0][j - 1] + COINS[0][j]
            prev[0][j] = 0
        elif j == 0:
            dp[i][0] = dp[i - 1][0] + COINS[i][0]
            prev[i][0] = 1
        else:
            dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + COINS[i][j]
            if dp[i - 1][j] > dp[i][j - 1]:
                prev[i][j] = 1
            else:
                prev[i][j] = 0

print(max(max(dp)))
i, j = N - 1, M - 1
answer = []
answer_directions = []
while i > 0 or j > 0:
    if prev[i][j] == 1:
        i -= 1
        answer_directions.append('D')
    else:
        j -= 1
        answer_directions.append('R')
    answer.append((i, j))

print (*answer_directions[::-1])
"""
"""
шашки
a=[[0] * 8 for i in range(8)]

n,k=map(int,input().split())

for i in range(8):
    a[0][i]=1
for i in range (1,8):
    for j in range (0,8):
        if ((j-1)>=0) and (j+1<8):
            a[i][j]=a[i-1][j-1]+a[i-1][j+1]
        elif j-1<0:
            a[i][j]=a[i-1][j+1]
        elif j+1>=8:
            a[i][j]=a[i-1][j-1]

a.reverse()

print(a[k-1][n-1])

""" 

"""
взрывоопасность
n=int(input())
if n==1:
    print('3')
elif n==2:
    print('8')
else:
    d= [[0] * 200 for i in range(200)]
    s=0
 
    d[0][0]=d[0][1]=d[0][2]=1
    for i in range(1,n):
        d[i][0]=d[i-1][1]+d[i-1][2]
        d[i][2]=d[i][1]=d[i-1][0]+d[i-1][1]+d[i-1][2]
    for i in range(n):
        s+=d[n-1][i]
    print(s)
"""
"""


lh = dict([(item, None) for item in t]).keys()-УДАЛЯЕТ ПОВТОРЫ
дещмань
n, m = map(int, input().split())
l= [[int(it) for it in input().split()] for _ in range(n)]
for c in range(m - 2, -1, -1):
    l[-1][c] += l[-1][c + 1]
for r in range(n - 2, -1, -1):
    l[r][-1] += l[r + 1][-1]
for r in range(n - 2, -1, -1):
    for c in range(m - 2, -1, -1):
        l[r][c] += min(l[r + 1][c], l[r][c + 1])
print(l[0][0])
"""





"""
рисование графиков в python
import matplotlib as mpl                        
mpl.use('Agg')  # Не рисовать на экране
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 2, 100) # от -5 до 2 сделать 100 точек
y1 = x**3 + 5*x**2 + 10 # y1 - тоже много точек


fig, ax = plt.subplots()   # будет 1 график, на нем:
ax.plot(x, y1, color="blue", label="y(x)") # функция y1(x), синий, надпись y(x)

ax.plot(x, y2, color="red", label="y'(x)") # функция y2(x), красный, надпись y'(x)
ax.plot(x, y3, color="green", label="y''(x)") # функция y3(x), зеленый, надпись y''(x)


ax.set_xlabel("x")  # подпись у горизонтальной оси х
ax.set_ylabel("y")   # подпись у вертикальной оси y

ax.legend()     # показывать условные обозначения

#plt.show()  # показать рисунок - не нужно
fig.savefig('1.png')    

"""




"""
коняшка
dp=[[0] * 55 for i in range(55)]
dp[0][0] = 1
n,m=map(int,input().split())
for i in range(1,n):
    for j in range(1,m):
        dp[i][j] = dp[i-1][j-2] + dp[i-2][j-1]+
print(dp[n-1][m-1])
"""




"""
коняшка 2
n, m = map(int, input().split())
dp = [[0 for j in range(m + 2)] for i in range(n + 2)]
dp[0][0] = 1
for j in range(m):
    for i in range(n):
        if j>=0:
            dp[i][j] += dp[i-2][j-1]+dp[i-1][j-2]+dp[i-2][j+1]+dp[i+1][j-2]
            i=i+1
            j=j-1
for i in range(1, n):
    j=m-1
    while i<n and j>=0:
        dp[i][j] += dp[i-2][j-1]+dp[i-1][j-2]+dp[i-2][j+1]+dp[i+1][j-2]
        i=i+1
        j=j-1
print(dp[n - 1][m - 1])
"""

"""
чтение из файла
f = open('cin.txt')
h=[]
line = f.readline()
while line:
    a=line
    line = f.readline()
    b = a.split(' ')[::1]
    c = list(map(lambda x: int(x), b)); h.append(c)
print(*h)
"""

"""
n=int(input())
final=0
savm=-1
savn=-1
sumw=0
points=[]
for i in range(n):
  st,end=map(int,input().split())
  points.append(tuple((st,1)))
  points.append(tuple((end,-1)))
points.sort()
for i in range(1,2*n):
  final+=points[i-1][1]
  if final>0:
    sumw+=abs(points[i][0]-points[i-1][0])
print(sumw)
"""
"""
n=int(input())
a=list(map(int,input.split()))

if n in a :
  print('Yes')
else:
  print('No')

"""
"""
import random
def prz(p):
  z=1
  for i in range(len(p)):
    z=p[i]*z
  return z

p=[random.randint(1,100) for i in range(1200)]
print(prz(p))
"""
"""
n=int(input())
p=[]
d=[]
for i in range(n):
  a,b=map(int,input().split()) 
  p.append(tuple((a,b)))
 
p=sorted(p, key=lambda x:(x[0],x[1]),reverse=True )
a=[]
for i in range(len(p)): 
  if p[i][0]==0:
    a.append(p[i])
for i in range(len(p)): 
  if p[i][0]!=0:
    a.append(p[i])

maxi=0
for i in range(1,len(a)):
  if abs(a[i][1]-a[i-1][1])>maxi:
    maxi=abs(a[i][1]-a[i-1][1])
print(maxi)
"""
"""
result = 0
balance = 0
N = int( input())
times = []
timeline = []
for i in range( N):
 times. append( list( map( int, input(). split())))
 time_open = times[ i][: 3]
 time_close = times[ i][ 3:]
 time_open = time_open[ 0] * 3600 + time_open[ 1] * 60 + time_open[ 2]
 time_close = time_close[ 0] * 3600 + time_close[ 1] * 60 + time_close[ 2]
 if time_open == time_close:
  balance += 1
 elif time_open > time_close:
  timeline. append( tuple(( 0, 0)))
  timeline. append( tuple(( time_close, 1)))
  timeline. append( tuple(( time_open, 0)))
  timeline. append( tuple(( 86400, 1)))
 else:
  timeline. append( tuple(( time_open, 0)))
  timeline. append( tuple(( time_close, 1)))
if balance == N:
 print( 86400)
 exit()
timeline. sort()
for i in range( len( timeline) - 1):
 if timeline[ i][ 1] == 0:
  balance += 1
 else:
  balance -= 1
 if balance == N:
  result += ( timeline[ i+1][ 0] - timeline[ i][ 0])
print( result)
"""
"""

n=int(input())
final=0
savm=-1
savn=-1
sumw=0
points=[]
maxy=10**9
for i in range(n):
  st,end=map(int,input().split())
  points.append(tuple(((st,1,i))))
  points.append(tuple(((end,-1,i))))
points.sort()
for i in range(2*n):
 
  
  final+=points[i][1]
  
  if final<maxy:
    
print(sumw)
"""


"""
правильная последовательность
a=[]
l=list(input())
for i in range (len(l)):
  if l[i]=='(' or l[i]=='[' or l[i]=='{':
    a.append(l[i])
  else:
    if len(a)==0:
      a.append(1)
      break
    else:
      if l[i]==')' :
        p=a.pop(len(a)-1)
        if p!='(':
          a.append(1)
          break
      elif l[i]==']':
        p=a.pop(len(a)-1)
        if p!='[':
          a.append(1)
          break
      elif l[i]=='}':
        p=a.pop(len(a)-1)
        if p!='{':
          a.append(1)
          break
      
  
if len(a)==0:
  print('yes')
  
else:
  print('no')
"""



"""

польская запись
def is_digit(string):
    if string.isdigit():
       return True
    else:
        try:
            float(string)
            return True
        except ValueError:
            return False





a=list(input().split())
b=[]
for i in range(len(a)):
  if a[i]!='+' and a[i]!='-' and a[i]!='*':
    b.append(a[i])
  else:
    if a[i]=='+':
      
      p=int(b[len(b)-1])+int(b[len(b)-2])
      b.pop(len(b)-1)
      b.pop(len(b)-1)
      b.append(p)
    
        
      
    elif a[i]=='*':
      p=int(b[len(b)-1])*int(b[len(b)-2])
      b.pop(len(b)-1)
      b.pop(len(b)-1)
      b.append(p)
    else:
      p=int(b[len(b)-2])-int(b[len(b)-1])
      b.pop(len(b)-1)
      b.pop(len(b)-1)
      b.append(p)
    
print(*b)
"""



"""
from collections import Counter
a=list(input())
b=list(input())
c=list(input())
d=list((Counter(a) & Counter(b) &Counter(c)).elements())
if len(d)==len(a):
  d=set(d)
  d.sort()
for i in d:
  print(i,end="")
"""



"""
ханойские башни
def hanoi(n, frome, to, end):
    if n==0:
        return
    hanoi(n - 1, frome, end, to)
    print(n,frome, to)
    hanoi(n - 1, end, to, frome)
n = int(input())
hanoi(n, 1, 3, 2)
"""

  
"""
b.append(a[0])
for i in range(1,len(a)):
  if a[i]>b[len(b)-1]:
    b.append(a[i])
  else:
    p=0
    while len(b)-1-p>0 and a[i]<=b[len(b)-1-p] :
      k=b[len(b)-1-p]*(p+1)
      p=p+1
      d.append(k)
 
    n=(p+1)*a[i]
    d.append(n)
    b.clear()
    b.append(a[i])
print(max(d))

"""
"""

h=list(map(int,input().split()))
l=[0]*(len(h)+1)
r=[0]*(len(h)+1)
d=[]
t=[]
t.append(h[0])
for i in range(len(h)):
  if t[len(t)-1]<h[i]:
    t.append(h[i])
    c=1
    n=len(t)-1
    while n>=0:
      b=t[n]*c
      c=c+1
      n=n-1
      d.append(b)
   
  else:
    p=0
    while (len(t)-1-p)>0 and t[len(t)-1-p]>=h[i]:
      p=p+1
     
     
    if t[len(t)-1-p]>=h[i]:
      l[i]=-1
      r[i]=i
    else:
      l[i]=len(t)-1-p
      r[i]=l[i]
    t.clear()
    t.append(h[i])
    y=h[i]*(r[i]-l[i])
  d.append(y)
print(max(d))

"""
"""
a = []
p = [0] * 2
z = []
n, m = map(int, input().split())
for i in range(m):
    b = list(map(int, input().split()))
    a.append(b)

for i in range(n):
    for j in range(n):
        p[0] = i+1
        p[1] = j+1
        z.append(p)
        if z[0] in a:
            print(1, end="")
        else:
            print(0, end="")
        z.clear()
    print()

"""
"""
a=[]
n,m=map(int,input().split())
for i in range(m):
  p,l=map(int,input().split())
  a.append(p)
  a.append(l)
for j in range(n):
  print(a.count(j+1))
"""
"""
Обход графа
def pol(v):
    p.append(v)
    for i in range(n):
        if a[v][i] == 1 and i not in p:
            pol(i)
n, m = map(int, input().split())
a=[list(map(int, input().split())) for i in range(n)]
p=[]
pol(m-1)
print(len(p))

"""

"""
Дерево реализация
import datetime


class TreeNode(object):
    def __init__(self, index, value):
        self.index = index
        self.children = []
        self.value = value

    def __repr__(self):
        return str(self.index)

    def add_child(self, node):
        self.children.append(node)

    def get_children(self):
        return self.children

    def get_rev_children(self):
        children = self.children[:]
        children.reverse()
        return children


def DFS(root):
    nodes = []
    stack = [root]

    min_node = stack[0]
    max_node = stack[0]

    while stack:
        current_node = stack[0]
        print("Visiting node", str(current_node), " with value", current_node.value)
        if current_node.value < min_node.value:
            min_node = current_node
        if current_node.value > max_node.value:
            max_node = current_node

        stack = stack[1:]  # выкидываем первый элемент
        nodes.append(current_node)

        stack = current_node.get_children() + stack

    print('\nThe min value is: ', min_node.value, 'of the node: ', str(min_node))
    print('The max value is: ', max_node.value, 'of the node: ', str(max_node))

    return nodes


def get_example_tree():
    # create nodes
    root = TreeNode("a0", 12323)

    b0 = TreeNode("b0", 13224)
    b1 = TreeNode("b1", 3456)
    b2 = TreeNode("b2", 2134)

    c0 = TreeNode("c0", 42345)
    c1 = TreeNode("c1", 522)
    c2 = TreeNode("c2", 624123)

    d0 = TreeNode("d0", 6243)
    d1 = TreeNode("d1", 62143)

    e0 = TreeNode("e0", 6143)

    # add nodes
    root.add_child(b0)
    root.add_child(b1)
    root.add_child(b2)

    b0.add_child(c0)
    b0.add_child(c1)

    b1.add_child(c2)

    c0.add_child(d0)

    c2.add_child(d1)

    d1.add_child(e0)

    return root


if __name__ == "__main__":

    root = get_example_tree()  # the tree

    print("\n------------------------- DFS -------------------------\n")
    start = datetime.datetime.now()
    DFS(root)
    done = datetime.datetime.now()
    elapsed = done - start
    print("\nFinished in ", elapsed.microseconds, " microseconds")
"""
"""
a = []
p = [0] * 2
z = []
n, m = map(int, input().split())
for i in range(m):
    b = list(map(int, input().split()))
    a.append(b)
 
for i in range(n):
    for j in range(n):
        p[0] = i+1
        p[1] = j+1
        z.append(p)
        if z[0] in a:
            print(1, end=" ")
        else:
            print(0, end=" ")
        z.clear()
    print()
"""
"""
 
import telebot
TOKEN = "642369214:AAF-K5ZfwuX_svEjU9amLTzgdGQGrJzya10"
bot = telebot.TeleBot(TOKEN)
n=1
while n!=0:
  bot.send_message(chat_id=690187902,text=n)
  n=n-1


"""
"""


def pol(v):
    p.append(v)
    for i in range(n):
        if a[v][i] == 1 and i not in p:
            pol(i)
n, m = map(int, input().split())
a = [[0] * n for i in range(n)]
for it in range(m):
    b, k = map(int, input().split())
    a[b - 1][k - 1] = a[k-1][b-1]=1

p=[]
pol(1)
z=len(p)
aj=[x+1 for x in p]
print(len(aj))
print(*aj)
i=max(p)
while z!=n:
  p.clear()
  pol(i+1)
  z=z+len(p)
  aj=[x+1 for x in p]
  print(len(aj))
  print(*aj)
  i=max(p)
"""
"""
n,m,k=map(int,input().split())
a=[0]*n
for i in range(1,n+1):
  a[i-1]=i
b=a[m-1:k]
c=a[k-1:n]
c1=a[0:m]
i=len(c)+len(c1)-2
i1=len(b)-2
if i>i1:
  print(i1)
else:
  print(i)
"""
"""
Дерево?
n, m = map(int, input().split())
a = [[0] * n for i in range(n)]
for it in range(m):
    b, k = map(int, input().split())
    a[b - 1][k - 1] = a[k-1][b-1]=1
visited = [False] * n
def DFS(v):
    visited[v] = 1
    for i in range(n):
        if a[v][i] == 1 and not visited[i]:
            DFS(i)
 
 
Edges=0
for i in range(n):
    for j in range(i):
        Edges+=a[i][j]
DFS(0)
if Edges!=n-1 or (False in visited):
    print('NO')
else:
    print('YES')
"""
"""
Баобаб
n=int(input())
a=[]
visited=[False]*n
for i in range(n):
    a.append([int(x) for x in input().split()])
def DFS(v):
    visited[v]=1
    for i in range(n):
        if a[v][i]==1 and not visited[i]:
            DFS(i)
Edges=0
for i in range(n):
    for j in range(i):
        Edges+=a[i][j]
DFS(0)
if Edges!=n-1 or (False in visited):
    print('NO')
else:
    print('YES')
"""
"""
Получи дерево
n, m = map(int, input().split())
a = [[0] * n for i in range(n)]
vg=[list(map(int,input().split())) for i in range(m)]
for i in vg: 
    a[i[0] - 1][i[1] - 1] =a[i[1] - 1][i[0] - 1] =1
visited = [False] * n
def DFS(v):
    visited[v] = 1 
    for i in range(n):
        if a[v][i] == 1 and visited[i]!=1:
            print(v+1,i+1)
            DFS(i)
DFS(0)
"""
"""
гисторгаммма
from typing import NamedTuple
 
 
class Node(NamedTuple):
    index: int
    height: int
def get_biggest_square(array):
    array.append(0)
    stack = [Node(0, -1)]
    result = 0 
    for i in range(0, len(array)):
        previous_index = i
        while array[i] <= stack[-1].height:
            previous_index = stack[-1].index
            previous_height = stack[-1].height
            stack.pop()
            area = previous_height * (i - previous_index)
 
            if area > result:
                result = area 
        stack.append(Node(previous_index, array[i])) 
    return result
input_array = list(map(int, input().split()))[1:]
print(get_biggest_square(input_array))
"""
"""
BFs пути
def bfspath(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        vertex, path = queue.pop(0)
        for current_vertex in graph[vertex] - set(path):
            if current_vertex == goal:
                yield path + [current_vertex]
            else:
                queue.append((current_vertex, path + [current_vertex]))
graph = {}
n = int(input())
for i in range(n):
    b = []
    for j, l in enumerate(map(int, input().split())):
        if l == 1:
            b.append(j + 1)
    graph[i + 1] = set(b)
k, h = map(int, input().split())
if h == k:
    print(0)
else:
    a = (bfspath(graph, k, h))
    try:
      a = next(a)
      print(len(a) - 1)
      print(*a)
    except:
      print(-1)
"""
"""
import json
from random import choice

def gen_person():
    name=''
    tell=''
    letters=['a','b','c','d','e','f','g']
    nums=['1','2','3','4','5','6','7']
    while len(name)!=5:
        name+=choice(letters)
    while len(tell)!=7:
        tell+=choice(nums)
    person={
        'name':name,
        'tell':tell
    }
    return person
def main():
    persons=[gen_person() for i in range(10)]
    with open('persons.json','w') as file:
        json.dump(persons,file,indent=2,ensure_ascii=False)

main()


"""

"""
веб на низком 
import socket
from views import *

URLS={
    '/':index,
    '/blog':blog
}

def parse_request(request):
    parsed=request.split(' ')
    print(parsed)
    method=parsed[0]
    url=parsed[1]

    return method,url
def generate_headers(method,url):
    if not method=='GET':
        return ('HTTP/1.1 405 MEthod not allowed\n\n',405)
    if not url in URLS:
        return ('HTTP/1.1 404 Not found\n\n',404)
    return ('HTTP/1.1 200 OK\n\n',200)
def generate_content(code,url):
    if code==404:
        return '<h1>404</h1><p>Not found</p> '
    if code ==405:
        return '<h1>405</h1><p>Method not allowed</p> '
    return URLS[url]()


def generate_response(request):
    method, url = parse_request(request)
    headers,code=generate_headers(method,url)
    body=generate_content(code,url)


    return (headers+body).encode()

def run():
    server_socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    server_socket.bind(('localhost',5000))
    server_socket.listen()
    while True:
        client_socket,addr =server_socket.accept()
        request=client_socket.recv(1024)
        #print(request.decode('utf-8'))
        #print(request)
        #print()
        #print(addr)

        response=generate_response(request.decode('utf-8'))

        client_socket.sendall(response)
        client_socket.close()


if __name__=='__main__':
    run()
"""

"""
тупенькая реализация простых чисел
"""
"""
import math
def get_primes(n):
  numbers = [None for i in range(n+1)]
  for i in range(2, int(math.sqrt(n))+1):
    if numbers[i]==None:
      for j in range(i*i, n+1, i):
        numbers[j] = 0
  return numbers
l=0
n,m=map(int,input().split())
a=(get_primes(m))
for i in range(len(a)):
    if i>=n and i<=m and a[i]==None :
        print(i)
        l=1
if l==0:
    print('Absent')
"""
"""
class Primelist:
    __slots__ = ['name']
    def __init__(self):
        self.name=set()
    def create(self,n):
        import math
        numbers = [True for i in range(n+1)]
        for i in range(2, int(math.sqrt(n)) + 1):
            if numbers[i]:
                    for j in range(i * i, n+1, i):
                        numbers[j] = False
        for i in range(2,n+1):
            if numbers[i]:
                self.name.add(i)
        return self.name
    def inspect(self,x):
        a=self.name
        b=len(a)
        a.add(x)
        if len(a)>b:
            return False
        return True
    def get_simple(self,Sorted):
        if self.name:
            p=list(self.name)
            if Sorted:
                p=sorted(p)
            return p
if __name__=='__main__':
    a=Primelist()
    m=int(input())
    a.create(m)
    print(1)

from dataclasses import dataclass, field
from typing import *
@dataclass
class Press:
    n:int
    name: Set[int] = field(default_factory=set)
    def create(self):
        n=self.n
        import math
        numbers = [True for i in range(n+1)]
        for i in range(2, int(math.sqrt(n)) + 1):
            if numbers[i]:
                    for j in range(i * i, n+1, i):
                        numbers[j] = False
        for i in range(2,n+1):
            if numbers[i]:
                self.name.add(i)
c = Press(101)

print(c)
"""
"""
связаный список
class Node:
    def __init__(self, value: int, next: 'Node'=None):
        self.value = value
        self.next = next
class LinkedList:
    def __init__(self):
        self.head = None
"""
"""
    def __str__(self):
        if self.head:
            current = self.head
            p=[]
            p.append(current.value)
            while current.next != None:
                current = current.next
                p.append(current.value)
      return 'LinkedList' +' '+str(p)
"""
"""
    def insert_before_head(self, newdata):
        new = Node(newdata)
        # Update the new nodes next val to existing node
        new.head = self.head
        self.head = new
    def add(self, x):
        if not self.head :
            self.head=Node(x, None)
        else:
            last = self.head
            while last.next != None:
                last = last.next
            last.next=Node(x,None)
    def get_simple(self):
        if self.head:
            current = self.head
            p=[]
            p.append(current.value)
            while current.next != None:
                current = current.next
                p.append(current.value)
        return p
linked = LinkedList()
n=int(input())
linked.head = Node(input())
for i in range(n-1):
    m=input()
    linked.add(m)
print(*linked.get_simple())


def reverse(cursor):
    reversed_list = None
    while cursor:
        temp = reversed_list
        next_item = cursor.next
        reversed_list = cursor
        reversed_list.next = temp
        cursor = next_item

    return reversed_list

head = Node(1, Node(2, Node(3)))
"""

"""
киви
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout

from kivy.config  import Config
Config.set('graphics','resizable',0)
Config.set('graphics','width',400)
Config.set('graphics','height',500)
class CalculatorApp(App):
    def update_label(self):
        self.lbl.text=self.formula


    def add_number(self,instance):
        if (self.formula=="0"):
            self.formula=""
        self.formula+=str(instance.text)
        self.update_label()
    def add_operation(self,instance):
        if str(instance.text).lower()=="x":
            self.formula+="*"
        else:
            self.formula+=str(instance.text)
        self.update_label()
    def calc_result(self):
        self.lbl.text=str(eval(self.lbl.text))
        self.formula="0"
    def build(self):
        self.formula=""
        bl=BoxLayout(orientation='vertical',padding=25)
        gl=GridLayout(cols=4,spacing=3,size_hint=(1,.6))


        self.lbl = Label(text="0",font_size=40,halign="right",valign="center",size_hint=(1,.4),text_size=(400-50,500*.4-50))

        bl.add_widget(self.lbl)

        gl.add_widget(Button(text="7",on_press=self.add_number))
        gl.add_widget(Button(text="8",on_press=self.add_number))
        gl.add_widget(Button(text="9",on_press=self.add_number))
        gl.add_widget(Button(text="x",on_press=self.add_operation))

        gl.add_widget(Button(text="4",on_press=self.add_number))
        gl.add_widget(Button(text="5",on_press=self.add_number))
        gl.add_widget(Button(text="6",on_press=self.add_number))
        gl.add_widget(Button(text="-",on_press=self.add_operation))

        gl.add_widget(Button(text="1",on_press=self.add_number))
        gl.add_widget(Button(text="2",on_press=self.add_number))
        gl.add_widget(Button(text="3",on_press=self.add_number))
        gl.add_widget(Button(text="+",on_press=self.add_operation))

        gl.add_widget(Widget())
        gl.add_widget(Button(text="0",on_press=self.add_number))
        gl.add_widget(Button(text=".",on_press=self.add_number))
        gl.add_widget(Button(text="=",on_press=self.calc_result))
        bl.add_widget(gl)
        return bl

if __name__=='__main__':
    CalculatorApp().run()
"""
"""
лифтовс
a=list(map(int,input()))
l=250
p=[0]*500
p[l]=1
for i in a:
    if i==1:
        l+=1
        p[l]=1
    else:
        l+=-1
        p[l]=1
print(p.count(1))
"""
"""
Домашнее задание
n=int(input())
a=list(map(int,input().split()))
b=[x for x in a if x>0]
b=sum(b)
p=a.index(min(a))
p1=a.index(max(a))
if p<p1:
    k=a[p+1:p1]
else:
    k=a[p1+1:p]
p=1
for i in k:
    p=p*i
print(b,p)
"""

"""
Дороги
n=int(input())
a=[list(map(int,input().split())) for i in range(n)]
d=0
for i in range(n):
    for j in range(n):
        if a[i][j]==1:
            d+=1
print(d//2)
"""
"""
Постулат Бертрана
import math
def get_primes(n,k):
  m = n+1
  numbers = [True for i in range(m)]
  for i in range(2, int(math.sqrt(n))+1):
    if numbers[i]:
      for j in range(i*i, m, i):
        numbers[j] = False
  primes = []
  for i in range(2, n):  
    if numbers[i] and i>k:
      primes.append(i)
  return primes
k=int(input())
print(len(get_primes(2*k,k)))


Пересечение множеств
c=list(map(int,input().split()))
a=list(map(int,input().split()))
b=list(map(int,input().split()))
a=set(a)
b=set(b)
b=list(b&a)
b=sorted(b)
print(*b)

n=int(input())
b=[]
for i in range(n):
    a=input()
 
    if a.count('1')==a.count('2')==a.count('a0'):
        if a[0:a.count('0')].count('1')!=0:
            b.append('NO')
        elif a[a.count('0'):a.count('0')+a.count('1')].count('2'):
            b.append('NO')
        else:
            b.append('YES')
    else:
        b.append('NO')
print('\n'.join(b))

технокубок 4.1
a=list(input())
s=[]
j=0
if len(a)%2==0:
    while len(a)!=0:
        if j%2==0:
            b = len(a) // 2
            n=a.pop(b-1)
            s.append(n)
        else:
            b = len(a) // 2
            n = a.pop(b)
            s.append(n)
        j=j+1
else:
    while len(a)!=0:
        b = len(a) // 2
        n=a.pop(b)
        s.append(n)
print(*s, sep='')
"""

"""
отгадай число
import sys
left=1
right=10**6
while right!=left:
  middle=(right+left+1)//2
  print(middle)
  a=input()
  sys.stdout.flush()
  if a!='>=':
    right=middle-1
  else:
    left=middle
print('! ',left)
"""
"""
def resare(a):
  a1=a[0:n//2]
  a2=a[n//2+1:len(n)-1]
  return max(a1,a2)
a=list(map(int,input().split()))
print(resare(a))
"""
'''def bfs(g, start):
    visited, queue = set(), [start]
    while queue:
        vertex=queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            print(vertex)
            neighbour_vertex = set(g[vertex])
            queue += list(neighbour_vertex - visited)
    return visited
g = [[1, 2],
     [0, 3, 4],
     [0, 5],
     [1],
     [1, 5],
     [2, 4]]
bfs(g, 0)
def bfs_path(g, start, goal):
    vertex = start
    path = [start]
    queue = [(vertex, path)]
    while queue:
        (vertex, path) = queue.pop(0)
        for current_vertex in list( set(g[vertex]) - set(path)):
            if current_vertex == goal:
                yield path + [current_vertex]
            else:
                queue.append(current_vertex, path)
bfs_path(g, 0, 3)'''

'''
def bfs(input_g: dict, start_vertex: int):
    n = len(g)
    dist = [n] * n
    dist[start_vertex] = 0
    queue = [start_vertex]
    while queue:
        current_vertex = queue.pop(0)
        for i in input_g[current_vertex]:
            if dist[i] > dist[current_vertex] + 1:
                dist[i] = dist[current_vertex] + 1
                print(i)
                queue.append(i)
    return dist

print(bfs(g,0))

def bfs_path(inp_g, start_vertex, goal_vertex):
    n = len(g)
    dist = [n] * n
    dist[start_vertex] = 0
    queue = [start_vertex]
    parents = [-1] * n
    while queue:
        current_vertex = queue.pop(0)
        for i in inp_g[current_vertex]:
            if dist[i] > dist[current_vertex] + 1:
                parents[i] = current_vertex
                dist[i] = dist[current_vertex] + 1
                queue.append(i)
    if dist[goal_vertex] == n:
        return []

    paths = []
    vertex = goal_vertex
    while vertex != -1:
        paths.insert(0, vertex)
        vertex = parents[vertex]
    print(paths)
bfs_path(g, 0, 5)

def dijkstra(g, start_vertex):
    n = len(g)
    dist = [inf] * n
    dist[start_vertex] = 0
    visited = []
    while True:
        available_vertex = list(set(g.keys()) - set(visited))
        if not available_vertex:
            break

        sorted_vertex = sorted([(i, dist[i]) for i in available_vertex], key = lambda x: x[1])
        current_vertex = sorted_vertex[0][0]

        for pair in g[current_vertex]:
            destination_vertex = pair.destination_vertex
            weight = pair.weight
            if dist[destination_vertex] > dist[current_vertex] + weight:
                dist[destination_vertex] = dist[current_vertex] + weight

        visited.append()
'''
"""

t = False
def bfs_path(inp_g, start_vertex, goal_vertex):
    global t
    n = len(g)
    dist = [n] * n
    dist[start_vertex] = 0
    queue = [start_vertex]
    parents = [-1] * n
    while queue:
        current_vertex = queue.pop(0)
        for i in inp_g[current_vertex]:
            if dist[i] > dist[current_vertex] + 1:
                parents[i] = current_vertex
                dist[i] = dist[current_vertex] + 1
                queue.append(i)
    if dist[goal_vertex] == n:
        return 0

    paths = []
    vertex = goal_vertex
    while vertex != -1:
        paths.insert(0, vertex)
        vertex = parents[vertex]
    print(len(paths) - 1)
    t = True
    if len(paths) != 1:
        for z in paths:
            print(z + 1, end=' ')

n = int(input())
g = [[] for i in range(n)]
for i in range(n):
  inp = input().split()
  for j in range(n):
    if inp[j] == '1':
      g[i].append(int(j))

start, goal = map(int, input().split())
bfs_path(g, start-1, goal-1)
if not t:
    print(-1, end='')
    import sys
sys.setrecursionlimit(100_002)

def graph_traversal(s):
    global N, exIt
    for i in matrix[s]:
        if not ways.get(((i, s if i < s else s, i)), False):
            points.add(i)
            ways[(i, s if i < s else s, i)] = True
            try:
                if all_points[i]:
                    exIt -= 1
                all_points[i] = False
            except: pass
            graph_traversal(i)




N, M = map(int, input().split())
matrix = [[] for i in range(N+1)]
all_points = [i for i in range(N+1)]
exIt = N
ways = {}
points = set()
answers = []
comps = 0
for i in range(M):
    a, b = map(int, input().split())
    matrix[a].append(b)
    matrix[b].append(a)
while exIt:
    print(all_points)
    points.add(all_points[1])
    exIt -= 1
    tmp = all_points[1]
    if not tmp: continue
    all_points[1] = False
    graph_traversal(tmp)
    answers.append(points)
    points = set()
    comps += 1
print(comps)
for i in answers:
    print(len(i))
    print(*i)
import sys

#sys.std.flush()

left = 0
right = 1000001
#print(8)
while right - left > 1:
    mid = (right + left)//2
    print(mid)
    sys.stdout.flush()
    inp = input()
    if inp == '>=':
        left = mid

    else:
        right = mid


print('!',left)    
import sys
simple = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
inp = 'no'
trigger = 0
i = 0
while trigger < 2 and i < 15:
    print(simple[i])
    sys.stdout.flush()
    inp = input()
    if inp == 'yes':
        trigger += 1
        if simple[i]<11:
            print(simple[i]**2)
            sys.stdout.flush()
            inp = input()
            if inp == 'yes':
                trigger+=1
    i += 1
if trigger > 1:
    print('composite')
else:
    print('prime')
sys.stdout.flush()
import sys
n = int(input())
arr = []
ans = []
for i in range(1,n):
    print('?',0,i)
    sys.stdout.flush()
    arr.append(int(input()))
print('?',1,2)
sys.stdout.flush()
xer = int(input())
ans.append((arr[0]+arr[1]-xer)//2)
for i in arr:
    ans.append(i-ans[0])
print('!',*ans)
0 23 1 0
"""
"""
a,x,b,y=map(int,input().split())
if a==0:
  a=24
m=a+b
if x<=y:  
  if m+(y-x)>=24:
    print((m+(y-x))%24)    
  else:    
    print(m+(y-x))
else:
  if (m-(x-y))>=24:
    print((m-(x-y))%24)
  elif (m-(x-y))<0:
    print(m-(x-y)+24)
  else:
    print(m-(x-y))
"""
"""

a=int(input())
a1=a-1
a2=a+1
a3=a+10000
a4=a-10000
b=[]
b.extend([a1,a2,a3,a4])
if a1<0:
  b.remove(a1)
if a2>10**8:
  b.remove(a2)
if a3>10**8:
  b.remove(a3)
if a4<0:
  b.remove(a4)
b=sorted(b)
print(*b)
"""
"""
import requests
from bs4 import BeautifulSoup as bs
headers={'accept':'*/*',
         'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 YaBrowser/18.11.1.805 Yowser/2.5 Safari/537.36'}
base_url='https://hh.ru/search/vacancy?area=1&search_period=3&text=python&page=0'
def hh_parse(base_url,headers):
    jobs=[]
    session=requests.Session()
    request=session.get(base_url,headers=headers)
    if request.status_code==200:
        soup=bs(request.content,'html.parser')
        divs = soup.find_all('div', attrs={'data-qa': 'vacancy-serp__vacancy'})
        for div in divs:
            title=div.find('a',attrs={'data-qa':'vacancy-serp__vacancy-title'}).text
            href= div.find('a', attrs={'data-qa': 'vacancy-serp__vacancy-title'})['href']
            company = div.find('a', attrs={'data-qa': 'vacancy-serp__vacancy-employer'}).text
            text1=div.find('div', attrs={'data-qa': 'vacancy-serp__vacancy_snippet_responsibility'}).text
            text2=div.find('div', attrs={'data-qa': 'vacancy-serp__vacancy_snippet_requirement'}).text
            content=text1+' '+text2
            jobs.append({
                'title':title,
                'href':href,
                'company':company,
                'content':content
            })
            print(jobs)
    else:
        print('Error')
hh_parse(base_url,headers)
"""

"""
n,k=map(int,input().split())
a=list(map(int,input().split()))
ans=0
right=1
for left in range(n):
  while right!=n and a[right]-a[left]<=k:
    right+=1
  ans+=n-right
print(ans)
"""
"""
number=4
ans=0
result=100001
arrays=[]
N=[]

for i in range(number):
  N.append(int(input()))
  sorted_colors=sorted(list(map(int,input().split())))
  arrays.append(sorted_colors)

pointers=[0]*number
while (pointers[0]<N[0]) and pointers[1]<N[1] and pointers[2]<N[2] and pointers[3]<N[3]:
  temp=[]
  for i in range(number):
    temp.append(arrays[i][pointers[i]])
  diff=max(temp)-min(temp)
  if diff<= result:
    result=diff
    ans=temp
  pointers[temp.index(min(temp))]+=1
print(ans)
"""
"""
i=int(input())
a=set(k*k for k in range(i+1))
b=set(k*k*k for k in range(i+1))

b=b|a
b=list(sorted(b))
print(b)
"""
"""
d=0
n=int(input())
a=list(map(int,input().split()))
for i in range(n):
  j=0
  while j+i<=n:
    p=sum(a[j:j+i])
    if p>d:
      d=p
    j+=1
print(d)
"""
"""







"""
"""
import heapq
a=list(map(int,input().split()))
heapq.heapify(a)
for i in range(len(a)):
  print(heapq.heappop(a))

"""
"""

def qsort(a):
  less=[]
  equal=[]
  greater=[]
  if (len(a)<2):
    return a
  pivot=a[0]
  for x in a:
    if x<pivot:
      less.append(x)
    if x==pivot:
      equal.append(x)
    if x>pivot:
      greater.append(x)
  return qsort(less)+equal+qsort(greater)    
a=list(map(int,input().split()))
print(qsort(a))
"""
"""
import heapq

def otrs(a):
  for i in range(len(a)):
    a[i]=-a[i]
  return a
n=int(input())
a=list(map(int,input().split()))
a=otrs(a)
d=[]
d=dict()
n=int(input())
for i in range(n):
  b,j=map(int,input().split())
  a[b-1]=a[b-1]-j
  d[b-1]=a[b-1]
heapq.heapify(a)
for i in d.values():
  print(a.index(i)+1)
print(*(otrs(a)))
"""
"""
from math import sqrt,ceil,sqrt,floor
def rouns(a):
    if a-int(a)>=0.5:
        return ceil(a)
    else:
        return int(a)
def builds(a,b):
    c=[]
    while b!=0:
        c.append(rouns(a/b))
        a-=rouns(a/b)
        b-=1
    return c


n=int(input())
if n>=4:
  h=int(sqrt(n))
  a=(builds(n,h))
  a=sorted(a)[::-1]
  d=0
  
  for i in range(2,len(a)+1):
      p=0
      while p+i<=len(a):
          # for j in k:
      #
      #     d+=(a.count(j)-(i-1))*(j-(i-1))
          d+=min(a[p:p+i])-(i-1)
          p+=1
  print(d)
else:
  print(0)

"""
"""
def image(a,b):
  d=[]
  while a>=b:
    k=a//b
    d.append(a%b)
    a=k
  d.append(k)
  d=d[::-1]
  return d
d=(image(2**1024-2**128-2**0,2))
print(d.count(1))

    
  """
"""
n,k=map(int,input().split())
a=list(map(int,input().split()))
a=sorted(a)
c=0
left=0
right=n-1
d=[]
p=0
while left!=right:
  p+=1 
  cursum = a[left]*a[right]
  if (cursum<k):
    if cursum>c:
      d.append([left,right])
    left+=1
  elif cursum>k :
    right-=1
  else:
    d.append([left,right])
    break
print(d[len(d)-1])
print(p)
"""
import heapq
h=[1,3,-5,9,0]

l=heapq.heapify(x)

print(l)