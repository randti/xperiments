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