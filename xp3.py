a=[[i,0] for i in range(1,13)]
#b=int(input())
#for i in range(b):
    #d=int(input())
    #a[d-1][1]+=1
#a=list(filter(lambda x: x[1]!=0, sorted(a)))
#print('\n'.join(map(str, a)))

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
