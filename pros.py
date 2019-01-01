class Primelist:
    def __init__self(self,siml):
        self.siml=set()
    def add(self,x):
        self.siml.add(2)
        if x%2==0:
            return self.siml
        else:
            i=3
            while i*i<=x:
                if x%i==0:
                    return self.siml
                i+=2
            return self.siml.add(x)
    def get_simple(self):
        p=[]
        if self.siml:
            current = list(self.siml)
            p.append(current)
        return p
a=
