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
    persons=[gen_person() for i in range(1000)]
    with open('persons.json','w') as file:
        json.dump(persons,file,indent=2,ensure_ascii=False)
main()