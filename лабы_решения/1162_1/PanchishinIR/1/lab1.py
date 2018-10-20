2 + 3
4 * 5
5 - 1
40 / 2
2 ** 3

"Vanya"
"Hello from" + "Vanya"
"Vanya" * 3
"Vanya's string"
'Vanya\'s string again'
"Vanya".upper()
len("Vanya")
len(str(304023))
#int("Hello") ValueError: invalid literal for int() with base 10: 'Hello'

name = "Vanya"
name
name = "Ivan"
name
len(name)
a = 4
b = 6
a * b
city = "Khanty"
#ctiy NameError: name 'ctiy' is not defined

print(name)

# list
lottery = [3, 42, 12, 19, 30, 59]
len(lottery)
lottery.sort()
print(lottery)
lottery.reverse()
print(lottery)
lottery.append(199)
print(lottery[0])
print(lottery[1])
lottery.pop(0)
print(lottery)
#lottery.pop(6) IndexError: pop index out of range
#lottery.pop(7) IndexError: pop index out of range
#lottery.pop(1000) IndexError: pop index out of range
lottery.pop(-1) # с конца
#lottery.pop(-6) IndexError: pop index out of range
#lottery.pop(-1000) IndexError: pop index out of range

# hash
{}
participant = {'name': 'Vanya', 'country': 'Russia', 'favorite_numbers': [4, 1, 10]}
print(participant['name'])
#participant['age'] KeyError: 'age'
participant['favorite_language'] = 'C++'
len(participant)
participant.pop('favorite_numbers')
participant['country'] = 'France'

5 > 2
3 < 1
5 > 2 * 2
1 == 1
5 != 2
6 >= 12 / 2
3 <= 2
6 > 2 and 2 < 3
3 > 2 and 2 < 1
3 > 2 or 2 < 1
#1 > 'apple' TypeError: '>' not supported between instances of 'int' and 'str'

a = True
a
a = 2 > 5
a
True and True
False and True
True or 1 == 1
1 != 2

if 3 > 2: 
    print('It works!')

if 5 > 2:
    print('5 is indeed greater than 2')
else:
    print('5 is not greater than 2')

name = 'Vanya'
if name == 'Vanya':
    print('Hello Vanya!')
elif name == 'Vasya':
    print('Hi Vasya!')
else:
    print('I don\'t know who you are!')

def hi():
    print('Hi there!')
    print('How are you?')
hi()

def hi(name):
    if name == 'Vanya':
        print('Hello Vanya!')
    elif name == 'Vasya':
        print('Hi Vasya!')
    else:
        print('I don\'t know who you are!')
hi("Vanya")
hi("Petya")

def hi(name):
    print('Hi ' + name + '!')
hi("Alyosha")

girls = ['Ann', 'Alyouna', 'Vika', 'Ola']
for name in girls: hi(name); print('Next girl!')

for i in range(1, 6): print(i)
