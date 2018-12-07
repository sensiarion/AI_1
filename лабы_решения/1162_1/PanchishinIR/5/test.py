import numpy as np

a = np.arange(8).reshape((4, 2))
print(a)
b = np.arange(8).reshape((4, 2))
print(b)
c = np.hstack((a, b))
print(c)
np.random.shuffle(c)
print(c)
print(c[:3, :1]) #от 0 до 3 строк, от 0 до 1 столбцов
np.print_
