lista = [0,1,2,3,4,5,6,7,8,9]

new = 90
index = 9
new_lista = lista[:index+1] + [new] +lista[index+1:]
for i in range(len(lista)):
    print(i)
print(lista[11:])
print(new_lista)