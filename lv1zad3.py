num_list=[]
print('Upišite brojeve!')
while True:
    num=input()
    if num.isdigit():
        num_list.append(float(num))
    else:
        if num=='Done':
            break
        else:
            print('Nije broj')
print('Broj elemenata:', len(num_list))
print('Srednja vrijednost:', sum(num_list)/len(num_list))
print('Minimalna vrijednost:', min(num_list))
print('Maksimalna vrijednost:', max(num_list))
num_list.sort()
print(num_list)