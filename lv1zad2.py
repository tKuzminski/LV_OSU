print('Upišite ocjenu!')
try:
    ocjena = float(input())
except:
    print('Nije broj!')


if ocjena < 0 or ocjena > 1:
    print('Izvan intervala!')
elif ocjena >= 0.9:
    print('A')
elif ocjena >= 0.8:
    print('B')
elif ocjena >= 0.7:
    print('C')
elif ocjena >= 0.6:
    print('D')
elif ocjena < 0.6:
    print('F')