import lv1total as t

print('Unesite broj ranih sati!')
radni_sati=float(input())

print('Unesite plaću po radnom satu!')
palaca_po_radnom_satu=float(input())

print('Radni sati: ', radni_sati, 'h')
print('\neura/h: ', palaca_po_radnom_satu)
print('\nUkupno:', t.total_euro(radni_sati,palaca_po_radnom_satu), 'eura')