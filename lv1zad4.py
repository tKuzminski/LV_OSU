words=dict()
song=open('song.txt')
for line in song:
    line = line.rstrip().lower().split()
    for lyric in line:
        lyric = lyric.strip().strip(',')
        if words.keys().__contains__(lyric):
            words[lyric]+=1
        else:
            words[lyric]=1
song.close()

unique_words = list(filter(lambda x: words[x] == 1, words))
print("Broj jedinstvenih rijeci:", len(unique_words))
print(unique_words)

