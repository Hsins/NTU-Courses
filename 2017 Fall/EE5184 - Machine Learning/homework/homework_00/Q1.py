import sys

inputFile = str(sys.argv[1])

file = open(inputFile, 'r')
words = file.read().split()
d = {}
for word in words:
    if word in d:
        continue
    else:
        d[word] = words.count(word)
newFile = open('Q1.txt', 'w')
i = 0
for key in d:
    if i == len(d) - 1:
        var = key + ' ' + str(i) + ' ' + str(d[key])
    else:
        var = key + ' ' + str(i) + ' ' + str(d[key]) + '\n'
    newFile.write(var)
    i += 1
newFile.close()
file.close()
