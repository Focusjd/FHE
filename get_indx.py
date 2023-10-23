
fname = "./tox21_data/ranged_data/tox21_ranged_out.txt"
fin = open(fname)
lines = [line.split(', ') for line in fin]

fin.close()

print(lines[0][1])
print(lines[1][1])

fname = "./tox21_data/ranged_data/tox21_indx.txt"
fout = open(fname,"w")
for line in lines:
    temp = line[1]
    temp = temp.replace('\n','')
    fout.write(temp + ', ') 
fout.close()
