
for i in range(1,11):

    fname = "model"+str(i)+".txt"
    fin = open(fname)
    lines = [line.replace(",", "") for line in fin]
    fin.close()

    print(lines[0])

    fname = "dt"+str(i)+"_clear.txt"
    fout = open(fname,"w")
    for line in lines:
        fout.write(line)
    fout.close()
