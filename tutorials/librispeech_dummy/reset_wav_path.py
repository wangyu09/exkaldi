import os

root = os.path.abspath(".")

for name in ["train","test"]:

    wavScpFile = os.path.join( name,"wav.scp" )

    with open(wavScpFile, "r", encoding="utf-8") as fr:
        lines = fr.readlines()

    for index,line in enumerate(lines):
        line = line.strip().split()
        if len(line) < 2:
            continue
        wavFileName = os.path.basename(line[-1])
        line[-1] =  os.path.join(root, "wav", wavFileName)
        lines[index] = " ".join(line)

    with open(wavScpFile, "w") as fw:
        fw.write("\n".join(lines))