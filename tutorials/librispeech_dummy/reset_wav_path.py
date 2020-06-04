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
        tempPath = line[-1].split("/")
        line[-1] =  os.path.join(root,"wav",tempPath[-1])
        lines[index] = " ".join(line)

    with open(wavScpFile, "w") as fw:
        fw.write("\n".join(lines))