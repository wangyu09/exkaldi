import os,glob

pwd = os.path.abspath(".")
dirName = os.path.dirname(pwd)
exkaldiRoot = os.path.join(dirName,"exkaldi") 

modules = glob.glob( os.path.join(dirName,"*.py"),
                     os.path.join(dirName,"*","*.py"),
)

funcName2index = {}

for path in modules:
  
  mName = os.path.basename(path)
  dirPath = os.path.dirname(path)

  if mName == "__init__.py" or dirPath == "config":
    continue

  with open(path, "r") as fr:

    lines = fr.readlines()

    

    for index,line in enumerate(lines):
      
      if line.startswith("class"):
        
        line = line.split(maxsplit=2)
        className = line[1].split("(",maxsplit=1)[0]
        funcName2index[className]


      line = line.strip().split()


     


  

openedFileHandles = {}

