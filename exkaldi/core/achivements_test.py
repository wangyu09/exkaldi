import achivements

import os

def ScriptTable_test():

    data = { "utt-1": "test one",
            "utt-2": "test two",
            "utt-3": "test third"
            }
    t1 =  achivements.ScriptTable(data, name="myTable")
    assert t1.is_void is False

    dataRe = { "utt-3": "test third",
            "utt-2": "test two",
            "utt-1": "test one"
            }
    t2 = t1.sort(reverse=True)
    for i,j in zip(t2.items(), dataRe.items()):
        assert i == j
    
def BytesDataIndex_test():

    table = achivements.BytesDataIndex()

    data = { "utt-1": table.spec(10,0,100),
            "utt-2": table.spec(20,100,200),
            "utt-3": table.spec(30,300,300)
            }
    table.update(data)

    dataRe = { "utt-3": table.spec(30,300,300), 
            "utt-2": table.spec(20,100,200),
            "utt-1": table.spec(10,0,100),
            }

    for i,j in zip( table.sort("utt", reverse=True).items(), dataRe.items() ):
        assert i == j
    

    


