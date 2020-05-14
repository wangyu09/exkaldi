import achivements

import os

def ListTable_test():

    t1 =  achivements.ListTable({1:"a", 3:"c", 2:"b"}, name="myTable")
    assert t1.is_void is False

    t2 = t1.sort(reverse=True)
    assert t2 == {3:"c", 2:"b", 1:"a"}

def Transcription_test():

    t = achivements.Transcription( {"utt-2":"b", "utt-1":"a" }, am_cost={"utt-2":-2, "utt-1":-1 })

    t1 = t.sort(reverse=False)
    for i, j in zip(t.sort(reverse=False).items(), {'utt-1': 'a', 'utt-2': 'b'}.items()):
        assert i == j

    assert t.am_cost("utt-1") == -1
    assert t.lm_cost("utt-1") is None

    if os.path.isfile("test.txt"):
        os.remove("test.txt")
    t.save("test.txt")
    assert os.path.isfile("test.txt")
    os.remove("test.txt")
