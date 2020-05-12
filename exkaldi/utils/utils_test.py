import utils

import subprocess
import os

def type_name_test():

    out = utils.type_name("Hello World")

    assert out == "str"

def run_shell_command_test():

    out, err, cod = utils.run_shell_command("echo Hello World", stdout=subprocess.PIPE)

    assert out.decode().strip() == "Hello World"
    assert err == None
    assert cod == 0

def make_dependent_dirs():

    dirName = "utilstest"
    if os.path.isdir(dirName):
        os.rmdir(dirName)
    utils.make_dependent_dirs(dirName, pathIsFile=False)

    assert os.path.isfile(dirName)
    os.rmdir(dirName)

def check_config_test():

    myconfig_0 = {'--sample-frequency': 8000}
    out = utils.check_config("compute_mfcc", myconfig_0)
    assert out is True

    myconfig_1 = {'--sample-frequency': 16000.0 }
    try:
        out = utils.check_config("compute_mfcc", myconfig_1)
    except Exception as e:
        assert utils.type_name(e) == "WrongDataFormat"

def compress_decompress_gz_file_test():

    if os.path.isfile("test.txt"):
        os.remove("test.txt")
    out, err, cod = utils.run_shell_command("echo Hello World > test.txt", stdout=subprocess.PIPE)

    outFile = utils.compress_gz_file("test.txt", overWrite=True)

    assert os.path.isfile("test.txt.gz") and os.path.getsize("test.txt.gz") > 0
    assert not os.path.isfile("test.txt")

    outFile = utils.decompress_gz_file("test.txt.gz", overWrite=True)
    
    assert os.path.isfile("test.txt") and os.path.getsize("test.txt") > 0
    assert not os.path.isfile("test.txt.gz")    

    os.remove("test.txt")

def flatten_test():

    inputs = [0,(1,2),[3],"456","7",np.array([[8,9],[10,11]])]
    results = utils.flatten( inputs )

    assert results == [0,1,2,3,"4","5","6","7",8,9,10,11]
