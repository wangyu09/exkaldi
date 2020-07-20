from __future__ import absolute_import

try:
    import pyaudio
except Exception as e:
    print("Cannot apply recording from microphone in this machine.")
    #raise e
    #return "Cannot apply recording from microphone in this machine."
    pass
else:
    from exkaldi.audio.audio import *