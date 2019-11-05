
import pyaudio
import wave
import queue
import threading
import time
import socket
import math
import tempfile
import os

class WrongOperation(Exception):pass
class PathError(Exception):pass
class NetworkError(Exception):pass
    
## Not receive but can send
## not send but receive wait

class Client(object):

    def __init__(self):

        self.p = pyaudio.PyAudio()
        self.client = None

        self.threadManager = {}
        self._counter = 0

        self.dataQueue = queue.Queue()
        self.resultQueue = queue.Queue()
        
        self.localErrFlag = False
        self.remoteErrFlag = False
        self.endFlag = False
        self.safeFlag = False

        self.config_wave_format()

    def __enter__(self,*args):
        self.safeFlag = True
        return self
    
    def __exit__(self,errType,errValue,errTrace):
        if errType == KeyboardInterrupt:
            self.endFlag = True
        self.wait()
        if self.client != None:
            self.client.close()
        self.p.terminate()

        #self._counter = 0
        #self.dataQueue.queue.clear()
        #self.resultQueue.queue.clear()
        #self.client=None
        #for name in self.threadManager.keys():
        #   self.threadManager[name] = None
        #self.endFlag = False
        #self.errFlag = False
    
    def wait(self):
        for name,thread in self.threadManager.items():
            if thread.is_alive():
                thread.join()

    def close(self):
        self.endFlag = True
        self.__exit__(None,None,None)

    def config_wave_format(self,Format=None,Width=None,Channels=1,Rate=16000,ChunkFrames=1024):

        assert Channels==1 or Channels==2, "Expected <Channels> is 1 or 2 but got {}.".format(Channels)

        if Format != None:
            assert Format in ['int8','int16','int32'], "Expected <Format> is int8, int16 or int32 but got{}.".format(Format)
            assert Width == None, 'Only one of <Format> and <Width> is expected to be assigned but both two are gotten.'
            if Format == 'int8':
                self.width = 1
            elif Format == 'int16':
                self.width = 2
            else:
                self.width = 4           
        else:
            assert Width != None, 'Expected to assign one value of <Format> aor <Width> but got two None.'
            assert Width in [1,2,4], "Expected <Width> is 1, 2 or 4 but got{}.".format(Width)
            self.width = Width
            if Width == 1:
                self.formats = 'int8'
            elif Width == 2:
                self.formats = 'int16'
            else:
                self.formats = 'int32'
    
        self.formats = Format
        self.channels = Channels
        self.rate = Rate
        self.chunkFrames = ChunkFrames
        self.chunkSize = self.width*Channels*ChunkFrames

    def read(self,wavFile):

        if self.safeFlag == False:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')

        if not os.path.isfile(wavFile):
            raise PathError('No such wav file: {}.'.format(wavFile))
        
        if 'read' in self.threadManager.keys() and self.threadManager['read'].is_alive():
            raise WrongOperation('Another read task is running now.')

        if 'record' in self.threadManager.keys() and self.threadManager['record'].is_alive():
            raise WrongOperation('Record and Read are not allowed to run concurrently.')

        def readWave(wavFile,dataQueue):
            try:
                self._counter = 0

                wf = wave.open(wavFile,'rb')
                wfRate = wf.getframerate()
                wfChannels = wf.getnchannels()
                wfWidth = wf.getsampwidth()
                if not wfWidth in [1,2,4]:
                    raise WrongOperation("Only int8, int16 or int32 wav data can be accepted.")
                
                self.config_wave_format(None,wfWidth,wfChannels,wfRate,1024)

                secPerRead = self.chunkFrames/self.rate

                firstMessage = "{},{},{},{}".format(self.formats,self.channels,self.rate,self.chunkFrames)
                firstMessage = firstMessage + " "*(32-len(firstMessage))
                dataQueue.put(firstMessage.encode())

                data = wf.readframes(self.chunkFrames)
                while len(data) == self.chunkSize:
                    self._counter += secPerRead
                    dataQueue.put(data)
                    if True in [self.localErrFlag, self.remoteErrFlag, self.endFlag]:
                        data = b""
                        break
                    time.sleep(secPerRead)
                    data = wf.readframes(self.chunkFrames)
                if data != b"":
                    self._counter += len(data)/self.width/self.channels/self.rate
                    lastChunkData = data + b" "*(self.chunkSize-len(data))
                    dataQueue.put(lastChunkData)
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if True in [self.remoteErrFlag, self.localErrFlag]:
                    pass
                else:
                    dataQueue.put('endFlag')

        self.threadManager['read'] = threading.Thread(target=readWave,args=(wavFile,self.dataQueue,))
        self.threadManager['read'].start()

    def record(self,seconds=None):

        if self.safeFlag == False:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')

        if 'record' in self.threadManager.keys() and self.threadManager['record'].is_alive():
            raise WrongOperation('Another record task is running now.')

        if 'read' in self.threadManager.keys() and self.threadManager['read'].is_alive():
            raise WrongOperation('Record and Read are not allowed to run concurrently.')       

        if seconds != None:
            assert isinstance(seconds,(int,float)) and seconds > 0,'Expected <seconds> is positive int or float number.'

        def recordWave(seconds,dataQueue):
            try:
                self._counter = 0

                secPerRecord = self.chunkFrames/self.rate

                firstMessage = "{},{},{},{}".format(self.formats,self.channels,self.rate,self.chunkFrames)
                firstMessage = firstMessage + " "*(32-len(firstMessage))
                dataQueue.put(firstMessage.encode())

                if self.formats == 'int8':
                    ft = pyaudio.paInt8
                elif self.formats == 'int16':
                    ft = pyaudio.paInt16
                else:
                    ft = pyaudio.paInt32
                stream = self.p.open(format=ft,channels=self.channels,rate=self.rate,input=True,output=False)

                if seconds != None:
                    recordLast = True
                    while self._counter <= (seconds-secPerRecord):
                        data = stream.read(self.chunkFrames)
                        dataQueue.put(data)
                        self._counter += secPerRecord
                        if True in [self.localErrFlag, self.endFlag, self.remoteErrFlag]:
                            recordLast = False
                            break
                    if recordLast:
                        lastRecordFrames = int((seconds-self._counter)*self.rate)
                        data = stream.read(lastRecordFrames)
                        data += b" "*(self.chunkSize-len(data))
                        dataQueue.put(data)
                        self._counter = seconds
                else:
                    while True:
                        stream.read(self.chunkFrames)
                        dataQueue.put(data)
                        self._counter += secPerRecord
                        if True in [self.localErrFlag, self.endFlag, self.remoteErrFlag]:
                            break
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if True in [self.localErrFlag, self.remoteErrFlag]:
                    pass
                else:
                    dataQueue.put('endFlag')
            finally:
                stream.stop_stream()
                stream.close()

        self.threadManager['record'] = threading.Thread(target=recordWave,args=(seconds,self.dataQueue,))
        self.threadManager['record'].start()

    def recognize(self,func,args=None,interval=0.3):

        if not self.safeFlag:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')
        
        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('Another recognition task is running now.')

        if ('send' in self.threadManager.keys() and self.threadManager['send'].is_alive()) or ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()):
            raise WrongOperation('<local> mode and <remote> mode are not expected to run meanwhile.')

        def recognizeWave(dataQueue,func,args,resultQueue,interval):
            
            class VOD(object):
                def __init__(self):
                    self.lastRe = None
                    self.c = 0
                def __call__(self,re):
                    if re == self.lastRe:
                        self.c  += 1
                        if self.c == 3:
                            self.c = 1
                            return True
                        else:
                            return False
                    self.lastRe = re
                    self.c = 1
                    return False
                    
            vod = VOD()

            dataPerReco = []
            timesPerReco = None
            count = 0

            try:
                while True:
                    if self.localErrFlag == True:
                        break
                    if dataQueue.empty():
                        if ('read' in self.threadManager.keys() and self.threadManager['read'].is_alive()) or ('record' in self.threadManager.keys() and self.threadManager['record'].is_alive()):
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted data input from Read(file) or Record(microphone).')
                    else:
                        chunkData = dataQueue.get()
                        if timesPerReco == None:
                            # Compute timesPerReco and Throw the first message
                            timesPerReco = math.ceil(self.rate*interval/self.chunkFrames)
                            continue
                        if chunkData == 'endFlag':
                            count = timesPerReco + 1
                        else:
                            dataPerReco.append(chunkData)
                            count += 1
                        if count >= timesPerReco:
                            if len(dataPerReco) > 0:
                                with tempfile.NamedTemporaryFile('w+b',suffix='.wav') as waveFile:
                                    wf = wave.open(waveFile.name, 'wb')
                                    wf.setsampwidth(self.width)
                                    wf.setnchannels(self.channels)
                                    wf.setframerate(self.rate)
                                    wf.writeframes(b''.join(dataPerReco))
                                    wf.close()
                                    if args != None:
                                        result = func(waveFile.name,args)
                                    else:
                                        result = func(waveFile.name)        
                            if count > timesPerReco:
                                resultQueue.put((True,result))
                                break
                            else:
                                sof = vod(result)
                                resultQueue.put((sof,result))
                                if sof:
                                    dataPerReco = []
                                count = 0
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if self.localErrFlag == True:
                    pass
                else:
                    resultQueue.append('endFlag')

        self.threadManager['recognize'] = threading.Thread(target=recognizeWave,args=(self.dataQueue,func,args,self.resultQueue,interval,))
        self.threadManager['record'].start()

    def connect_to(self, proto='TCP', targetHost=None, targetPort=9509, timeout=10):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')

        if self.client != None:
            raise WrongOperation('Another local client is working.')

        assert proto in ['TCP','UDP'], "Expected <proto> is TCP or UDP but got {}.".format(proto)

        self.proto = proto
        self.targetHost = targetHost
        self.targetPort = targetPort

        if timeout != None:
            assert isinstance(time,int),'Expected <timeout> seconds is positive int number but got {}.'.format(timeout)
            socket.setdefaulttimeout(timeout)
        
        if proto == 'TCP':
            try:
                self.client = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
                self.client.connect((targetHost,targetPort))
                verification = self.client.recv(32)
                if verification != b'hello world':
                    raise NetworkError('Connection anomaly.')
            except ConnectionRefusedError:
                raise NetworkError('Target server has not been activated.')
            finally:
                self.client.close()
                self.client = None
                self.localErrFlag = True
        else:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.client.sendto(b'hello world',(targetHost,targetPort))
            try:
                i = 0
                while i < 5:
                    verification,addr = self.client.recvfrom(32)
                    if verification == b'hello world' and addr == (targetHost,targetPort):
                        break
                    i += 1
                if i == 5:
                    raise NetworkError('Cannot communicate with target server.')
            except TimeoutError:
                raise NetworkError('Target server seems has not been activated.')
            finally:
                self.client.close()
                self.client = None
                self.localErrFlag = True

        return True

    def send(self):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')
        
        if 'send' in self.threadManager.keys() and self.threadManager['send'].is_alive():
            raise WrongOperation('Another send task is running now.')      

        if self.client == None:
            raise WrongOperation('No activated local client.')
        
        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('<local> mode and <remote> mode are not expected to run meanwhile.')        

        def sendWave(dataQueue):
            try:
                while True:
                    if True in [self.localErrFlag,self.remoteErrFlag]:
                        break
                    if dataQueue.empty():
                        if ('read' in self.threadManager.keys() and self.threadManager['read'].is_alive()) or ('record' in self.threadManager.keys() and self.threadManager['record'].is_alive()):
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted data input from Read(file) or Record(microphone).')
                    else:
                        message = dataQueue.get()
                        if message == 'endFlag':
                            break
                        elif self.proto == 'TCP':
                            self.client.send(message)
                        else:
                            self.client.sendto(message,(self.targetHost,self.targetPort))
            except Exception as e:
                self.localErrFlag = True
                raise e
            finally:
                if self.remoteErrFlag == True:
                    pass
                else:
                    if self.localErrFlag == True:
                        lastMessage = 'errFlag'.encode()
                    else:
                        lastMessage = 'endFlag'.encode()

                    if self.proto == 'TCP':
                        self.client.send(lastMessage)
                    else:
                        self.client.sendto(lastMessage,(self.targetHost,self.targetPort))
                 
        self.threadManager['send'] = threading.Thread(target=sendWave,args=(self.dataQueue,))
        self.threadManager['send'].start()
    
    def receive(self):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')
        
        if 'receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive():
            raise WrongOperation('Another receive task is running now.')

        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('Local mode and Remote mode are not expected to run meanwhile.')     

        if self.client == None:
            raise WrongOperation('No activated network client.')

        def recvResult(resultQueue):
            try:
                while True:
                    if self.localErrFlag == True:
                        break               
                    if self.proto == 'TCP':
                        message = self.client.recv(self.chunkSize)
                    else:
                        i = 0
                        while i < 5:
                            message, addr = self.client.recvfrom(self.chunkSize)
                            if addr == (self.targetHost,self.targetPort):
                                break
                        if i == 5:
                            raise NetworkError('Communication between local client and remote server worked anomaly.')
                    message = message.decode.strip()
                    if message in ['endFlag','errFlag']:
                        break
            except TimeoutError:
                raise NetworkError('Please ensure server has been activated.')
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if message == 'errFlag':
                    self.remoteErrFlag == True
                else:
                    resultQueue.put('endFlag')

        self.threadManager['receive'] = threading.Thread(target=recvResult,args=(self.resultQueue,))
        self.threadManager['receive'].start()

    def get(self):
        while True:
            if self.resultQueue.empty():
                if ('recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive()) or ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()):
                    time.sleep(0.01)
                else:
                    return None
            else:
                message = self.resultQueue.get()
                if message == 'endFlag':
                    return None
                else:
                    return message

    @property
    def timer(self):
        return self._counter

class Server(object):

    def __init__(self, proto='TCP', bindHost=None, bindPort=9509):

        self.client = None
        self.threadManager = {}

        self.dataQueue = queue.Queue()
        self.resultQueue = queue.Queue()

        self.localErrFlag = False
        self.remoteErrFlag = False
        self.safeFlag = False
        self.bindHost = None
        self.bindPort = None

        socket.setdefaulttimeout(20)
        
        if bindHost != None:
            self.bind(proto,bindHost,bindPort)
    
    def __enter__(self):
        self.safeFlag = True
        return self
    
    def __exit__(self,errType,errValue,errTrace):
        self.wait()
        time.sleep(1)
        if self.client != None:
            self.client
        

    def wait(self):
        for name,thread in self.threadManager.items():
            if thread.is_alive():
                thread.join()

    def _config_wave_format(self,Format=None,Width=None,Channels=1,Rate=16000,ChunkFrames=1024):

        assert Channels==1 or Channels==2,"Expected <Channels> is 1 or 2 but got {}.".format(Channels)

        if Format != None:
            assert Format in ['int8','int16','int32'], "Expected <Format> is int8, int16 or int32 but got{}.".format(Format)
            assert Width == None, 'Only one of <Format> and <Width> is expected to assigned but both two.'
            if Format == 'int8':
                self.width = 1
            elif Format == 'int16':
                self.width = 2
            else:
                self.width = 4           
        else:
            assert Width != None, 'Expected one value in <Format> and <Width> but got two None.'
            assert Width in [1,2,4], "Expected <Width> is 1, 2 or 4 but got{}.".format(Format)
            self.width = Width
            if Width == 1:
                self.formats = 'int8'
            elif Width == 2:
                self.formats = 'int16'
            else:
                self.formats = 'int32'
    
        self.formats = Format
        self.channels = Channels
        self.rate = Rate
        self.chunkFrames = ChunkFrames
        self.chunkSize = self.width*Channels*ChunkFrames
    
    def bind(self, proto='TCP', bindHost=None, bindPort=9509):

        assert proto in ['TCP','UDP'],'Expected proto TCP or UDP but got {}'.format(proto)

        if self.bindHost != None:
            raise WrongOperation('Server has already bound to {}.'.format((self.bindHost,self.bindPort)))
        
        assert bindHost != None, 'Expected <host> is not None.'

        if proto == 'TCP':
            self.server = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        else:
            self.server = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        self.server.bind((bindHost,bindPort))

        self.proto = proto
        self.bindHost = bindHost
        self.bindPort = bindPort

    def connect_from(self,targetHost):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')

        if self.bindHost == None:
            raise WrongOperation('Bind Host IP and Port firstly by using .bind() method.')

        if self.client != None:
            raise WrongOperation('Another connection is running right now.')

        try:
            i = 0
            while i < 5:
                if self.proto == 'TCP':
                    self.server.listen(1)
                    self.client, addr = self.server.accept()
                    if addr[0] == targetHost:
                        self.client.send(b'hello world')
                        self.targetAddr = addr
                        break
                    else:
                        self.client.close()
                else:
                    vertification,addr = self.server.recvfrom(32)
                    if vertification == b'hello world' and addr[0] == targetHost:
                        self.client = self.server
                        self.client.sendto(b'hello world',addr)
                        self.targetAddr = addr
                        break
                i += 1
        except Exception as e:
            self.localErrFlag = True
            raise e
        except socket.timeout:
            raise NetworkError('No connect-application from any remote client.')
        else:
            if i >= 5:
                self.client = None
                self.localErrFlag = True
                raise NetworkError("Cannot connect from {}".format(targetHost))
            else:
                return True

    def receive(self):

        if not self.safeFlag:
            raise WrongOperation('Please run with safe mode by using <with> grammar.')

        if 'receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive():
            raise WrongOperation('Another receive thread is running now')
        
        if self.client == None:
            raise WrongOperation('Did not connect from any remote client.')

        def recvWave(dataQueue):
            try:
                if self.proto == 'TCP':
                    vertification = self.client.recv(32)
                else:
                    i = 0
                    while i < 5:
                        vertification,addr = self.client.recvfrom(32)
                        if addr == self.targetAddr:
                            break                            
                    if i == 5:
                        raise NetworkError('Communicate between server and remote client worked anomaly.')
                vertification = vertification.decode().strip().split(',')
                self._config_wave_format(vertification[0],None,int(vertification[1]),int(vertification[2]),int(vertification[3]))
            except Exception as e:
                self.localErrFlag = True
                raise e                    
            except socket.timeout:
                raise NetworkError('Did not received any data from remote client.')
            else:
                while True:
                    if self.localErrFlag == True:
                        break
                    try:
                        if self.proto == 'TCP':
                            message = self.client.recv(self.chunkSize)
                        else:
                            i = 0
                            while i < 5:
                                vertification,addr = self.client.recvfrom(32)
                                if addr == self.targetAddr:
                                    break                            
                            if i == 5:
                                raise NetworkError('Communicate between server and remote client worked anomaly.')
                    except Exception as e:
                        self.localErrFlag = True
                        raise e 
                    except socket.timeout:
                        raise NetworkError('Did not received any data from remote client.')
                    else:
                        if message == b'errFlag':
                            self.remoteErrFlag = True
                            break
                        elif message == b'endFlag':
                            dataQueue.put('endFlag')
                            break
                        else:
                            dataQueue.put(message)
            
        self.threadManager['receive'] = threading.Thread(target=recvWave,args=(self.dataQueue,))
        self.threadManager['receive'].start()

    def recognize(self,func,args=None,interval=0.3):

        if not self.safeFlag:
            raise WrongOperation('We only allow user to run speak client by using <with> grammar.')
        
        if 'recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive():
            raise WrongOperation('Another recognize task is running now.')

        def recognizeWave(dataQueue,func,args,resultQueue,interval):

            class VOD(object):
                def __init__(self):
                    self.lastRe = None
                    self.c = 0
                def __call__(self,re):
                    if re == self.lastRe:
                        self.c  += 1
                        if self.c == 3:
                            self.c = 1
                            return True
                        else:
                            return False
                    self.lastRe = re
                    self.c = 1
                    return False
                    
            vod = VOD()

            dataPerReco = []
            timesPerReco = None
            count = 0

            try:
                while True:
                    if True in [self.localErrFlag,self.remoteErrFlag]:
                        break
                    elif dataQueue.empty():
                        if 'receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive():
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted data input from Receive().')
                    else:
                        chunkData = dataQueue.get()
                        if timesPerReco == None:
                            # Compute timesPerReco and Throw the first message
                            timesPerReco = math.ceil(self.rate*interval/self.chunkFrames)
                            continue
                        if chunkData == 'endFlag':
                            count = timesPerReco + 1
                        else:
                            dataPerReco.append(chunkData)
                            count += 1

                        if count >= timesPerReco:
                            if len(dataPerReco) > 0:
                                with tempfile.NamedTemporaryFile('w+b',suffix='.wav') as waveFile:
                                    wf = wave.open(waveFile.name, 'wb')
                                    wf.setsampwidth(self.width)
                                    wf.setnchannels(self.channels)
                                    wf.setframerate(self.rate)
                                    wf.writeframes(b''.join(dataPerReco))
                                    wf.close()
                                    if args != None:
                                        result = func(waveFile.name,args)
                                    else:
                                        result = func(waveFile.name)        
                            if count > timesPerReco:
                                resultQueue.put((True,result))
                                break
                            else:
                                sof = vod(result)
                                resultQueue.put((sof,result))
                                if sof:
                                    dataPerReco = []
                                count = 0
            except Exception as e:
                self.localErrFlag = True
                raise e
            else:
                if self.localErrFlag == True or self.remoteErrFlag == True:
                    pass
                else:
                    resultQueue.append('endFlag')

        self.threadManager['recognize'] = threading.Thread(target=recognizeWave,args=(self.dataQueue,func,args,self.resultQueue,interval,))
        self.threadManager['record'].start()

    def send(self):

        if not self.safeFlag:
            raise WrongOperation('Please run under safe state by using <with> grammar')
            #print('Running without safe mode. Please use exit() function finally to ensure tasks safely')

        if 'send' in self.threadManager.keys() and self.threadManager['send'].is_alive():
            raise WrongOperation('Another send thread is running now')

        if self.client == None:
            raise WrongOperation('No activated server client')

        if resultQueue == None:
            resultQueue = self.resultQueue

        def sendResult(resultQueue):
            try:
                while True:
                    if True in [self.localErrFlag,self.remoteErrFlag]:
                        break
                    if resultQueue.empty():
                        if ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()) or ('recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive()):
                            time.sleep(0.01)
                        else:
                            raise WrongOperation('Excepted production from Receive() and Recognize() task.')
                    else:
                        message = resultQueue.get()
                        if 'endFlag' == message:
                            break
                        else:
                            #message has a format: (sectionOverFlag,result)
                            data = []
                            #we will send data with a format: sectionOverFlag(Y/N/T) + ' ' + resultData + placeHolderSymbol(' ')
                            rSize = self.chunkSize - 2  # -len("Y ")
                            #result is likely longer than rSize, so cut it
                            #Get the N front rSize part, give them all 'Y' sign   
                            for i in range(len(message[1])//rSize):
                                data.append('T ' + message[1][i*rSize:(i+1)*rSize])
                            if len(message[1][i*rSize:]) > 0:
                                lastData = ''
                                if message[0]:
                                    lastData = 'Y ' + message[1][i*rSize:]
                                else:
                                    lastData = 'N ' + message[1][i*rSize:]
                                lastData += " "*(self.chunkSize-len(lastData))
                                data.append(lastData)
                            else:
                                if message[0]:
                                    data[-1] = 'Y ' + data[-1][2:]
                                else:
                                    data[-1] = 'N ' + data[-1][2:]                           

                            data = ''.join(data)
    
                            if self.proto == 'TCP':
                                self.client.send(data.encode())
                            else:
                                self.client.sendto(data.encode(),self.targetAddr)
            except Exception as e:
                self.localErrFlag = True
                raise e
            finally:
                if self.remoteErrFlag == True:
                    pass
                elif self.localErrFlag == True:
                    if self.proto == 'TCP':
                        self.client.send('errFlag'.encode())
                    else:
                        self.client.sendto('errFlag'.encode(),self.targetAddr)
                else:
                    if self.proto == 'TCP':
                        self.client.send('endFlag'.encode())
                    else:
                        self.client.sendto('endFlag'.encode(),self.targetAddr)
            
        self.threadManager['send'] = threading.Thread(target=sendResult,args=(self.resultQueue,))
        self.threadManager['send'].start()                

    def get(self):
        while True:
            if self.resultQueue.empty():
                if ('recognize' in self.threadManager.keys() and self.threadManager['recognize'].is_alive()) or ('receive' in self.threadManager.keys() and self.threadManager['receive'].is_alive()):
                    time.sleep(0.01)
                else:
                    return None
            else:
                message = self.resultQueue.get()
                if message == 'endFlag':
                    return None
                else:
                    return message