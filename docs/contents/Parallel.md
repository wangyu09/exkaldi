
# Parallel Processes In Exkaldi
Starting from version 1.3, we support multiple processes so as to deal with a large-scale corpus. When process a small one, such as TIMIT coupus in our examples, we prefer to apply the single process that will hold on data in memory, and defaultly appoint buffer as IO streams during the processing. For example, we want to compute the MFCC feature from a script-table file:
```python
# Single process
wavFile = "wav.scp"
feat = Exkaldi.compute_mfcc(wavFile, rate=16000, name="dummy_mfcc")
```
The returned object: ___feat___ would be an Exkaldi __BytesFeat__ object.
Implementing parallel processes is easy in Exkaldi because you only need to give the function multiple resources. Exkaldi will decide a parallel strategy and assign these recources automatically. For example:
```python
# Parallel processes
wavFiles = ["wav_0.scp", "wav_1.scp"]
feat = Exkaldi.compute_mfcc(wavFiles, rate=16000, name="dummy_mfcc", outFile="dummy_mfcc.ark")
```
This function will run double processes parallelly because it received two scripts. At the moment, the IO streams must be files and the currency will become index table. In above case, the returned object: ___feat___ would be two Exkaldi __IndexTable__ objects.  
In particular, we not only accept multiple recources but also different parameters. This is different with Kaldi. Just for example, we will use different sample rates to compute the MFCC feature:
```python
# Parallel processes
wavFile = "wav.scp"
rates = [16000, 8000]
feat = Exkaldi.compute_mfcc(wavFile, rate=rates, name="dummy_mfcc", outFile="dummy_mfcc.ark")
```