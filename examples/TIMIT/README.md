# TIMIT recipe with ExKaldi

This is a example of TIMIT Acoustic-Phonetic Continuous Speech Corpus. 
More information for TIMIT, look https://catalog.ldc.upenn.edu/LDC93S1.

## Start
Please run these scripts from 01 to 09 step by step.

### 01_prepare_data.py
Prepare wav.scp, text, utt2spk and spk2utt files of _train_, _dev_ and _test_ datasets.
You need specify the TIMIT data root path.
```bash
python 01_prepare_data.py -t [your TIMIT corpus]
```

### 02_make_dict_and_LM.py
Prepare lexicons, N-grams language model. Then make Lexicon fst and Grammar fst and compose them.
Please done the step 01 beforehand.
```bash
python 02_make_dict_and_LM.py
```

### 03_compute_mfcc.py
Compute MFCC features of _train_, _dev_ and _test_. Then compute the CMVN statistics respectively.
Please done the step 01 and 02 beforehand.
```bash
python 03_compute_mfcc.py
```

### 04_train_mono_and_decode.py
Train the monophone GMM-HMM model with MFCC+delta feature of _train_ dataset. Then decode and compute WER of _test_ dataset.
Please done the step 01, 02 and 03 beforehand.
```bash
python 04_train_mono_and_decode.py
```
You can skip the training step.
```bash
python 04_train_mono_and_decode.py -s True
```

### 05_train_delta_and_decode.py
Train the triphone GMM-HMM model with MFCC+delta feature of _train_ dataset. Then decode and compute WER of _test_ dataset.
Please done the step 01, 02, 03 and 04 beforehand.
```bash
python 05_train_delta_and_decode.py
```
You can skip the training step.
```bash
python 05_train_delta_and_decode.py -s True
```

### 06_train_lda_mllt.py
Train the triphone GMM-HMM model with MFCC+splice+LDA+MLLT feature of _train_ dataset. Then decode and compute WER of _test_ dataset.
Please done the step 01, 02, 03, 04 and 05 beforehand.
```bash
python 06_train_lda_mllt.py
```
You can skip the training step.
```bash
python 06_train_lda_mllt.py -s True
```

### 07_train_sat.py
Train the triphone GMM-HMM model with MFCC+splice+LDA+MLLT+fMLLR feature of _train_ dataset. Then decode and compute WER of _test_ dataset.
Please done the step 01, 02, 03, 04, 05 and 06 beforehand.
```bash
python 07_train_sat.py
```
You can skip the training step.
```bash
python 07_train_sat.py -s True
```

### 08_train_DNN_and_decode.py
Train the DNN model with fMLLR feature of _train_ dataset with Tensorflow. You need install Tensorflow beforehand. Then decode and compute WER of _test_ dataset.
Please done the step 01, 02, 03, 04, 05, 06 and 07 beforehand.
```bash
python 08_train_DNN_and_decode.py
```
You can skip the training step.
```bash
python 08_train_DNN_and_decode.py -s True
```

### 09_train_LSTM_and_decode.py
Train the LSTM model with fMLLR feature of _train_ dataset with Tensorflow. You need install Tensorflow beforehand. Then decode and compute WER of _test_ dataset.
Please done the step 01, 02, 03, 04, 05, 06, 07 and 08 beforehand.
```bash
python 09_train_LSTM_and_decode.py
```
You can skip the training step.
```bash
python 09_train_LSTM_and_decode.py -s True
```

### train_and_test_LM.py
Train the N-grams language models. Then decode and compute perplexity of _test_ dataset.
Please done the step 01 beforehand.
```bash
python train_and_test_LM.py
```