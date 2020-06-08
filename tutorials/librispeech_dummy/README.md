This dummy corpus were from librispeech train clean 100 dataset.

After you download this dummy corpus, you have to run the follow command to reset the abspath in wav.scp file. 
```
python reset_wav_path.py
```

The distribution of this dummy corpus is listed as follow:

.

reset_wav_path.py  //used to reset the file abspath info in wav.scp

pronunciation.txt  //the map from word to pronunciation

-- train  //data for training, 100 utterances from 10 speaker, each speaker has 10 utterances 

    -- wav.scp  //the map from utt-ID to wav file path 

    -- text  //reference transcription in text format

    -- spk2utt  //the map from speaker to utt-ID 

    -- utt2spk  //the map from utt-ID to speaker

-- test  //data for test, 20 utterances from 4 speaker, each speaker has 5 utterances 

    -- wav.scp  //the map from utt-ID to wav file path

    -- text  //reference transcription in text format

    -- spk2utt  //the map from speaker to utt-ID 

    -- utt2spk  //the map from utt-ID to speaker

-- wav //wav data directory, 120 files
    
    -- *.wav // wav files
