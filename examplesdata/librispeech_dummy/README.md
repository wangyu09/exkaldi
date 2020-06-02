This dummy corpus were from librispeech train clean 100 dataset.

After you download this dummy corpus. To ensure the using, you have to run this command to update the abspath is wav.scp file. 
```
python reset_wav_path.py
```

The distribution of this dummy corpus:

.

reset_wav_path.py  //reset the file abspath info in wav.scp

pronunciation.txt  //the map from word to pronunciation

-- train  //datafor

    -- wav.scp  //the map from utt-ID to wav file path 

    -- text  //reference transcription

    -- spk2utt  //the map from speaker to utt-ID 

    -- utt2spk  //the map from utt-ID to speaker

-- test  //datafor

    -- wav.scp  //the map from utt-ID to wav file path 

    -- text  //reference transcription

    -- spk2utt  //the map from speaker to utt-ID 

    -- utt2spk  //the map from utt-ID to speaker

-- wav //wav data directory
    
    -- *.wav // wav files
