**This project uses PyTorch and lightning-flash to fine-tune ASR (wav2vec2) with non-standard accented English varieties to better understand a wider range of Englishes.**

**Goals:**
-
- Applied: Improving ability for ML to understand non-standard accented English can improve transcriptions, subtitles, and communication between users with different accents or varieties of English.
- Theoretical: ML can simulate monolingual English speakers and help us understand, theoretically, what conditions are required to adapt to accented speech.

**Highlights:**
- 
- All ASR models were Wav2vec2 trained on 960 hours of Libri-speech
- American English -> Wav2vec2 = 0 WER
- Non-standard Accented English -> Wav2vec2 = .05-.6 WER, depending on the accent
- Wav2vec2 + fine-tuning: 5 minutes of one non-standard accent of English (Korean) = no improvement
- Wav2vec2 + fine-tuning: 5 minutes of multiple non-standard accent of English = no improvement

Next Steps/Broader Questions:
-
- How much fine-tuning is necessary? Quantity? Quality?
- How much improvement should we expect to see?
- Is this model representative of how humans perceive speech?
