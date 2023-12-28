# Whisper Automatic Speech Recognition Minds14

## Absatrct:

In digital era, automatic speech recognition (ASR) has become a key component in applications that require voice interaction. This project explores the application of pretrained ASR models, specifically the Whisper-tiny model, in speech recognition from the MINDS14 dataset. Whisper-tiny is known for its advantages in minimizing size and speed of inference, making it ideal for devices with limited resources. The MINDS14 dataset, rich in language and accent variations, challenges ASR models in understanding variations and nuances in everyday speech. By integrating these two components, this research aims to improve speech recognition accuracy and understand the limitations and potential of pretrained ASR models in real application contexts.

## Objective:

To develop a Automatic Speech Recognition (ASR) for English languages using pretrained model Whisper and dataset Minds14 by PolyAI.

## Datasets:

MINDS-14, is a dataset focused on multilingual intent detection from spoken data in the e-banking domain. By combining machine translation models with advanced multilingual sentence encoders like LaBSE, the research achieves robust intent detection across a diverse range of languages. Key findings underscore the efficacy of the "ASR-then-translate" approach, especially for major languages, and highlight the importance of in-domain model fine-tuning. The work emphasizes the potential of multilingual intent detection and sets the stage for broader applications in voice-based conversational AI.

For this case im not using full datasets and just using en-US, en-GB, and an-AU.

### Exploratory Data Analysis

- en-US
  - columns : ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id']
  - rows : 563
- en-AU
  - columns : ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id']
  - 654
- en-GB
  - columns : ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id']
  - 592

#### Check intent_class distribution

The intent_class label is have 14 different values:

- 0 = abroad
- 1= adress
- 2= app_error
- 3= atm_limit
- 4=balance
- 5= business_loan
- 6=card_issues
- 7= cash_deposite
- 8=direct_debit
- 9=freeze
- 10=latest_transactions
- 11=joint_account
- 12=high_value_payment
- 13=pay_bill

#### Visualize intent_class Audio

-
-
-

### Training

### Predict

### Evaluate
