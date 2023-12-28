# Whisper Automatic Speech Recognition Minds14

## Absatrct:

In digital era, automatic speech recognition (ASR) has become a key component in applications that require voice interaction. This project explores the application of pretrained ASR models, specifically the Whisper-tiny model, in speech recognition from the MINDS14 dataset. Whisper-tiny is known for its advantages in minimizing size and speed of inference, making it ideal for devices with limited resources. The MINDS14 dataset, rich in language and accent variations, challenges ASR models in understanding variations and nuances in everyday speech. By integrating these two components, this research aims to improve speech recognition accuracy and understand the limitations and potential of pretrained ASR models in real application contexts.

## Objective:

To cultivate an Automatic Speech Recognition (ASR) system tailored for English languages, harnessing the capabilities of the pretrained Whisper model, and leveraging the MINDS14 dataset by PolyAI.

## Datasets:

MINDS-14 is a specialized dataset concentrating on multilingual intent detection within the e-banking domain. Through the fusion of machine translation models and sophisticated multilingual sentence encoders such as LaBSE, this research has pioneered robust intent detection spanning a myriad of languages. Noteworthy insights highlight the prowess of the "ASR-then-translate" paradigm, particularly for predominant languages, underscoring the significance of in-domain model fine-tuning. This venture accentuates the prospective applications of multilingual intent detection, paving the way for expansive integration within voice-centric conversational AI realms.

For this study, a subset of the dataset encompassing en-US, en-GB, and en-AU samples was employed.

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

- 0 = aboard
- 1= address
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

distribution:
![image](https://github.com/zanuura/Whisper-ASR-Minds14-English/assets/73764446/a6ebbe25-ff64-4861-b43a-44412618ebaa)

- Aboard:
  - transcript: yes I'm going to be traveling to the United Kingdom for a couple of weeks next month I need to know if I can use my card from from my bank account while I'm in Europe
  - ![image](https://github.com/zanuura/Whisper-ASR-Minds14-English/assets/73764446/a13bed07-272b-44ba-b1d2-edb4bcdc708e)

- Address:
  - Transcript: hi yes I like to change my address
  - ![image](https://github.com/zanuura/Whisper-ASR-Minds14-English/assets/73764446/8a970884-9e67-4248-91ac-e96c34a61362)

- App Error:
  - Transcript: high on the app isn't loading information
  - ![image](https://github.com/zanuura/Whisper-ASR-Minds14-English/assets/73764446/0bd2bc10-ba01-4043-a945-4ba20e15c44e)


### Training:

For the model i using Whisper-tiny.
Trainng Configuration:
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-tiny-minds14-english",
    # num_train_epochs=4,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=3e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=400,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    args= training_args,
    model=model,
    train_dataset=encoded_datasets['train'],
    eval_dataset=encoded_datasets['test'],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
```

The training results reveal potential overfitting, potentially attributed to the amalgamation of three distinct subdatasets and the inherent accent variations across en-US, en-AU, and en-GB.

```shell
Step	Training Loss	Validation Loss	Wer Ortho	Wer	Accuracy	F1 Score	R2 Score	Bleu Score	Rouge 1 r	Rouge 1 p	Rouge 1 f
400	0.926500	0.484265	27.917695	0.289420	0.337017	0.504132	0.429257	0.658894	0.893532	0.855359	0.868297
800	0.033500	0.587064	25.168724	0.253734	0.348066	0.516393	0.755690	0.685192	0.888293	0.856936	0.866785
1200	0.008800	0.653538	23.374486	0.236755	0.353591	0.522449	0.756043	0.700642	0.894548	0.868341	0.876103
1600	0.003800	0.665814	26.962963	0.270712	0.325967	0.491667	0.712642	0.674692	0.888410	0.856119	0.865092
2000	0.004800	0.683626	25.810700	0.257507	0.334254	0.501035	0.747899	0.680418	0.890615	0.858121	0.868281
2400	0.004200	0.706959	26.320988	0.264581	0.328729	0.494802	0.742737	0.671144	0.886179	0.854190	0.864434
2800	0.004600	0.731004	28.757202	0.288162	0.331492	0.497925	0.574304	0.669641	0.870911	0.853557	0.855301
3200	0.002200	0.749025	30.600823	0.303883	0.348066	0.516393	0.451041	0.669324	0.879368	0.864448	0.865712
3600	0.001800	0.769251	27.884774	0.281088	0.328729	0.494802	0.514432	0.658689	0.877936	0.857053	0.862386
4000	0.006600	0.797992	27.835391	0.280931	0.342541	0.510288	0.752152	0.660415	0.877038	0.851512	0.858118
```

### Predict:

While the model demonstrates commendable performance on shorter audio segments, challenges arise with longer audio inputs. Predictions are summarized with metrics encompassing Word Error Rate (WER), accuracy, F1 score, BLEU score, and Rouge scores.

```shell
[{'num': 0,
  'wer': 0.0,
  'werotho': '0.00%',
  'accuracy': 100.0,
  'f1_score': 1.0,
  'bleu_score': 0.046385961395229026,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 1.0,
  'Rouge 1 f': 0.999999995,
  'predict': "i'm calling about a business loan",
  'target': "i'm calling about a business loan"},
 {'num': 1,
  'wer': 1.0,
  'werotho': '100.00%',
  'accuracy': 0.0,
  'f1_score': 0.0,
  'bleu_score': 0,
  'Rouge 1 r': 0.8181818181818182,
  'Rouge 1 p': 0.6428571428571429,
  'Rouge 1 f': 0.7199999950720002,
  'predict': "i'm about make a large payment and it says i'll leave an sms code",
  'target': 'make a large payment and it says on leaving sms code'},
 {'num': 2,
  'wer': 0.3888888888888889,
  'werotho': '38.89%',
  'accuracy': 61.111111111111114,
  'f1_score': 0.7586206896551725,
  'bleu_score': 0.01069801418567523,
  'Rouge 1 r': 0.8387096774193549,
  'Rouge 1 p': 0.896551724137931,
  'Rouge 1 f': 0.8666666616722222,
  'predict': "hello i'm trying to use your app from the buying low is buying but unfortunately it doesn't work and can you give me some indication when the app will be working please",
  'target': "hello i'm trying to use your app from the and the bank lloyds bank but unfortunately it doesn't work can you give me some indication when the app will be working please ok thank you bye"},
 {'num': 3,
  'wer': 0.0,
  'werotho': '0.00%',
  'accuracy': 100.0,
  'f1_score': 1.0,
  'bleu_score': 0.02944119854967607,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 1.0,
  'Rouge 1 f': 0.999999995,
  'predict': 'i want to set up a joint account with my partner',
  'target': 'i want to set up a joint account with my partner'},
 {'num': 4,
  'wer': 0.0,
  'werotho': '0.00%',
  'accuracy': 100.0,
  'f1_score': 1.0,
  'bleu_score': 0.024569933058931386,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 1.0,
  'Rouge 1 f': 0.999999995,
  'predict': 'hi can you let me know how i can deposit money into my account',
  'target': 'hi can you let me know how i can deposit money into my account'},
 {'num': 5,
  'wer': 0.0,
  'werotho': '0.00%',
  'accuracy': 100.0,
  'f1_score': 1.0,
  'bleu_score': 0.06287167148414677,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 1.0,
  'Rouge 1 f': 0.999999995,
  'predict': 'please block my card',
  'target': 'please block my card'},
 {'num': 6,
  'wer': 0.9444444444444444,
  'werotho': '94.44%',
  'accuracy': 5.555555555555555,
  'f1_score': 0.10526315789473684,
  'bleu_score': 0.009879330055771796,
  'Rouge 1 r': 0.7647058823529411,
  'Rouge 1 p': 0.8125,
  'Rouge 1 f': 0.7878787828833793,
  'predict': "i have a car from to bank but it's not working what was my payment declined",
  'target': "i am i have a car from this bank but it's not working why was my payment decline"},
 {'num': 7,
  'wer': 0.0,
  'werotho': '0.00%',
  'accuracy': 100.0,
  'f1_score': 1.0,
  'bleu_score': 0.023330903410537226,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 1.0,
  'Rouge 1 f': 0.999999995,
  'predict': "what's the maximum amount of money i can withdraw from cash machine at one time",
  'target': "what's the maximum amount of money i can withdraw from cash machine at one time"},
 {'num': 8,
  'wer': 0.6341463414634146,
  'werotho': '63.41%',
  'accuracy': 36.58536585365854,
  'f1_score': 0.5357142857142857,
  'bleu_score': 0.00853569636970874,
  'Rouge 1 r': 0.8214285714285714,
  'Rouge 1 p': 0.7666666666666667,
  'Rouge 1 f': 0.7931034432818074,
  'predict': "i'm hoping you can help me i have a card with your bank and i just tried to use it and it's being received could you help me with this as a bnd declined as there a problem with a balance",
  'target': "i'm hoping you can help me i have a card with your bank and i decide to use it and it's being with you help me with this is appendix wind is there a problem with the balance"},
 {'num': 9,
  'wer': 1.0,
  'werotho': '100.00%',
  'accuracy': 0.0,
  'f1_score': 0.0,
  'bleu_score': 0,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 0.8571428571428571,
  'Rouge 1 f': 0.9230769181065088,
  'predict': 'hi i wanted to change my address',
  'target': 'i wanted to change my address'},
 {'num': 10,
  'wer': 0.0,
  'werotho': '0.00%',
  'accuracy': 100.0,
  'f1_score': 1.0,
  'bleu_score': 0.024569933058931386,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 1.0,
  'Rouge 1 f': 0.999999995,
  'predict': 'i need to transfer some money into my account how can i do this',
  'target': 'i need to transfer some money into my account how can i do this'},
 {'num': 11,
  'wer': 0.0,
  'werotho': '0.00%',
  'accuracy': 100.0,
  'f1_score': 1.0,
  'bleu_score': 0.023330903410537226,
  'Rouge 1 r': 1.0,
  'Rouge 1 p': 1.0,
  'Rouge 1 f': 0.999999995,
  'predict': "my bank card doesn't work and refuses to pay out can you please help me",
  'target': "my bank card doesn't work and refuses to pay out can you please help me"},
 {'num': 12,
  'wer': 0.13333333333333333,
  'werotho': '13.33%',
  'accuracy': 86.66666666666667,
  'f1_score': 0.9285714285714286,
  'bleu_score': 0.022510989169234545,
  'Rouge 1 r': 0.8571428571428571,
  'Rouge 1 p': 0.9230769230769231,
  'Rouge 1 f': 0.8888888838957477,
  'predict': "a today to make a payment with my card and the card won't go through",
  'target': "i tried to make a payment with my card and the card won't go through"},]
```

### Evaluate:

Evaluation metrics, including overall accuracy, WER, F1 score, and BLEU score, provide a comprehensive overview of the model's efficacy across diverse samples.

```python
accuracy_values = [entry['accuracy'] for entry in preds_eval_list]
overall_accuracy = sum(accuracy_values) / len(accuracy_values)

print(f"Overall Accuracy: {overall_accuracy:.2f}%")

wer_values = [entry['wer'] for entry in preds_eval_list]
overall_wer = sum(wer_values) / len(wer_values)

print(f"Overall WER: {overall_wer:.2f}%")

f1_values = [entry["f1_score"] for entry in preds_eval_list]
overall_f1 = sum(f1_values) / len(f1_values)

print(f"Overall F1: {overall_f1:.2f}%")

bleu_values = [entry["bleu_score"] for entry in preds_eval_list]
overall_bleu = sum(bleu_values) / len(bleu_values)

print(f"Overall Bleu: {overall_bleu:.2f}%")
```

```shell
Overall Accuracy: 63.04%
Overall WER: 0.37%
Overall WER: 0.68%
Overall WER: 0.02%
```

# Conlusion:

The integration of the Whisper-tiny model with the MINDS14 dataset underscores the potential and challenges inherent in deploying pretrained ASR models for multilingual intent detection. While the model exhibits proficiency in certain contexts, notably shorter audio segments, inherent complexities associated with accent variations and diverse linguistic nuances necessitate further exploration and refinement. Future endeavors may benefit from focused fine-tuning strategies tailored to specific linguistic subsets, paving the way for more robust and versatile voice-centric AI applications.

