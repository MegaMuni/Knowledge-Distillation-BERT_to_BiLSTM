# Knowledge Distillation from BERT to BiLSTM

All the details about the experiments and their results are present in Report.pdf. Following notes corresponds to the explanation of the code files. 

NOTE: You should have the glove embeddings file and also load the datasets before you run the student model code (BiLSTM).


Firstly, we'll need to load the glue benchmark datasets.
We have the corresponding code for each dataset for doing this in the folder 'load'.

Secondly, we need to augment the data, code for doing this for each of the dataset is present inside the folder 'upgrade'

Thirdly, we append the teacher logits and true label logits of the train split to the dataset. We do this part of the preprocessing in the code files inside 'upgrade'.

BiLSTM_MNLI.py: Code for training student BiLSTM with teacher BERT-base-cased finetuned on MNLI and validating on MNLI dataset

BiLSTM_MRPC.py: Code for training student BiLSTM with teacher BERT-base-cased finetuned on MRPC and validating on MRPC dataset

BiLSTM_QQP.py: Code for training student BiLSTM with teacher BERT-base-cased finetuned on QQP and validating on QQP dataset

BiLSTM_RTE.py: Code for training student BiLSTM with teacher BERT-base-cased finetuned on RTE and validating on RTE dataset

BiLSTM_SST2.py: Code for training student BiLSTM with teacher BERT-base-cased finetuned on SST2 and validating on SST2 dataset
