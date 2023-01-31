import sentencepiece as spm
import torch
import pandas as pd

spm.SentencePieceTrainer.train('--input=135-0.txt --model_prefix=sentencepiece --vocab_size=23000')  # GATO uses 32000 subwords, use a bigger corpus later

sp = spm.SentencePieceProcessor()
sp.load('sentencepiece.model')

word_associations = pd.read_csv('/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/pagrawal/klab/ViT_universaljudge/word_associations.csv')

cues = word_associations["Cues"].tolist()
associations = word_associations["Associations"].tolist()
labels = word_associations["Label"].tolist()

for i in range(len(cues)):
    cue_vector = sp.encode_as_ids(cues[i])
    association_vector = sp.encode_as_ids(associations[i])
    print(cue_vector, association_vector)
    torch.save(cue_vector, '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/pagrawal/klab/ViT_universaljudge/Datasets/word_assoc/cues/'+str(i)+'_'+str(labels[i])+'.pt')
    torch.save(association_vector, '/n/holylfs05/LABS/pfister_lab/Lab/coxfs01/pfister_lab2/Lab/pagrawal/klab/ViT_universaljudge/Datasets/word_assoc/associations/'+str(i)+'_'+str(labels[i])+'.pt')

