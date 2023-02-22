import sentencepiece as spm
import torch
import pandas as pd
import json

spm.SentencePieceTrainer.train('--input=135-0.txt --model_prefix=sentencepiece --vocab_size=23000')  # GATO uses 32000 subwords, use a bigger corpus later

sp = spm.SentencePieceProcessor()
sp.load('sentencepiece.model')

# Opening JSON file
f = open('/Users/brandontang/Desktop/Harvard/Spring 2023/Thesis/main_caption_data.json')
  
# returns JSON object as a dictionary
captions = json.load(f)

# Closing file
f.close()

imgcap_human_tokens = []
imgcap_human_image_names = []
imgcap_machine_tokens = []
imgcap_human_machine_names = []

for img_name in list(captions.keys()):
    for i in range(len(captions[img_name]['human'])):
        human_caption = captions[img_name]['human'][i][0]
        human_caption_vec = sp.encode_as_ids(human_caption)
        imgcap_human_tokens.append(human_caption_vec)
        imgcap_human_image_names.append(img_name)

    for machine_name in list(captions[img_name]['machine'].keys()):
        machine_caption = captions[img_name]['machine'][machine_name]
        machine_caption_vec = sp.encode_as_ids(machine_caption)
        imgcap_machine_tokens.append(machine_caption_vec)
        imgcap_human_machine_names.append(img_name)

for i in range(len(imgcap_human_tokens)):
    torch.save(imgcap_human_tokens[i], '/Users/brandontang/Desktop/Harvard/Spring 2023/Thesis/image_caption_data/' + imgcap_human_image_names[i] + '_' + str(1) + '.pt')

for i in range(len(imgcap_machine_tokens)):
    torch.save(imgcap_machine_tokens[i], '/Users/brandontang/Desktop/Harvard/Spring 2023/Thesis/image_caption_data/' + imgcap_human_machine_names[i] + '_' + str(0) + '.pt')

