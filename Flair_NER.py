# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:05:38 2019

@author: God
"""

#import commands for flair ner
from flair.data import Sentence
from flair.models import SequenceTagger

#Load NER Model
tagger = SequenceTagger.load('ner')

#Sample text to run NER
text = 'Jackson is placed in Microsoft located in Redmond'

#passing text to sentence
sentence = Sentence(text)

# Run NER on sentence to identify Entities
tagger.predict(sentence)


# print the entities with below command
for entity in sentence.get_spans('ner'):
    print(entity)
    
    
print(sentence.to_tagged_string())

#Sample text
text1 = 'Redmond is coming to New York city'

#passing text to sentence
sentence = Sentence(text1)

# Run NER on sentence to identify Entities
tagger.predict(sentence)


# print the entities with below command
for entity in sentence.get_spans('ner'):
    print(entity)
    
    


#sample paragraph text
text2 = "During a heated deposition this past June, Elon Musk finally seemed to admit that his harshest critics were right. Since forcing through the controversial 2016 purchase of SolarCity Corp., the struggling solar sales-and-installation business he co-founded with his cousins, Tesla Inc.'s chief executive officer has faced almost-constant criticism: The move was called a catastrophe for Tesla, a $2 billion-plus bailout of a debt-saddled company of which Musk himself was chairman and the largest shareholder. Despite plummeting sales and substantial layoffs in the solar division under Tesla after the merger, Musk has fervently defended the SolarCity acquisition, once calling it “blindingly obvious” and a “no-brainer.”"


#Import segtok library to split the paragraph into sentences
from segtok.segmenter import split_single

sentences = [Sentence(sent, use_tokenizer=True) for sent in split_single(text2)]

# predict tags for list of sentences
tagger.predict(sentences)


# print the entities with below command
for sent in sentences:
    for entity in sent.get_spans('ner'):
        print(entity)















