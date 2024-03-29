#import spacy
import re

class tokenize(object):
    
    def __init__(self):
        with open ("vocab_file.txt", "r") as file:
            self.vocab_file= file.read().strip()
            
    def tokenizer(self, sentence):
        tokens= []
        for char in sentence:
            if char in self.vocab_file:
                tokens.append(char)
            else:
                tokens.append("UNK")
        return tokens

