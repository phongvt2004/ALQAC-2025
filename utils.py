import string
from underthesea import word_tokenize
import os
import json
from vncorenlp import VnCoreNLP

import os
import shutil


def download_model(save_dir='./'):
    # current_path = os.path.abspath(os.getcwd())
    if save_dir[-1] == '/':
        save_dir = save_dir[:-1]
    if os.path.isdir(save_dir + "/models") and os.path.exists(save_dir + '/VnCoreNLP-1.2.jar'):
        pass
    else:
        os.mkdir(save_dir + "/models")
        os.mkdir(save_dir + "/models/dep")
        os.mkdir(save_dir + "/models/ner")
        os.mkdir(save_dir + "/models/postagger")
        os.mkdir(save_dir + "/models/wordsegmenter")
        # jar
        os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar")
        shutil.move("VnCoreNLP-1.2.jar", save_dir + "/VnCoreNLP-1.2.jar")
        # wordsegmenter
        os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab")
        os.system(
            "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr")
        shutil.move("vi-vocab", save_dir + "/models/wordsegmenter/vi-vocab")
        shutil.move("wordsegmenter.rdr", save_dir + "/models/wordsegmenter/wordsegmenter.rdr")
        # postagger
        os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/postagger/vi-tagger")
        shutil.move("vi-tagger", save_dir + "/models/postagger/vi-tagger")
        # ner
        os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-500brownclusters.xz")
        os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-ner.xz")
        os.system(
            "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/vi-pretrainedembeddings.xz")
        shutil.move("vi-500brownclusters.xz", save_dir + "/models/ner/vi-500brownclusters.xz")
        shutil.move("vi-ner.xz", save_dir + "/models/ner/vi-ner.xz")
        shutil.move("vi-pretrainedembeddings.xz", save_dir + "/models/ner/vi-pretrainedembeddings.xz")
        # parse
        os.system("wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/dep/vi-dep.xz")
        shutil.move("vi-dep.xz", save_dir + "/models/dep/vi-dep.xz")

number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
stop_word = number + chars + ["của", "và", "các", "có", "được", "theo", "tại", "trong", "về", 
            "hoặc", "người",  "này", "khoản", "cho", "không", "từ", "phải", 
            "ngày", "việc", "sau",  "để",  "đến", "bộ",  "với", "là", "năm", 
            "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "điểm", "đồng",
            "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây", 
            "như", "đó", "mà", "nơi", "”", "“"]

def remove_stopword(w):
    return w not in stop_word
def remove_punctuation(w):
    return w not in string.punctuation
def lower_case(w):
    return w.lower()

def bm25_tokenizer(text):
    tokens = word_tokenize(text)
    tokens = list(map(lower_case, tokens))
    tokens = list(filter(remove_punctuation, tokens))
    tokens = list(filter(remove_stopword, tokens))
    return tokens

def calculate_f2(precision, recall):        
    return (5 * precision * recall) / (4 * precision + recall + 1e-20)

def load_json(path):
    return json.load(open(path))
os.makedirs("./vncorenlp", exist_ok=True)
download_model(save_dir="./vncorenlp")
rdrsegmenter = VnCoreNLP("./vncorenlp/VnCoreNLP-1.2.jar", annotators="wseg", max_heap_size='-Xmx2g')
def word_segmentation(text):
   
    word_sentences = rdrsegmenter.tokenize(text)
    if len(word_sentences) > 0:
        return " ".join([" ".join(sentence) for sentence in word_sentences])
    else:
        return ""