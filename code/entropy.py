import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import fasttext
import fasttext.util
from tqdm import tqdm

preprocessing_folder_path = '/home/local/PSYCH-ADS/xuqian_chen/Github/twitter_entropy/data/preprocessing/'
preprocessed_folder_path = '/home/local/PSYCH-ADS/xuqian_chen/Github/twitter_entropy/data/preprocessed/'

if not os.path.isdir(preprocessed_folder_path):
    os.mkdir(preprocessed_folder_path)
if not os.path.isdir(preprocessing_folder_path):
    os.mkdir(preprocessing_folder_path)


def load_en_nrc_emotion_lexicon(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    data = [line.strip().split('\t') for line in lines]
    df = pd.DataFrame(data, columns=["word", "emotion", "value"])
    df["value"] = df["value"].astype(int)
    
    df = df[df["value"] == 1]
    emotion_dict = {emotion: set(df[df["emotion"] == emotion]["word"]) for emotion in df["emotion"].unique()}
    
    return emotion_dict


def load_jp_nrc_emotion_lexicon(file_path):
    emotion_dict = {"anger": set(), "anticipation": set(), "disgust": set(), "fear": set(), "joy": set(),
                    "negative": set(), "positive": set(), "sadness": set(), "surprise": set(), "trust": set()}

    with open(file_path, encoding="utf-8") as f:
        next(f)  # Skip the header line
        for line in f:
            split_line = line.strip().split("\t")
            japanese_word = split_line[-1]
            emotions = [emotion for emotion, value in zip(emotion_dict.keys(), split_line[1:-1]) if int(value) == 1]

            for emotion in emotions:
                emotion_dict[emotion].add(japanese_word.lower())

    return emotion_dict


# path_to_english_nrc = os.path.join(preprocessing_folder_path, 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
path_to_english_nrc = '/home/local/PSYCH-ADS/xuqian_chen/Github/twitter_entropy/data/preprocessing/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
english_nrc_emotion_dict = load_en_nrc_emotion_lexicon(path_to_english_nrc)

basic_emotions = ['anger', 'fear', 'joy', 'sadness', 'disgust', 'surprise']

en_basic_emotion_concepts = {
    emotion: list(words) for emotion, words in english_nrc_emotion_dict.items() if emotion in basic_emotions
}

if not os.path.isfile('cc.ja.300.bin'):
    fasttext.util.download_model('ja', if_exists='ignore') 
if not os.path.isfile('cc.en.300.bin'):
    fasttext.util.download_model('en', if_exists='ignore')


en_embeddings = fasttext.load_model('cc.en.300.bin')


# load pre-trained fastText embeddings
def load_fasttext_embeddings(model_name):
    model = api.load(model_name)
    # Get the mean vector for the list of words
    # mean_vector = np.mean([model1[word] for word in wordlist if word in word_vectors], axis=0)

    return model


# Vectorize the dictionary of moral and emotional concepts
def vectorize_concepts(concepts, embeddings):
    concept_vectors = {}
    for concept, words in concepts.items():
        vectors = [embeddings[word] for word in words if word in embeddings]
        if vectors:  # Checking if the list is not empty
            concept_vectors[concept] = np.mean(vectors, axis=0)
        
    return concept_vectors
# Vectorize the input text
def vectorize_text(text, embeddings):
    tokens = word_tokenize(text)
    vectors = [embeddings[token] for token in tokens if token in embeddings and token is not None]
    if vectors:  # Checking if the list is not empty
        return np.mean(vectors, axis=0)

# Calculate cosine similarity between input text and moral and emotional concepts
def calculate_similarities(text_vector, concept_vectors):
    similarities = {}
    if text_vector is not None:
        for concept, vector in concept_vectors.items():
            similarities[concept] = cosine_similarity(text_vector.reshape(1, -1), vector.reshape(1, -1))[0][0]
    else:
        for concept, vector in concept_vectors.items():
            similarities[concept] = np.nan
    return similarities

def emotion_entropy(post_emotions, threshold=0.5):
    binary_emotions = [1 if emotion >= threshold else 0 for emotion in post_emotions]
    probabilities = [emotion_count / len(post_emotions) for emotion_count in binary_emotions]
    
    entropy = -np.sum([p * np.log2(p) if p > 0 else 0 for p in probabilities])
    return entropy


def calculate_mixed_emotion_entropy(sentence, embeddings,concept_vectors, language):
    if language == 'ja':
        
        text_vector = vectorize_jp_text(sentence, embeddings)
        
    else:
        text_vector = vectorize_text(sentence, embeddings)
    similarities = calculate_similarities(text_vector, concept_vectors)
    entropy = emotion_entropy(list(similarities.values()))
    return similarities, entropy

####################
# English
####################


# Define your moral and emotional concepts dictionary here
en_concept_vectors = vectorize_concepts(en_basic_emotion_concepts, en_embeddings)
from multiprocessing import Pool, cpu_count

def process_text_en(t):
    try:
        similarities, entropy = calculate_mixed_emotion_entropy(t, en_embeddings, en_concept_vectors, 'en')
    except Exception as e:
        print(f"An error occurred: {e}")
        # Input
        similarities = {'anger':np.nan, 'fear':np.nan, 'joy':np.nan, 'sadness':np.nan, 'disgust':np.nan, 'surprise':np.nan}
        entropy = np.nan

    return (similarities, entropy)
usdf = pd.read_csv(os.path.join(preprocessed_folder_path, 'us.csv'))
with Pool(cpu_count()-1) as p:
    results = list(tqdm(p.imap(process_text_en, usdf['text']), total=len(usdf)))

# save results
import pickle
with open(os.path.join(preprocessing_folder_path, 'usdf_mixed_emotion_results.pkl'), 'wb') as f:
    pickle.dump(results, f)

# delete results, en_embeddings, en_concept_vectors
del results, en_embeddings, en_concept_vectors, usdf

############
# Japanese #
############
jpdf = pd.read_csv(os.path.join(preprocessed_folder_path, 'jp.csv'))


# path_to_japanese_nrc = os.path.join(preprocessing_folder_path, 'Japanese-NRC-EmoLex.txt')
path_to_japanese_nrc = '/home/local/PSYCH-ADS/xuqian_chen/Github/twitter_entropy/data/preprocessing/Japanese-NRC-EmoLex.txt'
japanese_nrc_emotion_dict = load_jp_nrc_emotion_lexicon(path_to_japanese_nrc)
ja_embeddings = fasttext.load_model('cc.ja.300.bin')

import MeCab
mecab = MeCab.Tagger("-Owakati")
def tokenize_japanese_text_mecab(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip().split()

def vectorize_jp_concepts(concepts, ja_embeddings):
    concept_vectors = {}
    for concept, words in concepts.items():
        vectors = [ja_embeddings.get_word_vector(word) for word in words if word is not None]
        if vectors:  # Checking if the list is not empty
            concept_vectors[concept] = np.mean(vectors, axis=0)
    return concept_vectors

# Vectorize the input text
def vectorize_jp_text(text, ja_embeddings):
    tokens = mecab.parse(text).strip().split()
    vectors =[ja_embeddings.get_word_vector(token) for token in tokens if token is not None]
    if vectors:  # Checking if the list is not empty
        return np.mean(vectors, axis=0)


ja_basic_emotion_concepts = {
    emotion: list(words) for emotion, words in japanese_nrc_emotion_dict.items() if emotion in basic_emotions
}

# Define your moral and emotional concepts dictionary here
ja_concept_vectors = vectorize_jp_concepts(ja_basic_emotion_concepts, ja_embeddings)


def process_text_jp(t):
    try:
        similarities, entropy = calculate_mixed_emotion_entropy(t, ja_embeddings, ja_concept_vectors, 'ja')
    except Exception as e:
        print(f"An error occurred: {e}")
        # Input
        similarities = {'anger':np.nan, 'fear':np.nan, 'joy':np.nan, 'sadness':np.nan, 'disgust':np.nan, 'surprise':np.nan}
        entropy = np.nan

    return (similarities, entropy)

with Pool(cpu_count()-1) as p:
    results = list(tqdm(p.imap(process_text_jp, jpdf['text']), total=len(jpdf)))

with open(os.path.join(preprocessing_folder_path, 'jpdf_mixed_emotion_results.pkl'), 'wb') as f:
    pickle.dump(results, f)