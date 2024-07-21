import json

import fire
import torch

import nltk
import pandas as pd

from config import *
from inference import extract_sentences, classify_sent, extract_entities
from models import EntityRecognition, SentenceClassificationBERT, SentenceClassificationRoBERTa


def load_csv_file(csv_file):
    """

    :param csv_file:
    :return:
    """
    df = pd.read_csv(csv_file)

    return df


def load_json_file(json_file):
    """

    :param json_file:
    :return:
    """
    # Open the JSON file
    with open(json_file, errors="ignore") as json_file:
        data = json.load(json_file)
        return data


def extract_attack_patterns(text,
                            sentence_model,
                            entity_model,
                            tokenizer_sen,
                            token_style_sen,
                            sequence_len_sen,
                            tokenizer_ent,
                            token_style_ent,
                            sequence_len_ent,
                            device):
    sentences = extract_sentences(text)
    attack_patterns = []

    for sentence in sentences:
        # class 1: attack pattern sentence
        if classify_sent(sentence,
                         sentence_model,
                         tokenizer_sen,
                         token_style_sen,
                         sequence_len_sen,
                         device):
            ex = extract_entities(sentence,
                                  entity_model,
                                  tokenizer_ent,
                                  token_style_ent,
                                  sequence_len_ent,
                                  device)
            attack_pattern_list = ex.split("\n")
            attack_pattern_list = [item for item in attack_pattern_list if len(item) > 0]
            attack_patterns.extend(attack_pattern_list)

    return attack_patterns


def convert_string_to_list(string):
    # Remove the square brackets and single quotes
    cleaned_string = string[1:-1].replace("'", "")

    # Split the cleaned string by comma
    result_list = cleaned_string.split(", ")

    return result_list


def extract_dataset_ttps(dataset_path,
                         destination_path,
                         text_col,
                         label_col,
                         entity_extraction_weight,
                         sentence_classification_weight):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    entity_extraction_model = 'roberta-large'
    sentence_classification_model = 'roberta-large'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    entity_model = EntityRecognition(entity_extraction_model).to(device)
    entity_model.load_state_dict(torch.load(entity_extraction_weight, map_location=device))

    sequence_length_sentence = 256
    sequence_length_entity = 256

    if MODELS[sentence_classification_model][3] == 'bert':
        sentence_model = SentenceClassificationBERT(sentence_classification_model, num_class=2).to(device)
        sentence_model.load_state_dict(torch.load(sentence_classification_weight, map_location=device))
    elif MODELS[sentence_classification_model][3] == 'roberta':
        sentence_model = SentenceClassificationRoBERTa(sentence_classification_model, num_class=2).to(device)
        sentence_model.load_state_dict(torch.load(sentence_classification_weight, map_location=device))
    else:
        raise ValueError('Unknown sentence classification model')

    tokenizer_sen = MODELS[sentence_classification_model][1]
    token_style_sen = MODELS[sentence_classification_model][3]
    tokenizer_sen = tokenizer_sen.from_pretrained(sentence_classification_model)
    sequence_len_sen = sequence_length_sentence

    tokenizer_ent = MODELS[entity_extraction_model][1]
    token_style_ent = MODELS[entity_extraction_model][3]
    tokenizer_ent = tokenizer_ent.from_pretrained(entity_extraction_model)
    sequence_len_ent = sequence_length_entity

    data = pd.read_csv(dataset_path)

    for index, row in data.iterrows():
        text = row.iloc[text_col]
        labels = convert_string_to_list(row.iloc[label_col])

        # Extract attack patterns from the target text
        attack_patterns = extract_attack_patterns(text,
                                                  sentence_model,
                                                  entity_model,
                                                  tokenizer_sen,
                                                  token_style_sen,
                                                  sequence_len_sen,
                                                  tokenizer_ent,
                                                  token_style_ent,
                                                  sequence_len_ent,
                                                  device)

        for ap in attack_patterns:
            print(f'CVE {index}: {ap}')


fire.Fire(extract_dataset_ttps)
