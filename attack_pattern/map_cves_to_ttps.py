import ast
import csv
import json
import os
import time
from pathlib import Path

import fire
import torch

import nltk
import pandas as pd
from scipy import spatial
from sentence_transformers import SentenceTransformer

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


def save_csv_file(csv_file_path, data):
    # Open the file in write mode
    with open(csv_file_path, mode='a', newline='', encoding="utf-8") as file:
        # Create a writer object
        writer = csv.writer(file)

        for row in data:
            # Write the data to the file
            writer.writerow(row)

        file.close()


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


def get_embedding(txt, embedding_cache, bert_model):
    if txt in embedding_cache:
        return embedding_cache[txt]
    emb = bert_model.encode([txt])[0]
    embedding_cache[txt] = emb
    return emb


def get_embedding_distance(txt1, txt2, embedding_cache, bert_model):
    p1 = get_embedding(txt1, embedding_cache, bert_model)
    p2 = get_embedding(txt2, embedding_cache, bert_model)
    score = spatial.distance.cosine(p1, p2)
    return score


def get_relevant_ttp_ids(attack_pattern, embedding_cache, th, bert_model, ttps_dict):
    ttps_below_threshold = {}
    min_dist = 25
    ttp_id_min = None
    for id, tech_list in ttps_dict.items():
        for v in tech_list:
            d = (0.5 * get_embedding_distance(attack_pattern, v[0], embedding_cache, bert_model) +
                 0.5 * get_embedding_distance(attack_pattern, v[1], embedding_cache, bert_model))

            if d < th:
                if id in ttps_below_threshold:
                    if d < ttps_below_threshold[id]:
                        ttps_below_threshold[id] = d
                else:
                    ttps_below_threshold[id] = d

            if d < min_dist:
                min_dist = d
                ttp_id_min = id

    if min_dist >= th:
        closest_ttp = None
    else:
        closest_ttp = {ttp_id_min: min_dist}

    return {"ttps_below_threshold": ttps_below_threshold, "closest_ttp": closest_ttp}


def remove_consec_newline(s):
    ret = s[0]
    for x in s[1:]:
        if not (x == ret[-1] and ret[-1] == '\n'):
            ret += x
    return ret


def load_ttps_dictionary():
    df = pd.read_csv('data/enterprise_techniques_customized.csv')

    ttps_dict = {}

    prev_id = None

    for _, row in df.iterrows():
        _id = row['ID']
        if not pd.isnull(_id):
            ttps_dict[_id] = [[row['Name'], row['Description']]]
            prev_id = _id
        else:
            ttps_dict[prev_id].append([row['Name'], row['Description']])

    return ttps_dict


def get_all_ttps(attack_pattern, embedding_cache, bert_model, ttps_dictionary, th=0.6):
    attack_pattern = remove_consec_newline(attack_pattern)
    attack_pattern = attack_pattern.replace('\t', ' ')
    attack_pattern = attack_pattern.replace("\'", "'")

    if len(attack_pattern) > 0:
        return get_relevant_ttp_ids(attack_pattern, embedding_cache, th, bert_model, ttps_dictionary)

    return {}


def extract_dataset_ttps(dataset_path,
                         text_col,
                         label_col,
                         destination_dir,
                         entity_extraction_weight,
                         sentence_classification_weight,
                         distance_threshold,
                         continue_prediction=True):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    ttps_dictionary = load_ttps_dictionary()

    entity_extraction_model = 'roberta-large'
    sentence_classification_model = 'roberta-large'
    bert_model = SentenceTransformer('all-mpnet-base-v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    entity_model = EntityRecognition(entity_extraction_model).to(device)
    entity_model.load_state_dict(torch.load(entity_extraction_weight, map_location=device), strict=False)

    sequence_length_sentence = 256
    sequence_length_entity = 256

    if MODELS[sentence_classification_model][3] == 'bert':
        sentence_model = SentenceClassificationBERT(sentence_classification_model, num_class=2).to(device)
        sentence_model.load_state_dict(torch.load(sentence_classification_weight, map_location=device))
    elif MODELS[sentence_classification_model][3] == 'roberta':
        sentence_model = SentenceClassificationRoBERTa(sentence_classification_model, num_class=2).to(device)
        sentence_model.load_state_dict(torch.load(sentence_classification_weight, map_location=device), strict=False)
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

    # Set the destination file path
    ladder_dataset_path = os.path.join(destination_dir, 'ladder_prediction_results.csv')

    # Continue from previous covered CVEs
    last_data_point_index = None
    if continue_prediction and Path(ladder_dataset_path).is_file():
        covered_cve_dataset = pd.read_csv(ladder_dataset_path)
        try:
            last_data_point_index = covered_cve_dataset.index[-1]
        except:
            last_data_point_index = None

    # Save Header
    if not Path(ladder_dataset_path).is_file():
        save_csv_file(ladder_dataset_path, [['text_column',
                                             'label_column',
                                             'Ladder_Predictions',
                                             'Ladder_Delays',
                                             'No_Prediction']])

    embedding_cache = {}
    for index, row in data.iterrows():
        cve_desc = row[text_col]
        cve_labels = row[label_col]

        if continue_prediction and last_data_point_index:
            if index <= last_data_point_index:
                continue

        # Extract attack patterns from the target text
        start_time = time.time()
        attack_patterns = extract_attack_patterns(cve_desc,
                                                  sentence_model,
                                                  entity_model,
                                                  tokenizer_sen,
                                                  token_style_sen,
                                                  sequence_len_sen,
                                                  tokenizer_ent,
                                                  token_style_ent,
                                                  sequence_len_ent,
                                                  device)

        # Find the corresponding TTPs to each attack pattern
        mapping_result = []
        no_prediction = True
        for attack_pattern in attack_patterns:
            relevant_ttps = get_all_ttps(attack_pattern, embedding_cache, bert_model, ttps_dictionary,
                                         distance_threshold)
            closest_ttp = relevant_ttps['closest_ttp']
            ttps_below_threshold = relevant_ttps['ttps_below_threshold']

            mapping_result.append({'attack_pattern': attack_pattern,
                                   'closest_ttp': closest_ttp,
                                   'ttps_below_threshold': ttps_below_threshold})

            if closest_ttp is not None:
                no_prediction = False

        # Save the prediction results
        delay = time.time() - start_time
        data_list = [[cve_desc, cve_labels, str(mapping_result), delay, no_prediction]]
        save_csv_file(ladder_dataset_path, data_list)

        print(f'Ladder predictions for Text: {index}')
        if no_prediction:
            print('Empty')
        else:
            for index, result in enumerate(mapping_result):
                attack_pattern = result['attack_pattern']
                closest_ttp = result['closest_ttp']
                ttps_below_threshold = result['ttps_below_threshold']
                ap_id = index + 1
                print(f'\tAttack Pattern {ap_id}: {attack_pattern} -> {closest_ttp}')
                print('\tClosest TTPs:')
                for key, value in ttps_below_threshold.items():
                    print(f'\t\t{key}: {value}')
        print(f"--- {delay:.2f} seconds ---\n")


def add_prediction_columns(ladder_dataset_path, destination_dir, ladder_col):
    df = pd.read_csv(ladder_dataset_path)
    prediction_list1 = []  # Closest TTP to each attack pattern
    prediction_list2 = []  # TTPs below the threshold
    prediction_list1_col_index = df.columns.get_loc(ladder_col) + 1
    prediction_list2_col_index = prediction_list1_col_index + 1

    for index, row in df.iterrows():
        ladder_predictions = row[ladder_col]
        ladder_predictions = ast.literal_eval(ladder_predictions)
        closest_ttps_general = []
        ttps_below_threshold_general = []

        for prediction in ladder_predictions:
            if prediction:
                try:
                    attack_pattern_closest_ttps = list(prediction['closest_ttp'].keys())
                    for ttp in attack_pattern_closest_ttps:
                        if ttp not in closest_ttps_general:
                            closest_ttps_general.append(ttp)
                except:
                    pass

                try:
                    attack_pattern_closest_ttps_below_threshold = list(prediction['ttps_below_threshold'].keys())
                    for ttp in attack_pattern_closest_ttps_below_threshold:
                        if ttp not in ttps_below_threshold_general:
                            ttps_below_threshold_general.append(ttp)
                except:
                    pass

        prediction_list1.append(closest_ttps_general)
        prediction_list2.append(ttps_below_threshold_general)

    df.insert(loc=prediction_list1_col_index, column='closest_ttp', value=prediction_list1)
    df.insert(loc=prediction_list2_col_index, column='ttps_below_threshold', value=prediction_list2)

    df.to_csv(os.path.join(destination_dir, 'ladder_prediction_results_new_columns.csv'), index=False)


fire.Fire({
    'extract_dataset_ttps': extract_dataset_ttps,
    'add_prediction_columns': add_prediction_columns
})
