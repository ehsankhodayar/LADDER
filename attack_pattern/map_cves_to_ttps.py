import ast
import csv
import json
import os
import pathlib
import platform
from pathlib import Path

import fire
import nltk
import pandas as pd
import torch
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


def load_ttps_dictionary(dataset_supported_ttps: list = None):
    # Example: 'data/mitre_attack_technique_dataset.csv'
    # dictionary_path = input("Enter the path to the enterprise techniques dictionary: ")
    dictionary_path = get_absolute_file_path("./data/mitre_attack_technique_dataset.csv")
    df = pd.read_csv(dictionary_path)

    ttps_dict = {}

    for index, row in df.iterrows():
        if dataset_supported_ttps:
            if row['ID'] not in dataset_supported_ttps:
                continue

        ttps_dict[row['ID']] = [[row['Name'], row['Description']]]

    return ttps_dict


def get_all_ttps(attack_pattern, embedding_cache, bert_model, ttps_dictionary, th=0.6):
    attack_pattern = remove_consec_newline(attack_pattern)
    attack_pattern = attack_pattern.replace('\t', ' ')
    attack_pattern = attack_pattern.replace("\'", "'")

    if len(attack_pattern) > 0:
        return get_relevant_ttp_ids(attack_pattern, embedding_cache, th, bert_model, ttps_dictionary)

    return {}


def annotate_dataset_with_Ladder(dataset_path,
                                 text_col,
                                 ground_truth_col,
                                 ignore_unsupported_labels: bool,
                                 ignore_predictions_not_supported_by_ground_truth: bool,
                                 destination_dir,
                                 entity_extraction_weight,
                                 sentence_classification_weight,
                                 distance_threshold,
                                 continue_prediction=True):
    # Example: python .\map_cves_to_ttps.py annotate_dataset_with_Ladder --dataset_path ".\data\nexus_comparison_test_dataset_2\nexus_test.csv" --text_col "CVE_Description" --ground_truth_col "Adjusted_Labels_Whole_Report" --destination_dir ".\data\nexus_comparison_test_dataset_2\ladder_predictions\" -entity_extraction_weight "models/entity_ext.pt" --sentence_classification_weight "models/sent_cls.pt" --distance_threshold "0.6" --ignore_predictions_not_supported_by_ground_truth "True" --ignore_unsupported_labels "True"
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

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

    dataset_df = pd.read_csv(dataset_path)
    current_dataset_header = dataset_df.columns.tolist()
    dataset_ground_truth_col = dataset_df[ground_truth_col].tolist()
    dataset_total_labels = []

    for ttp_list in dataset_ground_truth_col:
        ttp_list = ast.literal_eval(ttp_list)

        for ttp in ttp_list:
            if ttp not in dataset_total_labels:
                dataset_total_labels.append(ttp)

    if ignore_predictions_not_supported_by_ground_truth:
        ttps_dictionary = load_ttps_dictionary(dataset_total_labels)
    else:
        ttps_dictionary = load_ttps_dictionary()

    ladder_supported_ttps = ttps_dictionary.keys()

    for ttp in dataset_total_labels:
        if ttp not in ladder_supported_ttps:
            # print(f"{ttp} does not exit in Ladder's TTP Dictionary")
            print(f"WARNING: {ttp} does not exit in Ladder's TTP Dictionary (It will be ignored)")

    # Set the destination file path
    ladder_dataset_path = os.path.join(destination_dir, 'ladder_prediction_results.csv')

    # Continue from previous covered CVEs
    if continue_prediction and Path(ladder_dataset_path).is_file():
        covered_cve_dataset = pd.read_csv(ladder_dataset_path)
        covered_report_ids = list(covered_cve_dataset.iloc[:, 0])

        if covered_report_ids:
            print(f"The latest processed report ID was: {covered_report_ids[-1]}")

    # Define Header
    extended_dataset_header = ['Supported_Ground_truth_Ladder', 'Attack_Patterns', 'Ladder_predictions']
    current_dataset_header.extend(extended_dataset_header)
    extended_dataset_header = current_dataset_header

    # Create a new dataset
    new_dataset_df, covered_report_ids = (
        create_new_csv_dataset(destination_path=ladder_dataset_path,
                               headers=extended_dataset_header,
                               overwrite=not continue_prediction))

    embedding_cache = {}
    for index, row in dataset_df.iterrows():
        remained_reports = f'{index + 1}/{len(dataset_df)}'
        print(f'\rProcessing  Report {remained_reports} ',
              sep='', end='', flush=True)

        if continue_prediction and covered_report_ids:
            if row[0] in covered_report_ids:
                continue

        ground_truth = ast.literal_eval(row[ground_truth_col])
        new_ground_truth = []

        if ignore_unsupported_labels:
            for label in ground_truth:
                if label in ladder_supported_ttps:
                    new_ground_truth.append(label)
        else:
            new_ground_truth = ground_truth

        row['Supported_Ground_truth_Ladder'] = str(new_ground_truth)

        # Extract attack patterns from the target text
        cve_desc = row[text_col]
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
        ap_list = []
        ttps_below_threshold_list = []

        for attack_pattern in attack_patterns:
            relevant_ttps = get_all_ttps(attack_pattern, embedding_cache, bert_model, ttps_dictionary,
                                         distance_threshold)
            ap_list.append(attack_pattern)
            attack_pattern_closest_ttps_below_threshold = list(relevant_ttps['ttps_below_threshold'].keys())
            for ttp in attack_pattern_closest_ttps_below_threshold:
                if ttp not in ttps_below_threshold_list:
                    ttps_below_threshold_list.append(ttp)

        row['Attack_Patterns'] = str(ap_list)
        row['Ladder_predictions'] = str(ttps_below_threshold_list)

        # Save the prediction results
        data_list = [row.tolist()]
        save_csv_file(ladder_dataset_path, data_list)


def create_directory(dir_path):
    try:
        dir_path = get_absolute_file_path(dir_path)

        # Create the directory
        os.makedirs(dir_path, exist_ok=True)

        return dir_path
    except Exception as e:
        print(f"An error occurred: {e}")


def get_absolute_file_path(file) -> os.path:
    # Set PosixPath
    sys_platform = platform.system()
    if sys_platform == 'Linux':
        pathlib.PosixPath = pathlib.PosixPath
    else:
        pathlib.PosixPath = pathlib.WindowsPath

    # Normalize the path
    file = os.path.normpath(file)

    # Get the absolute path to the project's root directory
    project_root = get_project_root()

    return os.path.join(project_root, file)


def get_project_root() -> Path:
    return os.path.dirname(os.path.abspath(__file__))


def create_new_csv_dataset(destination_path: os.path, headers: list, data_list=None, overwrite: bool = False):
    if data_list is None:
        data_list = []
    covered_ids = []
    if not overwrite and Path(destination_path).is_file():
        print(f"The dataset already exist in {destination_path} and wont be overwritten.")

        new_dataset = pd.read_csv(destination_path)
        covered_ids = list(new_dataset.iloc[:, 0])

        return new_dataset, covered_ids
    else:
        if Path(destination_path).is_file():
            print("\nWARNING: A dataset already exist in {destination_path}")
            remove_dataset = input("Do you want to remove it? (Yes)")

            if remove_dataset:
                os.remove(destination_path)
                print(f"\nThe file {destination_path} has been removed successfully.")

        new_dataset = pd.DataFrame(data_list, columns=headers)
        new_dataset.to_csv(destination_path, index=False)
        print(f"\nA new dataset has been created in {destination_path}.")

        return new_dataset, covered_ids


fire.Fire({
    'annotate_dataset_with_Ladder': annotate_dataset_with_Ladder
})
