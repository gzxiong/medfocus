import json
import os
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
import sys
sys.path.append("../src")
from utils import safe_open_image

resize_width = 224
question_suffix = " Answer with a letter, word, phrase, or sentence."
direct_suffix = " Answer directly with yes or no without any explanation."
cot_suffix = " Think step by step and answer with yes or no."

def load_attribution_data(dataset_name: str, data_dir: str, split: str = 'test'):
    if dataset_name == "ImaGenome-Attr":
        return load_imagenome_attr_data(data_dir, split)
    elif dataset_name == "VinDR-CXR-Attr":
        return load_vindr_cxr_attr_data(data_dir, split)
    elif dataset_name == "PadChest-GR-Attr":
        return load_padchest_gr_attr_data(data_dir, split)

def load_imagenome_attr_data(data_dir: str, split: str = 'test'):
    df = pd.read_csv(os.path.join(data_dir, "chest-imagenome/1.0.0/gold_dataset/gold_object_attribute_with_coordinates.txt"), sep="\t")
    df = df[(df['relation'] == 1) & ((df['categoryID'] == 'anatomicalfinding') | (df['categoryID'] == 'disease'))].reset_index(drop=True)
    df = df[['patient_id', 'study_id', 'image_id', 'label_name', 'coord_original', 'bbox', 'categoryID']]
    df = df.groupby(['patient_id', 'study_id', 'image_id', 'label_name', 'categoryID']).agg({
        'coord_original': list,
        'bbox': list
    }).reset_index()

    # if not os.path.exists("download_train_imagenoe.sh"):
    #     with open("download_train_imagenoe.sh", "w") as f:
    #         f.write("#!/usr/bin/env bash\n")
    #         f.write("set -euo pipefail\n")
    #         for item in df[['patient_id', 'study_id']].drop_duplicates().to_dict('records'):
    #             patient_id = str(item['patient_id'])
    #             study_id = str(item['study_id'])
    #             f.write(
    #                 f'wget -r -N -c -np --user "$WGET_USER" --password "$WGET_PASS" '
    #                 f'"https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p{patient_id[:2]}/p{patient_id}/s{study_id}/"\n'
    #             )

    output_items = []
    for idx in range(len(df)):

        item = df.iloc[idx].to_dict()
        patient_id = str(item['patient_id'])
        study_id = str(item['study_id'])
        image_id = str(item['image_id'])
        imgpath = f"mimic-cxr-jpg/2.1.0/files/p{patient_id[:2]}/p{patient_id}/s{study_id}/{image_id.replace('.dcm', '.jpg')}"
        
        question = f"Is there evidence of {item['label_name']} in the image?"
        answer = "Yes"
        reason = str(item['bbox'])
        locations = [eval(s) for s in item['coord_original']]

        output_items.append(
            {
                "index": idx,
                "imgpath": os.path.join(data_dir, imgpath),
                "question": question + question_suffix,
                "question_direct": question + direct_suffix,
                "question_cot": question + cot_suffix,
                "answer": answer,
                "reason": reason,
                "locations": locations,
            }
        )
    return output_items

def load_vindr_cxr_attr_data(data_dir: str, split: str = 'test'):

    df = pd.read_csv(os.path.join(data_dir, "vindr-cxr/1.0.0/annotations/annotations_test.csv"))
    image_ids = df["image_id"].drop_duplicates().tolist()
    df = df[df["image_id"].isin(image_ids) & (df['class_name'] != 'No finding')]
    df['locations'] = df.apply(lambda row: [round(row['x_min']), round(row['y_min']), round(row['x_max']), round(row['y_max'])], axis=1)
    df = df.groupby(['image_id', 'class_name']).agg({
        'locations': list,
    }).reset_index()

    # if not os.path.exists("download_train_vindr-cxr.sh"):
    #     with open("download_train_vindr-cxr.sh", "w") as f:
    #         f.write("#!/usr/bin/env bash\n")
    #         f.write("set -euo pipefail\n")
    #         for image_id in image_ids:
    #             f.write(
    #                 f'wget -r -N -c -np --user "$WGET_USER" --password "$WGET_PASS" '
    #                 f'"https://physionet.org/files/vindr-cxr/1.0.0/test/{image_id}.dicom"\n'
    #             )

    output_items = []
    for idx in range(len(df)):
        item = df.iloc[idx].to_dict()
        image_id = item['image_id']
        imgpath = f"vindr-cxr/1.0.0/test/{image_id}.dicom"
        
        question = f"Is there evidence of {item['class_name']} in the image?"
        answer = "Yes"
        reason = None
        locations = item['locations']
        output_items.append(
            {
                "index": idx,
                "imgpath": os.path.join(data_dir, imgpath),
                "question": question + question_suffix,
                "question_direct": question + direct_suffix,
                "question_cot": question + cot_suffix,
                "answer": answer,
                "reason": reason,
                "locations": locations,
            }
        )
    return output_items

def load_padchest_gr_attr_data(data_dir: str, split: str = 'test'):
    
    data = json.load(open(os.path.join(data_dir, 'filtered_studies.json')))
    

    if split == 'test':
        num_samples = 2000
        save_path = os.path.join(data_dir, f'filtered_data_{num_samples}.json')
        data_selected = data[:num_samples]
    elif split == 'train':
        num_samples = 100
        save_path = os.path.join(data_dir, f'filtered_data_train_{num_samples}.json')
        data_selected = data[-num_samples:][::-1]
    else:
        raise ValueError("split must be 'train' or 'test'")

    if not os.path.exists(save_path):
        items = []
        for item in tqdm(data_selected):
            imgpath = 'PadChest_GR/' + item['ImageID']
            img = None
            for q_idx in range(len(item['findings'])):
                q_item = item['findings'][q_idx]
                if not q_item['abnormal']:
                    continue
                if len(q_item['boxes']) == 0 and len(q_item['extra_boxes']) == 0:
                    continue
                if img is None:
                    img = safe_open_image(os.path.join(data_dir, imgpath))
                question = f"Is there evidence of {' or '.join(q_item['labels'])} in the image?"
                answer = "Yes"
                reason = str(q_item['locations'])
                locations = [
                    [
                        round(loc[0] * img.size[0]), 
                        round(loc[1] * img.size[1]), 
                        round(loc[2] * img.size[0]), 
                        round(loc[3] * img.size[1])
                    ] for loc in q_item['boxes'] + q_item['extra_boxes']]
                items.append({
                    'imgpath': imgpath,
                    'question': question,
                    'answer': answer,
                    'reason': reason,
                    'locations': locations
                })
        json.dump(items, open(save_path, 'w'), indent=4)
    else:
        items = json.load(open(save_path))

    for idx in range(len(items)):
        items[idx]['index'] = idx
        items[idx]['imgpath'] = os.path.join(data_dir, items[idx]['imgpath'])
        items[idx]['question'] = items[idx]['question'] + question_suffix
        items[idx]['question_direct'] = items[idx]['question'] + direct_suffix
        items[idx]['question_cot'] = items[idx]['question'] + cot_suffix
    return items