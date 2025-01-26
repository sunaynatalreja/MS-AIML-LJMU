import csv
import os
import json
import openai
from openai import OpenAI
from pathlib import Path
import time

prompt="You are Paul, a compassionate psychologist specializing in cognitive behavioral therapy. Engage naturally, using curiosity and positive regard. Ask clarifying, thought-provoking questions while exploring thoughts, feelings, behaviors, and life topics. Gently connect past and present, seeking user validation for insights. Never break character or end the session; always conclude with probing questions."
def prepare_conversation(row):
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": row['Context']},
            {"role": "assistant", "content": row['Response']},
        ]
    }

def csv_to_list_of_dicts(csv_file):
    data = []  
    with open(csv_file, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
        # Iterate over each row in the CSV file
        for row in csv_reader: 
            data.append(prepare_conversation(row))
            # data.append({'role': row['Context'], 
            #               'output': row['Response']})

    # Return the list of dictionaries
    return data  

def write_to_json(jsonfile,data):
    with open(jsonfile, "w") as out:
        for ddict in data:
            jout = json.dumps(ddict) + "\n"
            out.write(jout)

fine_tuning_data = csv_to_list_of_dicts('/Users/sunaynatalreja/PsychologyBotProjectFinal/Backend/Dataset/train.csv')
shape_80 = int(len(fine_tuning_data)*0.8)
fine_tuning_data_training=fine_tuning_data[:shape_80]
fine_tuning_data_validation=fine_tuning_data[shape_80:]

training_file_name = "fine_tuning_data_training.jsonl"
validation_file_name = "fine_tuning_data_validation.jsonl"
write_to_json(training_file_name,fine_tuning_data_training)
write_to_json(validation_file_name,fine_tuning_data_validation)