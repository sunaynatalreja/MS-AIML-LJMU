import csv
import os
import json
from langdetect import detect
from googletrans import Translator
import re
import asyncio
import contractions
import unicodedata
import ast
import random
import subprocess
import yaml

def prepare_conversation_for_openai_preprocessing(row):
    return {'prompt': row['Context'], 'completion': row['Response']}

def clean_text(text):
    text = text.lower().strip()  # Lowercase for consistency
    text = re.sub(r'\xa0+', ' ', text)  # Remove non-breaking spaces
    text = re.sub(r'[-]{10,}', ' ', text)  # Remove long dashes
    text = re.sub(r'\n+', ' ', text)  # Remove newlines
    text = re.sub(r'\d{3}-\d{3}-\d{4}', '', text)  # Remove phone numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def expand_text(text):
    return contractions.fix(text)

async def translate_non_english(text):
    translator = Translator()
    sentences = text.split(". ")  # Split into sentences
    translated_text = []
    translation=""
    for sentence in sentences:  
        lang = detect(sentence)  # Detect language
        if lang != "en":
            translation = await translator.translate(sentence, src=lang, dest="en")
            translation = expand_text(translation.text)  # Extract translated text
        translated_text.append(translation)

    return ". ".join(translated_text)

def csv_to_list_of_dicts(csv_file):
    data = []  
    translator = Translator()
    with open(csv_file, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.DictReader(file)
        # Iterate over each row in the CSV file
        for row in csv_reader: 
            if not (row['Response']=='' or row['Context']=='') and len(row['Response'])>1:
                row['Response'] = expand_text(row['Response'])
                row['Context'] = expand_text(row['Context'])
                if not is_english(row['Response']):
                    translated_response = asyncio.run(translate_non_english(clean_text(row['Response'])))
                    row['Response'] = translated_response
                if not is_english(row['Context']):
                    translated_text = asyncio.run(translate_non_english(clean_text(row['Context'])))
                    row['Context'] = translated_text
                data.append(prepare_conversation_for_openai_preprocessing(row))
    # Return the list of dictionaries
    return data  

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def normalise_text(text):
    normalized_text = unicodedata.normalize('NFKC', str(text)).encode('ascii','ignore').decode()
    normalized_text = normalized_text.replace('\\xa0','')
    return normalized_text

def write_to_json(jsonfile,data):
    with open(jsonfile, "w") as out:
        for ddict in data:
            jout = json.dumps(ast.literal_eval(normalise_text(ddict))) + "\n"
            out.write(jout)

def openai_prepare_data(file_path):
    command = "openai tools fine_tunes.prepare_data -f "+ file_path +" --quiet"
    subprocess.call(command, shell=True)

def prepare_conversation(row,prompt):
    return {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": row['prompt']},
            {"role": "assistant", "content": row['completion']},
        ]
    }

def get_data(file_name,prompt):
    final_data=[]
    with open(file_name, "r") as f:
        raw_data = [json.loads(line) for line in f]
        for item in raw_data:
            formatted_item=prepare_conversation(item,prompt)
            final_data.append(formatted_item)
    return final_data

def main():
    config_path = os.path.join(cwd,"PsychologyBot" ,"Config", "config.yaml")
    config = yaml.safe_load(open(config_path))
    prompt=config['BackendChatGptConfigs']['Prompt']
    cwd = os.getcwd()
    fine_tuning_data = csv_to_list_of_dicts(cwd+config['BackendChatGptConfigs']['CsvFilePath'])
    random.shuffle(fine_tuning_data)
    shape_80 = int(len(fine_tuning_data)*0.8)
    fine_tuning_data_training=fine_tuning_data[:shape_80]
    fine_tuning_data_validation=fine_tuning_data[shape_80:]
    training_file_name = cwd+config['BackendChatGptConfigs']['TrainingFileName']
    validation_file_name = cwd+config['BackendChatGptConfigs']['ValidationFileName']
    write_to_json(training_file_name,fine_tuning_data_training)
    write_to_json(validation_file_name,fine_tuning_data_validation)
    openai_prepare_data(training_file_name)
    openai_prepare_data(validation_file_name)
    training_file_after_openai_prepare = cwd+config['BackendChatGptConfigs']['TrainingFileAfterOpenaiPrepare']
    validation_file_after_openai_prepare = cwd+config['BackendChatGptConfigs']['ValidationFileAfterOpenaiPrepare']
    train_data=get_data(training_file_after_openai_prepare,prompt)
    validation_data=get_data(validation_file_after_openai_prepare,prompt)
    final_training_file_name = cwd+config['BackendChatGptConfigs']['FinalTrainingFileName']
    final_validation_file_name = cwd+config['BackendChatGptConfigs']['FinalValidationFileName']
    write_to_json(final_training_file_name,train_data)
    write_to_json(final_validation_file_name,validation_data)
    

if __name__ == "__main__":
    main()