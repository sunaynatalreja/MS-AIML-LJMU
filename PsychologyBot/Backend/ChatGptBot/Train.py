import os
import openai
from openai import OpenAI
import time
import yaml

def upload_file(file_name: str, purpose: str, client) -> str:
    with open(file_name, "rb") as file_fd:
        response = client.files.create(file=file_fd, purpose=purpose)
    return response.id

def train_gpt(client,training_file_id,validation_file_id,MODEL):
    response = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model=MODEL,
        suffix="psychologist",
        hyperparameters={
        "n_epochs": 8,  # Increase for better learning
        "batch_size": 32,  # Adjust based on dataset size
        "learning_rate_multiplier": 5  # Lower to improve stability
        }
    )

    job_id = response.id

    print("Job ID:", response.id)
    print("Status:", response.status)

    response = client.fine_tuning.jobs.retrieve(job_id)

    print("Job ID:", response.id)
    print("Status:", response.status)
    print("Trained Tokens:", response.trained_tokens)
    #job_id='ftjob-VU1idsuZyqoxEkVy2EDnXi98'
    fine_tuned_model_id=None
    while fine_tuned_model_id==None:
        response = client.fine_tuning.jobs.list_events(job_id)

        events = response.data
        events.reverse()

        for event in events:
            print(str(event.data)+" "+event.message)

        response = client.fine_tuning.jobs.retrieve(job_id)
        fine_tuned_model_id = response.fine_tuned_model
        time.sleep(60)

    if fine_tuned_model_id is None:
        raise RuntimeError(
            "Fine-tuned model ID not found. Your job has likely not been completed yet."
        )

    print("Fine-tuned model ID:", fine_tuned_model_id)

def main():
    cwd = os.getcwd()
    cwd=cwd+"/PsychologyBot/Backend/ChatGptBot/"
    config_path = os.path.join(cwd,"../../../","PsychologyBot" ,"Config", "config.yaml")
    config = yaml.safe_load(open(config_path))
    model=config['BackendChatGptConfigs']['Model']
    openai.api_key=os.environ.get('OPENAI_API_KEY')
    client = OpenAI()
    cwd = os.getcwd()
    training_file_name = cwd+config['BackendChatGptConfigs']['FinalTrainingFileName']
    validation_file_name = cwd+config['BackendChatGptConfigs']['FinalValidationFileName']
    training_file_id = upload_file(training_file_name, "fine-tune",client)
    validation_file_id = upload_file(validation_file_name, "fine-tune",client)
    train_gpt(client,training_file_id,validation_file_id,model)

if __name__ == "__main__":
    main()