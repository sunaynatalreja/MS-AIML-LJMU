import os
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess
import yaml

def main():
    cwd = os.getcwd()
    config_path = os.path.join(cwd,"PsychologyBot" ,"Config", "config.yaml")
    config = yaml.safe_load(open(config_path))
    dataset_path=config['BackendChatGptConfigs']['ChatDataset']
    save_dir=config['BackendChatGptConfigs']['ChatDataSaveDir']
    downloaded_dataset=config['BackendChatGptConfigs']['ChatDatasetDownloadFile']
    #Download Conversation Dataset
    dataset_name = "kaggle datasets download "+dataset_path
    save_dir = cwd+save_dir
    subprocess.call("mkdir "+save_dir,shell=True)
    subprocess.call(dataset_name,shell=True)
    subprocess.call("mv "+downloaded_dataset+ " "+save_dir,shell=True)
    print(f"ChatDataset downloaded and saved to: {save_dir}")
    os.chdir(save_dir)
    unzip="unzip " +save_dir+"*.zip"
    subprocess.call(unzip,shell=True)
    remove="rm "+save_dir+"*.zip"
    subprocess.call(remove,shell=True)

if __name__ == "__main__":
    main()