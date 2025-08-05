import os
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess


home = os.path.expanduser('~')
os.environ["KAGGLE_CONFIG_DIR"] = home
cwd = os.getcwd()
save_dir = cwd+"/PsychologyBot/Data/AlphabetDataset/"
subprocess.call("mkdir "+save_dir,shell=True)
download="kaggle datasets download koushikchouhan/indian-sign-language-animated-videos"
subprocess.call(download,shell=True)
subprocess.call("mv indian-sign-language-animated-videos.zip "+save_dir,shell=True)
print(f"AlphabetDataset downloaded and saved to: {save_dir}")
unzip="unzip " +save_dir+"*.zip -d "+save_dir
subprocess.call(unzip,shell=True)
remove="rm "+save_dir+"*.zip"
subprocess.call(remove,shell=True)


dataset_name = "kaggle datasets download thedevastator/nlp-mental-health-conversations"
save_dir = cwd+"/PsychologyBot/Data/ChatDataset/"   
subprocess.call("mkdir "+save_dir,shell=True)
subprocess.call(dataset_name,shell=True)
subprocess.call("mv nlp-mental-health-conversations.zip "+save_dir,shell=True)
print(f"ChatDataset downloaded and saved to: {save_dir}")
os.chdir(save_dir)
unzip="unzip " +save_dir+"*.zip"
subprocess.call(unzip,shell=True)
remove="rm "+save_dir+"*.zip"
subprocess.call(remove,shell=True)


target_dir = save_dir


for root, dirs, files in os.walk(target_dir, topdown=False):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        if not os.listdir(dir_path): 
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")

