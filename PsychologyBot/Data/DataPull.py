import os
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess

# Set up Kaggle API credentials

#Download Alphabet Dataset
home = os.path.expanduser('~')
os.environ["KAGGLE_CONFIG_DIR"] = home
cwd = os.getcwd()
save_dir = cwd+"/PsychologyBot/Data/AlphabetDataset/"
download="kaggle datasets download koushikchouhan/indian-sign-language-animated-videos"
subprocess.call(download,shell=True)
subprocess.call("mv indian-sign-language-animated-videos.zip "+save_dir,shell=True)
print(f"AlphabetDataset downloaded and saved to: {save_dir}")
unzip="unzip " +save_dir+"*.zip -d "+save_dir
subprocess.call(unzip,shell=True)
remove="rm "+save_dir+"*.zip"
subprocess.call(remove,shell=True)

#Download Conversation Dataset
dataset_name = "kaggle datasets download thedevastator/nlp-mental-health-conversations"
save_dir = cwd+"/PsychologyBot/Data/ChatDataset/"     # Replace with your desired save directory
subprocess.call(dataset_name,shell=True)
subprocess.call("mv nlp-mental-health-conversations.zip "+save_dir,shell=True)
print(f"ChatDataset downloaded and saved to: {save_dir}")
unzip="unzip " +save_dir+"*.zip"
subprocess.call(unzip,shell=True)
remove="rm "+save_dir+"*.zip"
subprocess.call(remove,shell=True)

#Download Categorical Dataset
script = cwd+"/PsychologyBot/Data/CategoricalDataset/download_data.sh" 
save_dir=cwd+"/PsychologyBot/Data/CategoricalDataset/"
subprocess.call("sh "+script,shell=True)
subprocess.call("mv *.zip "+save_dir,shell=True)
print(f"CategoricalDataset downloaded")
unzip="unzip " +save_dir+"*.zip"
subprocess.call(unzip,shell=True)
remove="rm "+save_dir+"*.zip"
subprocess.call(remove,shell=True)

directories = [d for d in os.listdir(save_dir) if os.path.isdir(d)]
subprocess.call("mkdir "+save_dir+"allvideos",shell=True)
os.chdir(save_dir)
for dir in directories:
    subprocess.call("mv "+dir+"/* "+save_dir+"allvideos/",shell=True )
os.chdir(save_dir+"allvideos/")
subprocess.call("for dir in */; do mv \"$dir\" \"${dir#*. }\"; done",shell=True )


