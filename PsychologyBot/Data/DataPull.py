import os
from kaggle.api.kaggle_api_extended import KaggleApi
import subprocess

# Set up Kaggle API credentials

#Download Alphabet Dataset
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

#Download Conversation Dataset
dataset_name = "kaggle datasets download thedevastator/nlp-mental-health-conversations"
save_dir = cwd+"/PsychologyBot/Data/ChatDataset/"     # Replace with your desired save directory
subprocess.call("mkdir "+save_dir,shell=True)
subprocess.call(dataset_name,shell=True)
subprocess.call("mv nlp-mental-health-conversations.zip "+save_dir,shell=True)
print(f"ChatDataset downloaded and saved to: {save_dir}")
os.chdir(save_dir)
unzip="unzip " +save_dir+"*.zip"
subprocess.call(unzip,shell=True)
remove="rm "+save_dir+"*.zip"
subprocess.call(remove,shell=True)

# #Download Categorical Dataset
script = cwd+"/PsychologyBot/Data/CategoricalDataset/download_data.sh" 
save_dir=cwd+"/PsychologyBot/Data/CategoricalDataset/"
subprocess.call("sh "+script,shell=True)
subprocess.call("mv *.zip "+save_dir,shell=True)
print(f"CategoricalDataset downloaded")
os.chdir(save_dir)
subprocess.call("mkdir "+save_dir+"allvideos",shell=True)
unzip="unzip " +save_dir+"*.zip"
for file in os.listdir(save_dir):
    if file.endswith(".zip"):  # Check if the file is a ZIP file
        subprocess.call("unzip " +save_dir+file,shell=True)
        directories = [d for d in os.listdir(save_dir) if os.path.isdir(d)]
        for dir in directories:
            if dir!="allvideos":
                subprocess.call("mv "+dir+"/* "+save_dir+"allvideos/",shell=True )

remove="rm "+save_dir+"*.zip"
subprocess.call(remove,shell=True)
os.chdir(save_dir+"allvideos/")
subprocess.call("for dir in */; do mv \"$dir\" \"${dir#*. }\"; done",shell=True )

# Specify the directory to clean up
target_dir = save_dir

# Walk through the directory tree and remove empty directories
for root, dirs, files in os.walk(target_dir, topdown=False):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        if not os.listdir(dir_path):  # Check if the directory is empty
            os.rmdir(dir_path)
            print(f"Removed empty directory: {dir_path}")

