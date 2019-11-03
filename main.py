from pathlib import Path

data_folder = Path("data/")

dev_data = data_folder/"pos-dev"

if not dev_data.exists():
    print("Oops, file doesn't exist!")
else:
    print("Yay, the file exists!")
