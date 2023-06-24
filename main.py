import torch
import toml
import glob

import utils
import infer
from train import Train

def main():
    config = toml.load("config.toml")
    if config['mode']:
        if config['train_audio_dir']["dir"] == "None":    
            config["train_config"]["audio_paths"] = config["train_audio_dir"]["paths"]
        else:
            train_audio_paths = sorted(glob.glob(config["train_audio_dir"]["dir"] + "/*.wav"))
            config["train_config"]["audio_paths"] = train_audio_paths
        train = Train(config['train_config'])
        train.run()
    else:
        if config['infer_audio_dir']["dir"] != "None":
            config['infer_audio_dir']['paths'] = sorted(glob.glob(config["infer_audio_dir"]["dir"] + "/*.wav"))
        res = infer.run(config['infer_audio_dir']['paths'])
        print(res)

if __name__ == "__main__":
    main()