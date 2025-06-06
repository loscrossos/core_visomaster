# <div style="font-family:'Arial', sans-serif;font-size: 3em;font-weight: bold;background: linear-gradient(45deg, #FFD700, #FF8C00);-webkit-background-clip: text;background-clip: text;color: transparent;text-align: center;padding: 20px;text-shadow: 2px 2px 4px rgba(0,0,0,0.2);border-radius: 10px;"><span style="color: #333;">core</span><span style="background: linear-gradient(45deg, #AAAAAA, #FF45FF);-webkit-background-clip: text;background-clip: text;color: transparent;">VisoMaster</span></div>



VisoMaster is a tool for face swapping and editing in images and videos. 

This project does not aim at more functionality. *It hardens the core.*

---

## Features  

- **Face Swap**
  - Supports multiple face swapper models  
  - Compatible with DeepFaceLab trained models (DFM)  
  - Advanced multi-face swapping with masking options for each facial part  
  - Occlusion masking support (DFL XSeg Masking)  
  - Works with all popular face detectors & landmark detectors  
  - Expression Restorer: Transfers original expressions to the swapped face  
  - Face Restoration: Supports all popular upscaling & enhancement models  
- **Face Editor (LivePortrait Models)**  
  - Manually adjust expressions and poses for different face parts  
  - Fine-tune colors for Face, Hair, Eyebrows, and Lips using RGB adjustments  
- **Other Powerful Features**  
  - **Live Playback**: See processed video in real-time before saving  
  - **Face Embeddings**: Use multiple source faces for better accuracy & similarity  
  - **Live Swapping via Webcam**: Stream to virtual camera for Twitch, YouTube, Zoom, etc.  
  - **User-Friendly Interface**: Intuitive and easy to use  
  - **Video Markers**: Adjust settings per frame for precise results  
  - **TensorRT Support**: Leverages supported GPUs for ultra-fast processing  
  - **Many More Advanced Features** 

- **core hardened extra features:**
  - Works on Windows and Linux.
  - Full support for all CUDA cards (yes, RTX 50 series Blackwell too)
  - Automatic model download and model self-repair (redownloads damaged files)
  - Configurable Model placement: retrieves the models from anywhere you stored them.
  - efficient unified Cross-OS install


# Installation 

 
The installation in general consists of:

- Pre-Requisites: Check that your system can actually run the model
- Project Installation. It consists of 
    - cloning the repository
    - creating and activating a virtual environment
    - installing the requirements
    - getting the models (optionally re-using existing models)
    - starting the app.


## TLDR Installation

- You need ffmpeg installed before using the app

**Windows**
```
git clone https://github.com/loscrossos/core_visomaster
cd core_visomaster

py -3.10 -m venv .env_win
.env_win\Scripts\activate

pip install -r requirements.txt
```

**Linux**
```
git clone https://github.com/loscrossos/core_visomaster
cd core_visomaster

python3.10 -m venv .env_lin
. ./.env_lin/bin/activate

pip install -r requirements.txt
```

**All OSes**
You can use one of these optional steps (detailed steps below):
- **Option 1**: automatic model download: just go to the next step and start the app!
- **Option 2**: reuse your models without changing their paths: run  `python appvisomaster.py --checkmodels` after install to generate `configmodels.txt` and edit the paths within the file. run the command again to verify it worked.
- **Option 3**: force model integrity check and downlaod missing: run `python appvisomaster.py --integritycheck`
**Run the app**


Whenever you want to start the apps open a console in the repository directory, activate your virtual environment:

```
Windows:
.env_win\Scripts\activate
Linux:
. ./.env_lin/bin/activate
```


start the app with:

`python appvisomaster.py`


Stop the app pressing `ctrl + c` on the console




## Pre-Requisites

In general you should have your PC setup for AI development when trying out AI models, LLMs and the likes. If you have some experience in this area, you likely already fulfill most if not all of these items. visomaster has however light requirements on the hardware.


### Hardware requirements



**Installation requirements**

This seem the minimum hardware requirements:


Hardware    | **Mac** | **Win/Lin**
---         | ---     | ---
CPU         | n.a.    | Will not be used much. So any modern CPU should do
VRAM        | n.a.    | Uses 4GB VRAM during generation
RAM         | n.a.    | Uses some 2GB RAM (peak) during generation
Disk Space  | n.a.    | 11GB for the models






### Software requirements

**Requirements**

You should have the following setup to run this project:

- Python 3.10
- latest GPU drivers
- latest cuda-toolkit 12.8+ (for nvidia 50 series support)
- Ffmpeg installed and configured
- Linux:
    - jpg headers:
      - install with:   sudo apt install libjpeg-dev zlib1g-dev

I am not using Conda but the original Free Open Source Python. This guide assumes you use that.

**Automated Software development setup**

If you want a fully free and open source, no strings attached, automated, beginner friendly but efficient way to setup a software development environment for AI and Python, you can use my other project: CrossOS_Setup, which setups your Mac, Windows or Linux PC automatically to a full fledged AI Software Development station. It includes a system checker to assess how well installed your current setup is, before you install anything:

https://github.com/loscrossos/crossos_setup

Thats what i use for all my development across all my systems. All my projects run out of the box if you PC is setup with it.


## Project Installation

If you setup your development environment using my `Crossos_Setup` project, you can do this from a normal non-admin account (which you should actually be doing anyway for your own security).

Hint: "CrossOS" means the commands are valid on MacWinLin

 ---

Lets install core_visomaster in 5 Lines on all OSes, shall we? Just open a terminal and enter the commands.



1. Clone the repo (CrossOS): 
```
git clone https://github.com/loscrossos/core_visomaster
cd core_visomaster
```

2. Create and activate a python virtual environment  

task       |  Windows                   | Linux
---        |  ---                       | ---
create venv|`py -3.10 -m venv .env_win`|`python3.10 -m venv .env_lin`
activate it|`.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`

At this point you should see at the left of your prompt the name of your environment (e.g. `(.env_win)`)


3. Install the libraries (CrossOS):
```
pip install -r requirements.txt
```

Thats it.

---

At this point you *could* just start the apps and start generating away... but it would first automatically download the models (11GB of them). If you dont have the models yet thats ok. But if you have already downloaded them OR if you have a dual/trial/multiboot machine and want to make them portable, read on...


## Model Installation

The needed models are about 6GB in total. You can get them in 3 ways:
- **Automatic Download** as huggingface cache (easiest way)
- **Re-use existing models**: hf_download or manual
- **Integrity check**: The app self checks its models and repairs broken models.

to see the status of the model recognition start any app with the parameter `--checkmodels`

e.g. `python appstudio.py --checkmodels`
The app will report the models it sees and quit without downloading or loading anything.


### Automatic download
just start the app. 

Missing models will be downloaded. This is for when you never had the app installed before. The models will be downloaded to a huggingface-type folder in the "models" directory. This is ok if you want the most easy solution and dont care about portability (which is ok!). This is not reccomended as its not very reusable for software developers: e.g. if you want to do coding against the models from another project or want to store the models later. This supports multi-boot.



### Re-use existing models


You can re-use your existing models by configuring the path in the configuration file `modelconfig.txt`.
This file is created when you first start any app. Just call e.g. `python appstudio.py --checkmodels` to create it.
Now open it with any text editor and put in the path of the directory that points to your models. 
You can use absolute or relative paths. If you have a multiboot-Setup (e.g. dualboot Windows/Linux) you should use relative paths with forward slashes e.g. `../mydir/example`

There are 2 types of model downloads: the hugginface (hf) cache and manual model download.

### Integrity check

If you think some models are damaged you can force a model integrity check and download missing by running: `python appvisomaster.py --integritycheck`


**Checking that the models are correctly configured**

You can easily check that the app sees the models by starting any of the demos with the parameter `--checkmodels` and checking the last line.

e.g. `python appstudio.py --checkmodels`

```
[!FOUND!]: /Users/Shared/github/core_projectexample/models/somemodel/
[!FOUND!]: /Users/Shared/github/core_projectexample/models/someothermodel/
[!FOUND!]: /Users/Shared/github/core_projectexample/models/modeltoo/
----------------------------
FINAL RESULT: It seems all model directories were found. Nothing will be downloaded!
```

# Usage 

You can use app as you always have. Just start the app and be creative!

## Starting the Apps


The app has the following name:

- `appvisomaster.py`

To start just open a terminal, change to the repository directory, enable the virtual environment and start the app. The `--inbrowser` option will automatically open a browser with the UI.

task         |  Windows                   | Linux
---          |  ---                       | ---
activate venv| `.env_win\Scripts\activate`|`. ./.env_lin/bin/activate`


for example (CrossOS)
```
python appvisomaster.py
```

A browser should pop up with the UI


To stop the app press `ctrl-c` on the console (CrossOS)






# Benchmark

not done as the app runs fast on pretty much anything.

# Known Issues
Documentation of Issues i encountered and know of.

 









## **Troubleshooting**
- If you face CUDA-related issues, ensure your GPU drivers are up to date.
- For missing models, double-check that all models are placed in the correct directories.


- if you have problems getting the softwar to run and you open an issue it is mandatory to include the output of 
```
python appvisomaster.py --sysreport
```

## Credits

The original project can be found at:

https://github.com/visomaster/VisoMaster

There you can find the original authors

## Disclaimer: ##

(From original authors)

**VisoMaster** is a hobby project that we are making available to the community as a thank you to all of the contributors ahead of us.
We've copied the disclaimer from [Swap-Mukham](https://github.com/harisreedhar/Swap-Mukham) here since it is well-written and applies 100% to this repo.
 
We would like to emphasize that our swapping software is intended for responsible and ethical use only. We must stress that users are solely responsible for their actions when using our software.

Intended Usage: This software is designed to assist users in creating realistic and entertaining content, such as movies, visual effects, virtual reality experiences, and other creative applications. We encourage users to explore these possibilities within the boundaries of legality, ethical considerations, and respect for others' privacy.

Ethical Guidelines: Users are expected to adhere to a set of ethical guidelines when using our software. These guidelines include, but are not limited to:

Not creating or sharing content that could harm, defame, or harass individuals. Obtaining proper consent and permissions from individuals featured in the content before using their likeness. Avoiding the use of this technology for deceptive purposes, including misinformation or malicious intent. Respecting and abiding by applicable laws, regulations, and copyright restrictions.

Privacy and Consent: Users are responsible for ensuring that they have the necessary permissions and consents from individuals whose likeness they intend to use in their creations. We strongly discourage the creation of content without explicit consent, particularly if it involves non-consensual or private content. It is essential to respect the privacy and dignity of all individuals involved.

Legal Considerations: Users must understand and comply with all relevant local, regional, and international laws pertaining to this technology. This includes laws related to privacy, defamation, intellectual property rights, and other relevant legislation. Users should consult legal professionals if they have any doubts regarding the legal implications of their creations.

Liability and Responsibility: We, as the creators and providers of the deep fake software, cannot be held responsible for the actions or consequences resulting from the usage of our software. Users assume full liability and responsibility for any misuse, unintended effects, or abusive behavior associated with the content they create.

By using this software, users acknowledge that they have read, understood, and agreed to abide by the above guidelines and disclaimers. We strongly encourage users to approach this technology with caution, integrity, and respect for the well-being and rights of others.

Remember, technology should be used to empower and inspire, not to harm or deceive. Let's strive for ethical and responsible use of deep fake technology for the betterment of society.
