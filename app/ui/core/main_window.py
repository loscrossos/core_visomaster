

# Input data
in_dotenv_needed_models = {
}

in_dotenv_needed_paths = {
    "MODELS_HOME": "./model_assets",
}


in_dotenv_needed_params = {
    "DEBUG_MODE": False,
}



#LCX1.03##################################################################
#FILELOADER##############################################################
#########################################################################
debug_mode=False
LCX_APP_NAME="CROSSOS_FILE_CHECK"
in_model_config_file="configmodel.txt"
# --- Helper Functions ---
#dotenv prefixes
PREFIX_MODEL="PATH_MODEL_"
PREFIX_PATH="PATH_NEEDED_"
LOG_PREFIX="CROSSOS_LOG"
import re
import os 
from pathlib import Path
from typing import Dict, Set, Any, Union
def model_to_varname(model_path: str, prefix: str) -> str:
    """Converts a model path to a dotenv-compatible variable name"""
    model_name = model_path.split("/")[-1]
    varname = re.sub(r"[^a-zA-Z0-9]", "_", model_name.upper())
    return f"{prefix}{varname}"

def varname_to_model(varname: str, prefix: str) -> str:
    """Converts a variable name back to original model path format"""
    if varname.startswith("PATH_MODEL_"):
        model_part = varname[prefix.len():].lower().replace("_", "-")
        return f"Zyphra/{model_part}"
    return ""

def read_existing_config(file_path: str) -> Dict[str, str]:
    """Reads existing config file and returns key-value pairs"""
    existing = {}
    path = Path(file_path)
    if path.exists():
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        existing[parts[0].strip()] = parts[1].strip()
    else:
        print(f"{LCX_APP_NAME}: ERROR config file not found: {file_path}")
    if debug_mode:
        print(f"{LCX_APP_NAME}: found config file: {file_path}")
    return existing

def update_model_paths_file(
    models: Set[str],
    paths: Dict[str, str],
    params: Dict[str, Any],
    file_path: str 
) -> None:
    """Updates config file, adding only new variables"""
    existing = read_existing_config(file_path)
    new_lines = []
    
    # Process models
    for model in models:
        varname = model_to_varname(model, PREFIX_MODEL)
        if varname not in existing:
            print(f"{LOG_PREFIX}: Adding Model rquirement to config: {model}")
            new_lines.append(f"{varname} = ./models/{model.split('/')[-1]}")
    
    # Process paths - now handles any path keys
    for key, value in paths.items():
        varname = model_to_varname(key, PREFIX_PATH)
        if varname not in existing:
            print(f"{LOG_PREFIX}: Adding path rquirement to config: {key}")
            new_lines.append(f"{varname} = {value}")
    
    # Process params
    for key, value in params.items():
        if key not in existing:
            print(f"{LOG_PREFIX}: Adding Parameter rquirement to config: {key}")
            new_lines.append(f"{key} = {value}")
    
    # Append new lines if any
    if new_lines:
        with open(file_path, 'a') as f:
            f.write("\n" + "\n".join(new_lines) + "\n")

def parse_model_paths_file(file_path: str , dotenv_needed_models, dotenv_needed_paths ) -> tuple[
    Set[str], Dict[str, str], Dict[str, Union[bool, int, float, str]]
]:
    """Reads config file and returns loaded variables"""
    loaded_models = {}
    loaded_paths = {}
    loaded_params = {}
    loaded_models_values= {}
    existing = read_existing_config(file_path)
    
    for key, value in existing.items():
        # Handle model paths
        if key.startswith(PREFIX_MODEL):
            for mod in dotenv_needed_models:
                #we find out if the current key value belongs to one of our models
                if key == model_to_varname(mod,PREFIX_MODEL):
                    #if a path has been defined and it exists we use the local path
                    if value and os.path.isdir(value):
                        loaded_models[mod] = value
                    else:
                        #else we use the model id so its downloaded from github later
                        loaded_models[mod] = mod
                    #still we collect the values to show to the user so he knows what to fix in config file
                    loaded_models_values[mod] = value
        # Handle ALL paths (not just HF_HOME)
        elif key.startswith(PREFIX_PATH):
            for mod in dotenv_needed_paths:
                if key == model_to_varname(mod,PREFIX_PATH):
                    loaded_paths[mod] = value
        # Handle params with type conversion
        else:
            if value.lower() in {"true", "false"}:
                loaded_params[key] = value.lower() == "true"
            elif value.isdigit():
                loaded_params[key] = int(value)
            else:
                try:
                    loaded_params[key] = float(value)
                except ValueError:
                    loaded_params[key] = value
    
    return loaded_models, loaded_paths, loaded_params, loaded_models_values

def is_online_model(model: str,dotenv_needed_models, debug_mode: bool = False) -> bool:
    """Checks if a model is in the online models set."""
    is_onlinemodel = model in dotenv_needed_models
    if debug_mode:
        print(f"Model '{model}' is online: {is_onlinemodel}")
    return is_onlinemodel

import os
def count_existing_paths(paths):
    """
    Checks if each path in the list exists.
    Returns:
        - summary (str): Summary of found/missing count
        - all_found (bool): True if all paths were found
        - none_found (bool): True if no paths were found
        - details (list of str): List with "[found]" or "[not found]" per path
    """
    total = len(paths)
    if total == 0:
        return "No paths provided.", False, True, []
    found_count = 0
    details = []
    for path in paths:
        if os.path.exists(path):
            found_count += 1
            details.append(f"[!FOUND!]: {path}")
        else:
            details.append(f"[MISSING]: {path}")
    missing_count = total - found_count
    all_found = (missing_count == 0)
    none_found = (found_count == 0)
    summary = f"Found {found_count}, missing {missing_count}, out of {total} paths."
    return summary, all_found, none_found, details


def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text


def get_hf_model_cache_dirname(model_id: str) -> str:
    """
    Returns the HF cache directory name for a given model.
    """
    base = "models--"
    return base + model_id.replace('/', '--')

def check_do_all_files_exist(dotenv_needed_models,dotenv_loaded_paths, dotenv_loaded_models, dotenv_loaded_models_values, in_files_to_check_in_paths=None, silent=False  ):
    test_models_hf = []
    test_models_dir=[]
    test_paths_dir=[]
    
    retval_models_exist=True
    retval_paths_exist=True
    
    #add model paths as path and as hf cache path
    for currmodel in dotenv_needed_models:
        test_models_hf.append(f"{dotenv_loaded_paths['HF_HOME']}{os.sep}hub{os.sep}{get_hf_model_cache_dirname(currmodel)}{os.sep}snapshots")
        test_models_dir.append(f"{dotenv_loaded_models[  currmodel]}")
    
    #add needed dirs as path
    for curr_path in dotenv_loaded_paths:
        test_paths_dir.append(f"{dotenv_loaded_paths[  curr_path]}")
    
    if debug_mode:
        print(f"test pathf hf: {test_models_hf}")
        print(f"test pathf dirs: {test_models_dir}")
    
    if not silent:
        print(f"{LCX_APP_NAME}: checking model accessibility")
    summary_hf, all_exist_hf, none_exist_hf, path_details_hf = count_existing_paths(test_models_hf)

    if not silent:
        print(f"\n-Searching Group1: Model HF_HOME----------------------------------------------")
        for line in path_details_hf:
            print_line= remove_suffix(line, "snapshots")
            print(print_line)

    summary_dir, all_exist_dir, none_exist_dir, path_details_dir = count_existing_paths(test_models_dir)
    if not silent:
        print("-Searching Group2: Manual Model Directories-----------------------------------")
        for line in path_details_dir:
            print_line= remove_suffix(line, "model_index.json")
            print_line= remove_suffix(print_line, "config.json")
            print(print_line)

    summary_path, all_exist_path, none_exist_path, path_details_path = count_existing_paths(test_paths_dir)
    if not silent:
        print("-Searching Group3: Needed Directories-----------------------------------------")
        for line in path_details_path:
            print(line)
            
    if not silent:
        print("-checking explicite Files---------------------------------------------------")

    for mapping in in_files_to_check_in_paths:
        for env_var, relative_path in mapping.items():
            if dotenv_loaded_paths and env_var in dotenv_loaded_paths:
                base_path = dotenv_loaded_paths[env_var]
                full_path = Path(base_path) / relative_path.strip(os.sep)
                if full_path.exists():
                    if not silent:
                        print(f"[!FOUND!]: {full_path}")
                else:
                    if not silent:
                        print(f"[!MISSING!]: {full_path}")
                    retval_paths_exist = False
    if not silent:
        print("")
    #we show the dir values to the user
    if not silent:
        if all_exist_dir==False:
            print("-Values in config (resolved to your OS)---------------------------------------")
            for key in dotenv_loaded_models_values:
                print(f"{key}: {os.path.abspath(dotenv_loaded_models_values[key])}")
        if all_exist_path==False:
            for key in dotenv_loaded_paths:
                print(f"{key}: {os.path.abspath(dotenv_loaded_paths[  key])}")
    if not silent:
        print("")
    
    #Needed Dirs summary
    if in_dotenv_needed_paths and not silent:
        print("-Needed Paths---------------------------------------------------")     
    if in_dotenv_needed_paths and all_exist_path == False:
        if not silent:
            print("Not all paths were found. Check documentation if you need them")
        retval_paths_exist=False
    if not silent:
        if in_dotenv_needed_paths and all_exist_path:
            print("All Needed PATHS exist.")
    if in_dotenv_needed_models:
        if not silent:
            print("-Needed Models--------------------------------------------------")
        #some model directories were missing 
            if none_exist_dir == False and all_exist_dir == False: 
                print ("Some manually downloaded models were found. Some might need to be downloaded!")
            #some hf cache models were missing
            if  all_exist_hf == False and none_exist_hf==False:
                print ("Some HF_Download models were found. Some might need to be downloaded!")
            if none_exist_dir and none_exist_hf:
                print ("No models were found! Models will be downloaded at next app start")

            if all_exist_hf==True or all_exist_dir==True:
                print("RESULT: It seems all models were found. Nothing will be downloaded!") 
        if all_exist_hf==False and all_exist_dir==False:
            retval_models_exist=False


    retval_final=retval_models_exist == True and retval_paths_exist ==True

    return retval_final
        

def lcx_checkmodels(dotenv_needed_models,dotenv_loaded_paths, dotenv_loaded_models, dotenv_loaded_models_values, in_files_to_check_in_paths=None  ):
    check_do_all_files_exist(dotenv_needed_models,dotenv_loaded_paths, dotenv_loaded_models, dotenv_loaded_models_values, in_files_to_check_in_paths=in_files_to_check_in_paths  )

    sys.exit()


### SYS REPORT START##################
import sys
import platform
import subprocess
import os
import shutil
import torch
import psutil
from datetime import datetime

def generate_troubleshooting_report(in_model_config_file=None):
    """Generate a comprehensive troubleshooting report for AI/LLM deployment issues."""
    
    # Create a divider for better readability
    divider = "=" * 80
    
    # Initialize report
    report = []
    report.append(f"{divider}")
    report.append(f"TROUBLESHOOTING REPORT - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    #report.append(f"{divider}\n")
    
    # 1. Hardware Information
    #report.append(f"{divider}")
    report.append("HARDWARE INFORMATION")
    #report.append(f"{divider}")
    
    # CPU Info
    report.append("\nCPU:")
    report.append(f"  Model: {platform.processor()}")
    try:
        cpu_freq = psutil.cpu_freq()
        report.append(f"  Max Frequency: {cpu_freq.max:.2f} MHz")
        report.append(f"  Cores: Physical: {psutil.cpu_count(logical=False)}, Logical: {psutil.cpu_count(logical=True)}")
    except Exception as e:
        report.append(f"  Could not get CPU frequency info: {str(e)}")
    
    # RAM Info
    ram = psutil.virtual_memory()
    report.append("\nRAM:")
    report.append(f"  Total: {ram.total / (1024**3):.2f} GB: free: {ram.available / (1024**3):.2f} used: {ram.used / (1024**3):.2f} GB")
     
    # GPU Info (try with nvidia-smi first, then fallback to torch if available)
    report.append("\nGPU:")
    try:
        nvidia_smi = shutil.which('nvidia-smi')
        if nvidia_smi:
            try:
                gpu_info = subprocess.check_output([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"], encoding='utf-8').strip()
                gpu_name, vram_total = gpu_info.split(',')
                report.append(f"  Model: {gpu_name.strip()}")
                report.append(f"  VRAM: {vram_total.strip()}")
                
                # Get current VRAM usage if possible
                try:
                    gpu_usage = subprocess.check_output([nvidia_smi, "--query-gpu=memory.used", "--format=csv,noheader"], encoding='utf-8').strip()
                    report.append(f"  VRAM Used: {gpu_usage.strip()}")
                except:
                    pass
            except Exception as e:
                report.append(f"  Could not query GPU info with nvidia-smi: {str(e)}")
    except:
        pass
    
    # If torch is available and has CUDA, get GPU info from torch
    try:
        if torch.cuda.is_available():
            report.append("\nGPU Info from PyTorch:")
            for i in range(torch.cuda.device_count()):
                report.append(f"  Device {i}: {torch.cuda.get_device_name(i)}, VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
    except:
        pass
    
    # Disk Space
    report.append("\nDISK:")
    try:
        disk = psutil.disk_usage('/')
        report.append(f"  Total: {disk.total / (1024**3):.2f} GB.  Free: {disk.free / (1024**3):.2f} GB, Used: {disk.used / (1024**3):.2f} GB")
    except Exception as e:
        report.append(f"  Could not get disk info: {str(e)}")
    
    # 2. Software Information
    report.append(f"\n{divider}")
    report.append("SOFTWARE INFORMATION")
    #report.append(f"{divider}")
    
    # OS Info
    report.append("\nOPERATING SYSTEM:")
    report.append(f"  System: {platform.system()}")
    report.append(f"  Release: {platform.release()}")
    report.append(f"  Version: {platform.version()}")
    report.append(f"  Machine: {platform.machine()}")
    
    # Python Info
    report.append("\nPYTHON:")
    report.append(f"  Version: {platform.python_version()}")
    report.append(f"  Implementation: {platform.python_implementation()}")
    report.append(f"  Executable: {sys.executable}")
    
    # Installed packages
    report.append("\nINSTALLED PACKAGES (pip freeze):")
    try:
        pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], encoding='utf-8')
        report.append(pip_freeze)
    except Exception as e:
        report.append(f"  Could not get pip freeze output: {str(e)}")
    
    # CUDA Info
    report.append("CUDA INFORMATION:")
    try:
        # Check nvcc version
        nvcc_path = shutil.which('nvcc')
        if nvcc_path:
            nvcc_version = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
            report.append(nvcc_version.strip())
#            report.append(nvcc_version.split('\n')[0])
        else:
            report.append("NVCC not found in PATH")
    except Exception as e:
        report.append(f"  Could not get NVCC version: {str(e)}")
    
    # PyTorch CUDA version if available
    try:
        if 'torch' in sys.modules:
            report.append("\nPYTORCH CUDA:")
            report.append(f"  PyTorch version: {torch.__version__}")
            report.append(f"  CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                report.append(f"  CUDA version: {torch.version.cuda}")
                report.append(f"  Current device: {torch.cuda.current_device()}")
                report.append(f"  Device name: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        report.append(f"  Could not get PyTorch CUDA info: {str(e)}")
    
    # 3. Model Configuration
    if in_model_config_file:
        report.append(f"\n{divider}")
        report.append("MODEL CONFIGURATION")
        #report.append(f"{divider}")
        
        try:
            # Read config file content
            with open(in_model_config_file, 'r') as f:
                config_content = f.read()
            report.append(f"Content of {in_model_config_file}:")
            report.append(config_content)
        except Exception as e:
            report.append(f"\nCould not read model config file {in_model_config_file}: {str(e)}")
    
    # 4. Environment Variables
    report.append(f"\n{divider}")
    report.append("RELEVANT ENVIRONMENT VARIABLES")
    #report.append(f"{divider}")
    
    relevant_env_vars = [
        'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
        'PYTHONPATH', 'CONDA_PREFIX', 'VIRTUAL_ENV'
    ]
    
    for var in relevant_env_vars:
        if var in os.environ:
            report.append(f"{var}: {os.environ[var]}")
    
    # 5. Additional System Info
    report.append(f"\n{divider}")
    report.append("ADDITIONAL SYSTEM INFORMATION")
    #report.append(f"{divider}")
    
    try:
        # Check if running in container
        report.append("\nContainer/Virtualization:")
        if os.path.exists('/.dockerenv'):
            report.append("  Running inside a Docker container")
        elif os.path.exists('/proc/1/cgroup'):
            with open('/proc/1/cgroup', 'r') as f:
                if 'docker' in f.read():
                    report.append("  Running inside a Docker container")
                elif 'kubepods' in f.read():
                    report.append("  Running inside a Kubernetes pod")
        
        # Check virtualization
        try:
            virt = subprocess.check_output(['systemd-detect-virt'], encoding='utf-8').strip()
            if virt != 'none':
                report.append(f"  Virtualization: {virt}")
        except:
            pass
    except Exception as e:
        report.append(f"  Could not check container/virtualization info: {str(e)}")
    
    # Final divider
    #report.append(f"\n{divider}")
    report.append("END OF REPORT")
    report.append(f"{divider}")
    
    # Join all report lines
    full_report = '\n'.join(report)
    
    # Print to terminal
    
    
    return full_report
#END SYSREPORT###################################################################




# Update the config file
update_model_paths_file(in_dotenv_needed_models, in_dotenv_needed_paths, in_dotenv_needed_params, in_model_config_file)

# Read back the values
out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params , out_dotenv_loaded_models_values= parse_model_paths_file(in_model_config_file, in_dotenv_needed_models,in_dotenv_needed_paths)



if debug_mode:
    print("Loaded models:", out_dotenv_loaded_models)
    print("Loaded models values:", out_dotenv_loaded_models_values)
    print("Loaded paths:", out_dotenv_loaded_paths)
    print("Loaded params:", out_dotenv_loaded_params)
    
if "HF_HOME" in in_dotenv_needed_paths:
    os.environ['HF_HOME'] = out_dotenv_loaded_paths["HF_HOME"]
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#os.environ["TOKENIZERS_PARALLELISM"] = "true"


#originalblock#################################
import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--output_dir", type=str, default='./outputs')
parser.add_argument("--checkmodels", action='store_true')
parser.add_argument("--integritycheck", action='store_true')
parser.add_argument("--sysreport", action='store_true')
args = parser.parse_args()
###################################
# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

#return out_dotenv_loaded_models, out_dotenv_loaded_paths, out_dotenv_loaded_params 

if args.checkmodels: 
    lcx_checkmodels(in_dotenv_needed_models,out_dotenv_loaded_paths, out_dotenv_loaded_models, out_dotenv_loaded_models_values, in_files_to_check_in_paths )

if args.sysreport: 
    full_report=generate_troubleshooting_report(in_model_config_file=in_model_config_file)
    print(full_report)
    sys.exit()

if debug_mode:
    print("---current model paths---------")
    for id in out_dotenv_loaded_models:
        print (f"{id}: {out_dotenv_loaded_models[id]}")

####################################################################################################################
####################################################################################################################
####################################################################################################################
#prefix end#########################################################################################################
#example_var=out_dotenv_loaded_params["DEBUG_MODE"]












import sys
ext_lib_path = os.path.abspath(f"{out_dotenv_loaded_paths['MODELS_HOME']}")
sys.path.append(ext_lib_path)

import media_rc

# -*- coding: utf-8 -*-
################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING
################################################################################
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDockWidget, QGraphicsView,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QListView, QListWidget, QListWidgetItem,
    QMainWindow, QMenu, QMenuBar, QProgressBar,
    QPushButton, QSizePolicy, QSlider, QSpacerItem,
    QTabWidget, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1376, 585)
        font = QFont()
        font.setPointSize(10)
        MainWindow.setFont(font)
        icon = QIcon()
        icon.addFile(u":/media/media/visomaster_small.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionLoad_Embeddings = QAction(MainWindow)
        self.actionLoad_Embeddings.setObjectName(u"actionLoad_Embeddings")
        self.actionSave_Embeddings = QAction(MainWindow)
        self.actionSave_Embeddings.setObjectName(u"actionSave_Embeddings")
        self.actionSave_Embeddings_As = QAction(MainWindow)
        self.actionSave_Embeddings_As.setObjectName(u"actionSave_Embeddings_As")
        self.actionOpen_Videos_Folder = QAction(MainWindow)
        self.actionOpen_Videos_Folder.setObjectName(u"actionOpen_Videos_Folder")
        self.actionOpen_Video_Files = QAction(MainWindow)
        self.actionOpen_Video_Files.setObjectName(u"actionOpen_Video_Files")
        self.actionLoad_Source_Images_Folder = QAction(MainWindow)
        self.actionLoad_Source_Images_Folder.setObjectName(u"actionLoad_Source_Images_Folder")
        self.actionLoad_Source_Image_Files = QAction(MainWindow)
        self.actionLoad_Source_Image_Files.setObjectName(u"actionLoad_Source_Image_Files")
        self.actionView_Fullscreen_F11 = QAction(MainWindow)
        self.actionView_Fullscreen_F11.setObjectName(u"actionView_Fullscreen_F11")
        self.actionTest = QAction(MainWindow)
        self.actionTest.setObjectName(u"actionTest")
        self.actionLoad_Saved_Workspace = QAction(MainWindow)
        self.actionLoad_Saved_Workspace.setObjectName(u"actionLoad_Saved_Workspace")
        self.actionSave_Current_Workspace = QAction(MainWindow)
        self.actionSave_Current_Workspace.setObjectName(u"actionSave_Current_Workspace")
        self.actionTest_2 = QAction(MainWindow)
        self.actionTest_2.setObjectName(u"actionTest_2")
        self.actionLoad_SavedWorkspace = QAction(MainWindow)
        self.actionLoad_SavedWorkspace.setObjectName(u"actionLoad_SavedWorkspace")
        self.actionSave_CurrentWorkspace = QAction(MainWindow)
        self.actionSave_CurrentWorkspace.setObjectName(u"actionSave_CurrentWorkspace")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout = QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.mediaLayout = QWidget(self.centralwidget)
        self.mediaLayout.setObjectName(u"mediaLayout")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(2)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mediaLayout.sizePolicy().hasHeightForWidth())
        self.mediaLayout.setSizePolicy(sizePolicy)
        self.verticalLayout = QVBoxLayout(self.mediaLayout)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.panelVisibilityCheckBoxLayout = QHBoxLayout()
        self.panelVisibilityCheckBoxLayout.setObjectName(u"panelVisibilityCheckBoxLayout")
        self.mediaPanelCheckBox = QCheckBox(self.mediaLayout)
        self.mediaPanelCheckBox.setObjectName(u"mediaPanelCheckBox")
        self.mediaPanelCheckBox.setChecked(True)
        self.panelVisibilityCheckBoxLayout.addWidget(self.mediaPanelCheckBox)
        self.facesPanelCheckBox = QCheckBox(self.mediaLayout)
        self.facesPanelCheckBox.setObjectName(u"facesPanelCheckBox")
        self.facesPanelCheckBox.setChecked(True)
        self.panelVisibilityCheckBoxLayout.addWidget(self.facesPanelCheckBox)
        self.parametersPanelCheckBox = QCheckBox(self.mediaLayout)
        self.parametersPanelCheckBox.setObjectName(u"parametersPanelCheckBox")
        self.parametersPanelCheckBox.setChecked(True)
        self.panelVisibilityCheckBoxLayout.addWidget(self.parametersPanelCheckBox)
        self.horizontalSpacer_8 = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.panelVisibilityCheckBoxLayout.addItem(self.horizontalSpacer_8)
        self.faceCompareCheckBox = QCheckBox(self.mediaLayout)
        self.faceCompareCheckBox.setObjectName(u"faceCompareCheckBox")
        self.panelVisibilityCheckBoxLayout.addWidget(self.faceCompareCheckBox)
        self.faceMaskCheckBox = QCheckBox(self.mediaLayout)
        self.faceMaskCheckBox.setObjectName(u"faceMaskCheckBox")
        self.panelVisibilityCheckBoxLayout.addWidget(self.faceMaskCheckBox)
        self.verticalLayout.addLayout(self.panelVisibilityCheckBoxLayout)
        self.graphicsViewFrame = QGraphicsView(self.mediaLayout)
        self.graphicsViewFrame.setObjectName(u"graphicsViewFrame")
        self.verticalLayout.addWidget(self.graphicsViewFrame)
        self.verticalLayoutMediaControls = QVBoxLayout()
        self.verticalLayoutMediaControls.setObjectName(u"verticalLayoutMediaControls")
        self.horizontalLayoutMediaSlider = QHBoxLayout()
        self.horizontalLayoutMediaSlider.setObjectName(u"horizontalLayoutMediaSlider")
        self.videoSeekSlider = QSlider(self.mediaLayout)
        self.videoSeekSlider.setObjectName(u"videoSeekSlider")
        self.videoSeekSlider.setOrientation(Qt.Orientation.Horizontal)
        self.horizontalLayoutMediaSlider.addWidget(self.videoSeekSlider)
        self.videoSeekLineEdit = QLineEdit(self.mediaLayout)
        self.videoSeekLineEdit.setObjectName(u"videoSeekLineEdit")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.videoSeekLineEdit.sizePolicy().hasHeightForWidth())
        self.videoSeekLineEdit.setSizePolicy(sizePolicy1)
        self.videoSeekLineEdit.setMaximumSize(QSize(70, 16777215))
        self.videoSeekLineEdit.setClearButtonEnabled(False)
        self.horizontalLayoutMediaSlider.addWidget(self.videoSeekLineEdit)
        self.verticalLayoutMediaControls.addLayout(self.horizontalLayoutMediaSlider)
        self.horizontalLayoutMediaButtons = QHBoxLayout()
        self.horizontalLayoutMediaButtons.setObjectName(u"horizontalLayoutMediaButtons")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer)
        self.frameRewindButton = QPushButton(self.mediaLayout)
        self.frameRewindButton.setObjectName(u"frameRewindButton")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.frameRewindButton.sizePolicy().hasHeightForWidth())
        self.frameRewindButton.setSizePolicy(sizePolicy2)
        self.frameRewindButton.setMaximumSize(QSize(100, 16777215))
        icon1 = QIcon()
        icon1.addFile(u":/media/media/previous_marker_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.frameRewindButton.setIcon(icon1)
        self.frameRewindButton.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.frameRewindButton)
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_3)
        self.buttonMediaRecord = QPushButton(self.mediaLayout)
        self.buttonMediaRecord.setObjectName(u"buttonMediaRecord")
        icon2 = QIcon()
        icon2.addFile(u":/media/media/rec_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.buttonMediaRecord.setIcon(icon2)
        self.buttonMediaRecord.setCheckable(True)
        self.buttonMediaRecord.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.buttonMediaRecord)
        self.horizontalSpacer_6 = QSpacerItem(30, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_6)
        self.buttonMediaPlay = QPushButton(self.mediaLayout)
        self.buttonMediaPlay.setObjectName(u"buttonMediaPlay")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Minimum)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.buttonMediaPlay.sizePolicy().hasHeightForWidth())
        self.buttonMediaPlay.setSizePolicy(sizePolicy3)
        self.buttonMediaPlay.setMaximumSize(QSize(100, 16777215))
        icon3 = QIcon()
        icon3.addFile(u":/media/media/play_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.buttonMediaPlay.setIcon(icon3)
        self.buttonMediaPlay.setCheckable(True)
        self.buttonMediaPlay.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.buttonMediaPlay)
        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_4)
        self.frameAdvanceButton = QPushButton(self.mediaLayout)
        self.frameAdvanceButton.setObjectName(u"frameAdvanceButton")
        sizePolicy2.setHeightForWidth(self.frameAdvanceButton.sizePolicy().hasHeightForWidth())
        self.frameAdvanceButton.setSizePolicy(sizePolicy2)
        self.frameAdvanceButton.setMaximumSize(QSize(100, 16777215))
        icon4 = QIcon()
        icon4.addFile(u":/media/media/next_marker_off.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.frameAdvanceButton.setIcon(icon4)
        self.frameAdvanceButton.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.frameAdvanceButton)
        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_2)
        self.addMarkerButton = QPushButton(self.mediaLayout)
        self.addMarkerButton.setObjectName(u"addMarkerButton")
        icon5 = QIcon()
        icon5.addFile(u":/media/media/add_marker_hover.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.addMarkerButton.setIcon(icon5)
        self.addMarkerButton.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.addMarkerButton)
        self.removeMarkerButton = QPushButton(self.mediaLayout)
        self.removeMarkerButton.setObjectName(u"removeMarkerButton")
        icon6 = QIcon()
        icon6.addFile(u":/media/media/remove_marker_hover.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.removeMarkerButton.setIcon(icon6)
        self.removeMarkerButton.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.removeMarkerButton)
        self.previousMarkerButton = QPushButton(self.mediaLayout)
        self.previousMarkerButton.setObjectName(u"previousMarkerButton")
        icon7 = QIcon()
        icon7.addFile(u":/media/media/previous_marker_hover.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.previousMarkerButton.setIcon(icon7)
        self.previousMarkerButton.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.previousMarkerButton)
        self.nextMarkerButton = QPushButton(self.mediaLayout)
        self.nextMarkerButton.setObjectName(u"nextMarkerButton")
        icon8 = QIcon()
        icon8.addFile(u":/media/media/next_marker_hover.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.nextMarkerButton.setIcon(icon8)
        self.nextMarkerButton.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.nextMarkerButton)
        self.viewFullScreenButton = QPushButton(self.mediaLayout)
        self.viewFullScreenButton.setObjectName(u"viewFullScreenButton")
        icon9 = QIcon()
        icon9.addFile(u":/media/media/fullscreen.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.viewFullScreenButton.setIcon(icon9)
        self.viewFullScreenButton.setFlat(True)
        self.horizontalLayoutMediaButtons.addWidget(self.viewFullScreenButton)
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.horizontalLayoutMediaButtons.addItem(self.horizontalSpacer_5)
        self.verticalLayoutMediaControls.addLayout(self.horizontalLayoutMediaButtons)
        self.verticalLayout.addLayout(self.verticalLayoutMediaControls)
        self.verticalSpacer = QSpacerItem(20, 5, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self.verticalLayout.addItem(self.verticalSpacer)
        self.facesPanelGroupBox = QGroupBox(self.mediaLayout)
        self.facesPanelGroupBox.setObjectName(u"facesPanelGroupBox")
        sizePolicy1.setHeightForWidth(self.facesPanelGroupBox.sizePolicy().hasHeightForWidth())
        self.facesPanelGroupBox.setSizePolicy(sizePolicy1)
        self.facesPanelGroupBox.setMinimumSize(QSize(0, 180))
        self.facesPanelGroupBox.setMaximumSize(QSize(16777215, 225))
        self.facesPanelGroupBox.setAutoFillBackground(False)
        self.facesPanelGroupBox.setFlat(True)
        self.facesPanelGroupBox.setCheckable(False)
        self.gridLayout_2 = QGridLayout(self.facesPanelGroupBox)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.facesButtonsWidget = QWidget(self.facesPanelGroupBox)
        self.facesButtonsWidget.setObjectName(u"facesButtonsWidget")
        sizePolicy4 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.facesButtonsWidget.sizePolicy().hasHeightForWidth())
        self.facesButtonsWidget.setSizePolicy(sizePolicy4)
        self.verticalLayout_8 = QVBoxLayout(self.facesButtonsWidget)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalWidget = QWidget(self.facesButtonsWidget)
        self.verticalWidget.setObjectName(u"verticalWidget")
        self.controlButtonsLayout = QVBoxLayout(self.verticalWidget)
        self.controlButtonsLayout.setObjectName(u"controlButtonsLayout")
        self.findTargetFacesButton = QPushButton(self.verticalWidget)
        self.findTargetFacesButton.setObjectName(u"findTargetFacesButton")
        self.findTargetFacesButton.setMinimumSize(QSize(100, 0))
        self.findTargetFacesButton.setCheckable(False)
        self.findTargetFacesButton.setFlat(True)
        self.controlButtonsLayout.addWidget(self.findTargetFacesButton)
        self.clearTargetFacesButton = QPushButton(self.verticalWidget)
        self.clearTargetFacesButton.setObjectName(u"clearTargetFacesButton")
        self.clearTargetFacesButton.setCheckable(False)
        self.clearTargetFacesButton.setFlat(True)
        self.controlButtonsLayout.addWidget(self.clearTargetFacesButton)
        self.swapfacesButton = QPushButton(self.verticalWidget)
        self.swapfacesButton.setObjectName(u"swapfacesButton")
        self.swapfacesButton.setCheckable(True)
        self.swapfacesButton.setFlat(True)
        self.controlButtonsLayout.addWidget(self.swapfacesButton)
        self.editFacesButton = QPushButton(self.verticalWidget)
        self.editFacesButton.setObjectName(u"editFacesButton")
        self.editFacesButton.setCheckable(True)
        self.editFacesButton.setFlat(True)
        self.controlButtonsLayout.addWidget(self.editFacesButton)
        self.verticalLayout_8.addWidget(self.verticalWidget)
        self.gridLayout_2.addWidget(self.facesButtonsWidget, 1, 0, 1, 1)
        self.inputEmbeddingsList = QListWidget(self.facesPanelGroupBox)
        self.inputEmbeddingsList.setObjectName(u"inputEmbeddingsList")
        sizePolicy5 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sizePolicy5.setHorizontalStretch(4)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.inputEmbeddingsList.sizePolicy().hasHeightForWidth())
        self.inputEmbeddingsList.setSizePolicy(sizePolicy5)
        self.inputEmbeddingsList.setMinimumSize(QSize(320, 120))
        self.inputEmbeddingsList.setMaximumSize(QSize(16777215, 120))
        self.inputEmbeddingsList.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.inputEmbeddingsList.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.inputEmbeddingsList.setUniformItemSizes(True)
        self.inputEmbeddingsList.setViewMode(QListView.IconMode)
        self.inputEmbeddingsList.setMovement(QListView.Static)
        self.inputEmbeddingsList.setWrapping(True)
        self.inputEmbeddingsList.setLayoutMode(QListView.Batched)
        self.inputEmbeddingsList.setSpacing(4)
        self.inputEmbeddingsList.setAutoScroll(False)
        self.gridLayout_2.addWidget(self.inputEmbeddingsList, 1, 2, 1, 1)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.saveImageButton = QPushButton(self.facesPanelGroupBox)
        self.saveImageButton.setObjectName(u"saveImageButton")
        self.saveImageButton.setFlat(True)
        self.horizontalLayout_4.addWidget(self.saveImageButton)
        self.gridLayout_2.addLayout(self.horizontalLayout_4, 0, 0, 1, 1)
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.inputEmbeddingsSearchBox = QLineEdit(self.facesPanelGroupBox)
        self.inputEmbeddingsSearchBox.setObjectName(u"inputEmbeddingsSearchBox")
        self.horizontalLayout_3.addWidget(self.inputEmbeddingsSearchBox)
        self.openEmbeddingButton = QPushButton(self.facesPanelGroupBox)
        self.openEmbeddingButton.setObjectName(u"openEmbeddingButton")
        icon10 = QIcon()
        icon10.addFile(u":/media/media/open_file.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.openEmbeddingButton.setIcon(icon10)
        self.openEmbeddingButton.setFlat(True)
        self.horizontalLayout_3.addWidget(self.openEmbeddingButton)
        self.saveEmbeddingButton = QPushButton(self.facesPanelGroupBox)
        self.saveEmbeddingButton.setObjectName(u"saveEmbeddingButton")
        icon11 = QIcon()
        icon11.addFile(u":/media/media/save_file.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.saveEmbeddingButton.setIcon(icon11)
        self.saveEmbeddingButton.setFlat(True)
        self.horizontalLayout_3.addWidget(self.saveEmbeddingButton)
        self.saveEmbeddingAsButton = QPushButton(self.facesPanelGroupBox)
        self.saveEmbeddingAsButton.setObjectName(u"saveEmbeddingAsButton")
        icon12 = QIcon()
        icon12.addFile(u":/media/media/save_file_as.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.saveEmbeddingAsButton.setIcon(icon12)
        self.saveEmbeddingAsButton.setFlat(True)
        self.horizontalLayout_3.addWidget(self.saveEmbeddingAsButton)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 0, 2, 1, 1)
        self.targetFacesList = QListWidget(self.facesPanelGroupBox)
        self.targetFacesList.setObjectName(u"targetFacesList")
        self.targetFacesList.setAutoFillBackground(True)
        self.targetFacesList.setAutoScroll(False)
        self.gridLayout_2.addWidget(self.targetFacesList, 1, 1, 1, 1)
        self.verticalLayout.addWidget(self.facesPanelGroupBox)
        self.horizontalLayout.addWidget(self.mediaLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.input_Target_DockWidget = QDockWidget(MainWindow)
        self.input_Target_DockWidget.setObjectName(u"input_Target_DockWidget")
        sizePolicy6 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy6.setHorizontalStretch(4)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.input_Target_DockWidget.sizePolicy().hasHeightForWidth())
        self.input_Target_DockWidget.setSizePolicy(sizePolicy6)
        self.input_Target_DockWidget.setMinimumSize(QSize(340, 369))
        self.input_Target_DockWidget.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable|QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.gridLayout_4 = QGridLayout(self.dockWidgetContents)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.vboxLayout = QVBoxLayout()
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.groupBox_TargetVideos_Select = QGroupBox(self.dockWidgetContents)
        self.groupBox_TargetVideos_Select.setObjectName(u"groupBox_TargetVideos_Select")
        self.gridLayout_3 = QGridLayout(self.groupBox_TargetVideos_Select)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.labelTargetVideosPath = QLabel(self.groupBox_TargetVideos_Select)
        self.labelTargetVideosPath.setObjectName(u"labelTargetVideosPath")
        self.labelTargetVideosPath.setWordWrap(False)
        self.horizontalLayout_7.addWidget(self.labelTargetVideosPath)
        self.buttonTargetVideosPath = QPushButton(self.groupBox_TargetVideos_Select)
        self.buttonTargetVideosPath.setObjectName(u"buttonTargetVideosPath")
        sizePolicy7 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.buttonTargetVideosPath.sizePolicy().hasHeightForWidth())
        self.buttonTargetVideosPath.setSizePolicy(sizePolicy7)
        self.buttonTargetVideosPath.setIcon(icon10)
        self.buttonTargetVideosPath.setIconSize(QSize(18, 18))
        self.buttonTargetVideosPath.setFlat(True)
        self.horizontalLayout_7.addWidget(self.buttonTargetVideosPath)
        self.gridLayout_3.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        self.vboxLayout.addWidget(self.groupBox_TargetVideos_Select)
        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.targetVideosSearchBox = QLineEdit(self.dockWidgetContents)
        self.targetVideosSearchBox.setObjectName(u"targetVideosSearchBox")
        self.horizontalLayout_9.addWidget(self.targetVideosSearchBox)
        self.filterImagesCheckBox = QCheckBox(self.dockWidgetContents)
        self.filterImagesCheckBox.setObjectName(u"filterImagesCheckBox")
        icon13 = QIcon()
        icon13.addFile(u":/media/media/image.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.filterImagesCheckBox.setIcon(icon13)
        self.filterImagesCheckBox.setChecked(True)
        self.horizontalLayout_9.addWidget(self.filterImagesCheckBox)
        self.filterVideosCheckBox = QCheckBox(self.dockWidgetContents)
        self.filterVideosCheckBox.setObjectName(u"filterVideosCheckBox")
        icon14 = QIcon()
        icon14.addFile(u":/media/media/video.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.filterVideosCheckBox.setIcon(icon14)
        self.filterVideosCheckBox.setChecked(True)
        self.horizontalLayout_9.addWidget(self.filterVideosCheckBox)
        self.filterWebcamsCheckBox = QCheckBox(self.dockWidgetContents)
        self.filterWebcamsCheckBox.setObjectName(u"filterWebcamsCheckBox")
        icon15 = QIcon()
        icon15.addFile(u":/media/media/webcam.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        self.filterWebcamsCheckBox.setIcon(icon15)
        self.filterWebcamsCheckBox.setChecked(False)
        self.horizontalLayout_9.addWidget(self.filterWebcamsCheckBox)
        self.vboxLayout.addLayout(self.horizontalLayout_9)
        self.targetVideosList = QListWidget(self.dockWidgetContents)
        self.targetVideosList.setObjectName(u"targetVideosList")
        self.targetVideosList.setAcceptDrops(True)
        self.targetVideosList.setAutoScroll(False)
        self.vboxLayout.addWidget(self.targetVideosList)
        self.groupBox_InputFaces_Select = QGroupBox(self.dockWidgetContents)
        self.groupBox_InputFaces_Select.setObjectName(u"groupBox_InputFaces_Select")
        self.gridLayout = QGridLayout(self.groupBox_InputFaces_Select)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.labelInputFacesPath = QLabel(self.groupBox_InputFaces_Select)
        self.labelInputFacesPath.setObjectName(u"labelInputFacesPath")
        self.horizontalLayout_8.addWidget(self.labelInputFacesPath)
        self.buttonInputFacesPath = QPushButton(self.groupBox_InputFaces_Select)
        self.buttonInputFacesPath.setObjectName(u"buttonInputFacesPath")
        sizePolicy7.setHeightForWidth(self.buttonInputFacesPath.sizePolicy().hasHeightForWidth())
        self.buttonInputFacesPath.setSizePolicy(sizePolicy7)
        self.buttonInputFacesPath.setIcon(icon10)
        self.buttonInputFacesPath.setIconSize(QSize(18, 18))
        self.buttonInputFacesPath.setFlat(True)
        self.horizontalLayout_8.addWidget(self.buttonInputFacesPath)
        self.gridLayout.addLayout(self.horizontalLayout_8, 0, 0, 1, 1)
        self.vboxLayout.addWidget(self.groupBox_InputFaces_Select)
        self.inputFacesSearchBox = QLineEdit(self.dockWidgetContents)
        self.inputFacesSearchBox.setObjectName(u"inputFacesSearchBox")
        self.vboxLayout.addWidget(self.inputFacesSearchBox)
        self.inputFacesList = QListWidget(self.dockWidgetContents)
        self.inputFacesList.setObjectName(u"inputFacesList")
        self.inputFacesList.setAcceptDrops(True)
        self.inputFacesList.setAutoScroll(False)
        self.vboxLayout.addWidget(self.inputFacesList)
        self.gridLayout_4.addLayout(self.vboxLayout, 0, 0, 1, 1)
        self.input_Target_DockWidget.setWidget(self.dockWidgetContents)
        MainWindow.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.input_Target_DockWidget)
        self.controlOptionsDockWidget = QDockWidget(MainWindow)
        self.controlOptionsDockWidget.setObjectName(u"controlOptionsDockWidget")
        self.controlOptionsDockWidget.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable|QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.dockWidgetContents_2 = QWidget()
        self.dockWidgetContents_2.setObjectName(u"dockWidgetContents_2")
        self.gridLayout_5 = QGridLayout(self.dockWidgetContents_2)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.tabWidget = QTabWidget(self.dockWidgetContents_2)
        self.tabWidget.setObjectName(u"tabWidget")
        sizePolicy8 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy8.setHorizontalStretch(1)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy8)
        font1 = QFont()
        font1.setFamilies([u"Segoe UI Semibold"])
        font1.setPointSize(10)
        font1.setBold(True)
        self.tabWidget.setFont(font1)
        self.tabWidget.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.tabWidget.setTabPosition(QTabWidget.TabPosition.North)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(True)
        self.tabWidget.setTabBarAutoHide(False)
        self.face_swap_tab = QWidget()
        self.face_swap_tab.setObjectName(u"face_swap_tab")
        self.verticalLayout_4 = QVBoxLayout(self.face_swap_tab)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.swapWidgetsLayout = QVBoxLayout()
        self.swapWidgetsLayout.setObjectName(u"swapWidgetsLayout")
        self.verticalLayout_4.addLayout(self.swapWidgetsLayout)
        self.tabWidget.addTab(self.face_swap_tab, "")
        self.face_editor_tab = QWidget()
        self.face_editor_tab.setObjectName(u"face_editor_tab")
        self.verticalLayout_3 = QVBoxLayout(self.face_editor_tab)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.faceEditorWidgetsLayout = QVBoxLayout()
        self.faceEditorWidgetsLayout.setObjectName(u"faceEditorWidgetsLayout")
        self.verticalLayout_3.addLayout(self.faceEditorWidgetsLayout)
        self.tabWidget.addTab(self.face_editor_tab, "")
        self.common_tab = QWidget()
        self.common_tab.setObjectName(u"common_tab")
        self.commonWidgetsLayout_1 = QVBoxLayout(self.common_tab)
        self.commonWidgetsLayout_1.setObjectName(u"commonWidgetsLayout_1")
        self.commonWidgetsLayout = QVBoxLayout()
        self.commonWidgetsLayout.setObjectName(u"commonWidgetsLayout")
        self.commonWidgetsLayout_1.addLayout(self.commonWidgetsLayout)
        self.tabWidget.addTab(self.common_tab, "")
        self.settings_tab = QWidget()
        self.settings_tab.setObjectName(u"settings_tab")
        self.verticalLayout_2 = QVBoxLayout(self.settings_tab)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label = QLabel(self.settings_tab)
        self.label.setObjectName(u"label")
        self.verticalLayout_2.addWidget(self.label)
        self.settingsWidgetsLayout = QVBoxLayout()
        self.settingsWidgetsLayout.setObjectName(u"settingsWidgetsLayout")
        self.outputFolderSelectionLayout = QHBoxLayout()
        self.outputFolderSelectionLayout.setObjectName(u"outputFolderSelectionLayout")
        self.outputFolderLineEdit = QLineEdit(self.settings_tab)
        self.outputFolderLineEdit.setObjectName(u"outputFolderLineEdit")
        self.outputFolderLineEdit.setReadOnly(True)
        self.outputFolderSelectionLayout.addWidget(self.outputFolderLineEdit)
        self.outputFolderButton = QPushButton(self.settings_tab)
        self.outputFolderButton.setObjectName(u"outputFolderButton")
        self.outputFolderButton.setFlat(False)
        self.outputFolderSelectionLayout.addWidget(self.outputFolderButton)
        self.settingsWidgetsLayout.addLayout(self.outputFolderSelectionLayout)
        self.verticalLayout_2.addLayout(self.settingsWidgetsLayout)
        self.tabWidget.addTab(self.settings_tab, "")
        self.gridLayout_5.addWidget(self.tabWidget, 1, 0, 1, 1)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.vramProgressBar = QProgressBar(self.dockWidgetContents_2)
        self.vramProgressBar.setObjectName(u"vramProgressBar")
        self.vramProgressBar.setValue(24)
        self.horizontalLayout_2.addWidget(self.vramProgressBar)
        self.clearMemoryButton = QPushButton(self.dockWidgetContents_2)
        self.clearMemoryButton.setObjectName(u"clearMemoryButton")
        self.clearMemoryButton.setFlat(True)
        self.horizontalLayout_2.addWidget(self.clearMemoryButton)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.controlOptionsDockWidget.setWidget(self.dockWidgetContents_2)
        MainWindow.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.controlOptionsDockWidget)
        self.topMenuBar = QMenuBar(MainWindow)
        self.topMenuBar.setObjectName(u"topMenuBar")
        self.topMenuBar.setGeometry(QRect(0, 0, 1376, 33))
        self.menuFile = QMenu(self.topMenuBar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuEdit = QMenu(self.topMenuBar)
        self.menuEdit.setObjectName(u"menuEdit")
        self.menuView = QMenu(self.topMenuBar)
        self.menuView.setObjectName(u"menuView")
        MainWindow.setMenuBar(self.topMenuBar)
        self.topMenuBar.addAction(self.menuFile.menuAction())
        self.topMenuBar.addAction(self.menuEdit.menuAction())
        self.topMenuBar.addAction(self.menuView.menuAction())
        self.menuFile.addAction(self.actionLoad_SavedWorkspace)
        self.menuFile.addAction(self.actionSave_CurrentWorkspace)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Videos_Folder)
        self.menuFile.addAction(self.actionOpen_Video_Files)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Source_Images_Folder)
        self.menuFile.addAction(self.actionLoad_Source_Image_Files)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionLoad_Embeddings)
        self.menuFile.addAction(self.actionSave_Embeddings)
        self.menuFile.addAction(self.actionSave_Embeddings_As)
        self.menuEdit.addAction(self.actionTest_2)
        self.menuView.addAction(self.actionView_Fullscreen_F11)
        self.retranslateUi(MainWindow)
        self.editFacesButton.setDefault(False)
        self.tabWidget.setCurrentIndex(0)
        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi
    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"VisoMaster v0.1.6", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionLoad_Embeddings.setText(QCoreApplication.translate("MainWindow", u"Load Embeddings", None))
        self.actionSave_Embeddings.setText(QCoreApplication.translate("MainWindow", u"Save Embeddings", None))
        self.actionSave_Embeddings_As.setText(QCoreApplication.translate("MainWindow", u"Save Embeddings As", None))
        self.actionOpen_Videos_Folder.setText(QCoreApplication.translate("MainWindow", u"Load Target Images/Videos Folder", None))
        self.actionOpen_Video_Files.setText(QCoreApplication.translate("MainWindow", u"Load Target Image/Video Files", None))
        self.actionLoad_Source_Images_Folder.setText(QCoreApplication.translate("MainWindow", u"Load Source Images Folder", None))
        self.actionLoad_Source_Image_Files.setText(QCoreApplication.translate("MainWindow", u"Load Source Image Files", None))
        self.actionView_Fullscreen_F11.setText(QCoreApplication.translate("MainWindow", u"View Fullscreen (F11)", None))
        self.actionTest.setText(QCoreApplication.translate("MainWindow", u"Test", None))
        self.actionLoad_Saved_Workspace.setText(QCoreApplication.translate("MainWindow", u"Load Saved Workspace", None))
        self.actionSave_Current_Workspace.setText(QCoreApplication.translate("MainWindow", u"Save Current Workspace", None))
        self.actionTest_2.setText(QCoreApplication.translate("MainWindow", u"Test", None))
        self.actionLoad_SavedWorkspace.setText(QCoreApplication.translate("MainWindow", u"Load Saved Workspace", None))
        self.actionSave_CurrentWorkspace.setText(QCoreApplication.translate("MainWindow", u"Save Current Workspace", None))
        self.mediaPanelCheckBox.setText(QCoreApplication.translate("MainWindow", u"Media Panel", None))
        self.facesPanelCheckBox.setText(QCoreApplication.translate("MainWindow", u"Faces Panel", None))
        self.parametersPanelCheckBox.setText(QCoreApplication.translate("MainWindow", u"Parameters Panel", None))
        self.faceCompareCheckBox.setText(QCoreApplication.translate("MainWindow", u"VIew Face Compare", None))
        self.faceMaskCheckBox.setText(QCoreApplication.translate("MainWindow", u"View Face Mask", None))
#if QT_CONFIG(tooltip)
        self.videoSeekLineEdit.setToolTip(QCoreApplication.translate("MainWindow", u"Frame Number", None))
#endif // QT_CONFIG(tooltip)
        self.frameRewindButton.setText("")
        self.buttonMediaRecord.setText("")
        self.buttonMediaPlay.setText("")
        self.frameAdvanceButton.setText("")
#if QT_CONFIG(tooltip)
        self.addMarkerButton.setToolTip(QCoreApplication.translate("MainWindow", u"Add Marker", None))
#endif // QT_CONFIG(tooltip)
        self.addMarkerButton.setText("")
#if QT_CONFIG(tooltip)
        self.removeMarkerButton.setToolTip(QCoreApplication.translate("MainWindow", u"Remove Marker", None))
#endif // QT_CONFIG(tooltip)
        self.removeMarkerButton.setText("")
#if QT_CONFIG(tooltip)
        self.previousMarkerButton.setToolTip(QCoreApplication.translate("MainWindow", u"Move to Previous Marker", None))
#endif // QT_CONFIG(tooltip)
        self.previousMarkerButton.setText("")
#if QT_CONFIG(tooltip)
        self.nextMarkerButton.setToolTip(QCoreApplication.translate("MainWindow", u"Move to Next Marker", None))
#endif // QT_CONFIG(tooltip)
        self.nextMarkerButton.setText("")
#if QT_CONFIG(tooltip)
        self.viewFullScreenButton.setToolTip(QCoreApplication.translate("MainWindow", u"View Fullscreen (F11)", None))
#endif // QT_CONFIG(tooltip)
        self.viewFullScreenButton.setText("")
        self.findTargetFacesButton.setText(QCoreApplication.translate("MainWindow", u"Find Faces", None))
        self.clearTargetFacesButton.setText(QCoreApplication.translate("MainWindow", u"Clear Faces", None))
        self.swapfacesButton.setText(QCoreApplication.translate("MainWindow", u"Swap Faces", None))
        self.editFacesButton.setText(QCoreApplication.translate("MainWindow", u"Edit Faces", None))
#if QT_CONFIG(tooltip)
        self.inputEmbeddingsList.setToolTip(QCoreApplication.translate("MainWindow", u"Save Embedding", None))
#endif // QT_CONFIG(tooltip)
        self.saveImageButton.setText(QCoreApplication.translate("MainWindow", u"Save Image", None))
        self.inputEmbeddingsSearchBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search Embeddings", None))
#if QT_CONFIG(tooltip)
        self.openEmbeddingButton.setToolTip(QCoreApplication.translate("MainWindow", u"Open Embedding File", None))
#endif // QT_CONFIG(tooltip)
        self.openEmbeddingButton.setText("")
#if QT_CONFIG(tooltip)
        self.saveEmbeddingButton.setToolTip(QCoreApplication.translate("MainWindow", u"Save Embedding", None))
#endif // QT_CONFIG(tooltip)
        self.saveEmbeddingButton.setText("")
#if QT_CONFIG(tooltip)
        self.saveEmbeddingAsButton.setToolTip(QCoreApplication.translate("MainWindow", u"Save Embedding As", None))
#endif // QT_CONFIG(tooltip)
        self.saveEmbeddingAsButton.setText("")
        self.input_Target_DockWidget.setWindowTitle(QCoreApplication.translate("MainWindow", u"Target Videos and Input Faces", None))
        self.groupBox_TargetVideos_Select.setTitle(QCoreApplication.translate("MainWindow", u"Target Videos/Images", None))
        self.labelTargetVideosPath.setText(QCoreApplication.translate("MainWindow", u"Select Videos/Images Path", None))
#if QT_CONFIG(tooltip)
        self.buttonTargetVideosPath.setToolTip(QCoreApplication.translate("MainWindow", u"Choose Target Media Folder", None))
#endif // QT_CONFIG(tooltip)
        self.buttonTargetVideosPath.setText("")
        self.targetVideosSearchBox.setText("")
        self.targetVideosSearchBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search Videos/Images", None))
#if QT_CONFIG(tooltip)
        self.filterImagesCheckBox.setToolTip(QCoreApplication.translate("MainWindow", u"Include Images", None))
#endif // QT_CONFIG(tooltip)
        self.filterImagesCheckBox.setText("")
#if QT_CONFIG(tooltip)
        self.filterVideosCheckBox.setToolTip(QCoreApplication.translate("MainWindow", u"Include Videos", None))
#endif // QT_CONFIG(tooltip)
        self.filterVideosCheckBox.setText("")
#if QT_CONFIG(tooltip)
        self.filterWebcamsCheckBox.setToolTip(QCoreApplication.translate("MainWindow", u"Include Webcams", None))
#endif // QT_CONFIG(tooltip)
        self.filterWebcamsCheckBox.setText("")
        self.groupBox_InputFaces_Select.setTitle(QCoreApplication.translate("MainWindow", u"Input Faces", None))
        self.labelInputFacesPath.setText(QCoreApplication.translate("MainWindow", u"Select Face Images Path", None))
#if QT_CONFIG(tooltip)
        self.buttonInputFacesPath.setToolTip(QCoreApplication.translate("MainWindow", u"Choose Input Faces Folder", None))
#endif // QT_CONFIG(tooltip)
        self.buttonInputFacesPath.setText("")
        self.inputFacesSearchBox.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Search Faces", None))
        self.controlOptionsDockWidget.setWindowTitle(QCoreApplication.translate("MainWindow", u"Control Options", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.face_swap_tab), QCoreApplication.translate("MainWindow", u"Face Swap", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.face_editor_tab), QCoreApplication.translate("MainWindow", u"Face Editor", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.common_tab), QCoreApplication.translate("MainWindow", u"Common", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Output Directory", None))
        self.outputFolderButton.setText(QCoreApplication.translate("MainWindow", u"Browse Folder", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.settings_tab), QCoreApplication.translate("MainWindow", u"Settings", None))
        self.clearMemoryButton.setText(QCoreApplication.translate("MainWindow", u"Clear VRAM", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("MainWindow", u"Edit", None))
        self.menuView.setTitle(QCoreApplication.translate("MainWindow", u"View", None))
    # retranslateUi
