from app.helpers.downloader import download_file
from app.processors.models_data import models_list, extra_lib_list

"""
for model_data in models_list:
    download_file(model_data['model_name'], model_data['local_path'], model_data['hash'], model_data['url'])
"""




for lib_data in extra_lib_list:
    download_file(lib_data['model_name'], lib_data['local_path'], lib_data['hash'], lib_data['url'])
