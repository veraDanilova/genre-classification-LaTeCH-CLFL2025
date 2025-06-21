import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import *



def load_CORE(main_folder):

    base_url = 'https://raw.githubusercontent.com/TurkuNLP/'

    corpora = [('FreCORE', 'Multilingual-register-corpora/main/data/FreCORE/'),
    ('SweCORE', 'Multilingual-register-corpora/main/data/SweCORE/'),
    ('FinCORE', 'FinCORE/master/data/'),
    ('CORE', 'CORE-corpus/master/')]
    files = ['train','dev','test']

    for corpus in corpora:
        
        folder = os.path.join(main_folder, corpus[0])
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Directory {folder} created successfully!")
        else:
            print(f"Directory {folder} already exists!")

        for file in files:
            
            if corpus[0] == "CORE":

                extension = ".tsv.gz"          
                url = os.path.join(base_url, corpus[1], file+extension)

                download_file(url, os.path.join(folder, file+extension))
                unzip_file(os.path.join(folder, file+extension), os.path.join(folder, file+'.tsv'))

            else:

                extension = ".tsv"
                url = os.path.join(base_url, corpus[1], file+extension)
                download_file(url, os.path.join(folder, file+extension))


def load_FTD(main_folder):

    base_url = 'https://raw.githubusercontent.com/ssharoff/genre-keras/master/'
    
    files = ['en.ol.xz','en.gold.dat','ru.ol.xz', 'ru.csv']

    folder = os.path.join(main_folder, "FTD")
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory {folder} ..")
    else:
        print(f"Directory {folder} already exists ..")

    for file in files:

        url = os.path.join(base_url, file)
        
        download_file(url, os.path.join(folder, file))

        if '.dat' in file:
            unzip_file(os.path.join(folder, file), os.path.join(folder, 'unzipped_'+file))

        if file.endswith('.xz'):
            unxz_file(os.path.join(folder, file), os.path.join(folder, file).split('.xz')[0])