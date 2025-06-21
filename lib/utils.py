import requests
import gzip
import shutil
import os
import lzma

def unxz_file(xz_path, out_path):

    print(f"Unpacking {xz_path} ...")
    with lzma.open(xz_path, 'rb') as f_in:
        with open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Unpacking complete.")
    os.remove(xz_path)

def download_file(url, output_file):
    
    print(f"Downloading file from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Error: Failed to download file. HTTP status code: {response.status_code}")
        print(f"Response content: {response.text[:200]}")
        return False
    # Decide si el archivo es de texto o binario por la extensión
    text_exts = ('.txt', '.tsv', '.csv', '.json', '.xml')
    if any(output_file.endswith(ext) for ext in text_exts):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
    else:
        with open(output_file, 'wb') as f:
            shutil.copyfileobj(response.raw, f)
    print("Download complete.")
    return True


def unzip_file(gz, file):

    print(f"Unzipping file {gz} ...")
    with gzip.open(gz, 'rb') as f_in:
        with open(file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print("Unzipping complete.")

    os.remove(gz)
    print("Zipped file removed.")


def percentage_coroutine(to_process, print_on_percent = 0.005):
    print ("Starting progress percentage monitor...")

    processed = 0
    count = 0
    print_count = to_process*print_on_percent
    while True:
        yield #waits for the next send call
        # the value sent (None) is not used because the yield statement is not assigned to a variable (e.g., data = yield).
        processed += 1
        count += 1
        if (count >= print_count):
            count = 0
            pct = (float(processed)/float(to_process))*100

            print ("{}% finished".format(pct))

def trace_progress(func, progress = None):
    
    def callf(*args, **kwargs):
        
        if (progress is not None):
            progress.send(None)

        return func(*args, **kwargs)

    return callf

def get_file_structures(root_folder):
    """
    Returns:
        list: A list of lists, where each inner list represents the
              path components to a file.
    """
    all_paths = []
    
    for dirpath, _, filenames in os.walk(root_folder):
        
        for file_name in filenames:
    
            relative_dir = os.path.relpath(dirpath, root_folder)
            
            if relative_dir == '.':
                path_components = []
            else:
                path_components = relative_dir.split(os.path.sep)
            
            path_components.append(file_name) # journal name+ (can be several subfolders) / year/ file
            all_paths.append(path_components)
            
    return all_paths