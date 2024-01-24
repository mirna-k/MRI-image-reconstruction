import os

def read_files_in_folder(folder_path):
    # Check if the provided path is a directory
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return None

    file_path = []

    for filename in os.listdir(folder_path):
        resulting_path = folder_path + '/' + filename
        file_path.append(resulting_path)
    return file_path

# Example usage
# folder_path = 'brain_data'
# resulting_string = read_files_in_folder(folder_path)
# print(resulting_string)

# for string in resulting_string:
#     if string:
#         print(string)
#     else:
#         print("Failed to read files.")