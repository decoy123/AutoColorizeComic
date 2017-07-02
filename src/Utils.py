import os

# Get file list on selected directory.
def getFileList(self, directory):
    file_list = []
    for (root, dirs, files) in os.walk(directory):
        for file in files:
            target = os.path.join(root, file).replace("\\", "/")
            if os.path.isfile(target):
                file_list.append(target)
    return file_list
