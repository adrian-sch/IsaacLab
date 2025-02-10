import os
import shutil

# Define the path to the log directories
log_dir_path = 'logs/rl_games/robomaster_direct'
script_dir = os.path.dirname(os.path.realpath(__file__))

dirs_to_delete = []
# Iterate over all items in the log directory
dirs = os.listdir(os.path.join(script_dir,log_dir_path))
dirs.sort()
for item in dirs:
    item_path = os.path.join(log_dir_path, item)
    
    # Check if the item is a directory
    if os.path.isdir(item_path):
        nn_subdir_path = os.path.join(item_path, 'nn')
        
        # Check if the 'nn' subdirectory exists and is empty
        if os.path.exists(nn_subdir_path) and os.path.isdir(nn_subdir_path) and not os.listdir(nn_subdir_path):
            print(f"Found directory: {item_path}")
            dirs_to_delete.append(item_path)

if len(dirs_to_delete) == 0:
    print("No directories to delete.")
    exit()

print(f"Total number ob log dirs: {len(dirs)}")
print(f"Found {len(dirs_to_delete)} directories to delete.")
print("Do you want to proceed? (y/n)")
proceed = input()
if proceed.lower() == 'y':
    for dir_to_delete in dirs_to_delete:
        shutil.rmtree(dir_to_delete)
    print("Directories deleted.")
else:
    print("No directories were deleted.")
