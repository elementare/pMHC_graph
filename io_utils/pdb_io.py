import os
def list_pdb_files(pdb_dir):
    pdb_files = [f for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    if not pdb_files:
        print("No PDB files found in the directory.")
        return []
    
    print("Select PDB files to associate:")
    print("1 - All")
    for i, pdb_file in enumerate(pdb_files, start=2):
        print(f"{i} - {pdb_file}")
    
    return pdb_files

def get_user_selection(pdb_files, pdb_dir):

    selection = input("\nEnter the numbers of the proteins to associate (comma-separated) or '1' to select all: ")
    
    if "1" in selection.split(","):
        return [(os.path.join(pdb_dir, pdb), pdb) for pdb in pdb_files]
    
    try:
        selected_numbers = [int(num.strip()) for num in selection.split(',')]
        selected_files = [(os.path.join(pdb_dir, pdb_files[i-2]), pdb_files[i-2]) for i in selected_numbers if i > 1]  
        return selected_files
    except (ValueError, IndexError):
        print("Invalid input. Please enter valid numbers.")
        return get_user_selection(pdb_files)