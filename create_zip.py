import os
import zipfile

def create_zip():
    print("Creating zip file of the cryptocurrency prediction app...")
    
    files_to_include = [
        "main.py",
        "data_retrieval.py",
        "ml_models.py",
        "utils.py",
        "visualizations.py", 
        "README.md",
        "pyproject.toml",
    ]
    
    # Include .streamlit folder
    if os.path.exists(".streamlit"):
        if os.path.exists(".streamlit/config.toml"):
            files_to_include.append(".streamlit/config.toml")
    
    # Include assets folder
    if os.path.exists("assets"):
        if os.path.exists("assets/custom.css"):
            files_to_include.append("assets/custom.css")
    
    # Create the zip file
    with zipfile.ZipFile("crypto_prediction_app.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
                print(f"Added {file} to zip")
            else:
                print(f"Warning: {file} not found, skipping")
    
    print("Zip file created successfully: crypto_prediction_app.zip")

if __name__ == "__main__":
    create_zip()