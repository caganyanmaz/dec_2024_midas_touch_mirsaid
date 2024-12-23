import subprocess

# List of scripts to run in sequence
# train_models.py and test_models.py are commented out as they are not required for the API
# They are only used once to create the models for research purposes, but after that, the same models can be used
scripts = ["dataset_prep.py", "feature_eng.py", "co_feature_eng.py", "knowledge_graph.py", "add_graph_info_to_investors.py"] # "train_models.py"] "test_models.py"]

for script in scripts:
    try:
        print(f"Running {script}...")
        # Execute the script
        subprocess.run(['python', script], check=True)
        print(f"Completed {script} successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}. Exiting pipeline.\n")
        break

print("Pipeline execution completed.")