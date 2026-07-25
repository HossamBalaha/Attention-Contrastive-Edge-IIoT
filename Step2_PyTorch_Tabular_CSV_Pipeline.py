from HMB.Examples.PyTorch_Tabular_CSV_Pipeline import *

# If executed as a script, run the Run function.
if (__name__ == "__main__"):
  # Update default matplotlib settings to configured preferences.
  UpdateMatplotlibSettings()

  # Parse command-line arguments provided to the script into an args namespace.
  args = GetArgs()
  # Validate parsed arguments to ensure required inputs and correct types.
  ValidateArgs(args)
  # Extract the data directory path from the parsed arguments.
  dataDir = args.dataDir
  # # Build a list of CSV file paths found inside the provided data directory.
  # csvFiles = [os.path.join(dataDir, f) for f in os.listdir(dataDir) if (f.lower().endswith(".csv"))]
  # # Raise an error when no CSV files were discovered in the specified directory.
  # if (len(csvFiles) == 0):
  #   # Notify the user that no CSV files were found and abort execution.
  #   raise RuntimeError(f"No CSV files found in data directory: {dataDir}")
  trainFiles = [
    os.path.join(dataDir, f)
    for f in os.listdir(dataDir)
    if ("Train" in f and f.lower().endswith(".csv"))
  ]
  if (len(trainFiles) == 0):
    # Notify the user that no training CSV files were found and abort execution.
    raise RuntimeError(f"No training CSV files found in data directory: {dataDir}")

  # Read the project-wide configuration JSON into a dictionary for runtime options.
  configs = ReadProjectConfig(args.configPath)
  # Get the list of model names defined in the configuration file.
  modelsNames = configs.get("Models", [])

  # Transfer commonly used CLI argument values into local variables for convenience.
  labelColumn = args.labelColumn
  dropFirstColumn = args.dropFirstColumn
  noOfTrials = args.noOfTrials
  saveDir = args.saveDir
  phase = args.phase

  fprint(
    f"Starting pipeline with phase: {phase}, dataDir: {dataDir}, saveDir: {saveDir}, "
    f"labelColumn: {labelColumn}, dropFirstColumn: {dropFirstColumn}, noOfTrials: {noOfTrials}"
  )

  # Execute the training phase when requested by the CLI argument.
  if (phase == "training"):
    # Iterate through each model configured in the project.
    for modelName in modelsNames:
      # Iterate through each CSV dataset discovered in the data directory.
      for trainFile in trainFiles[2:]:
        try:
          # Print a header announcing which model and dataset are being processed.
          fprint(f"\n\n=== Running pipeline for model: {modelName} on dataset: {trainFile} ===")
          # Compute a base file name for the dataset from the CSV filename.
          baseFileName = os.path.splitext(os.path.basename(trainFile))[0]
          # Determine the corresponding test file name by replacing "Train" with "Test" in the training file name.
          testFile = os.path.join(dataDir, os.path.basename(trainFile).replace("Train", "Test"))
          fprint(f"Using test file: {testFile}")
          # Determine the base save directory for this dataset and model.
          if (saveDir is None):
            # Use a default results folder name when no explicit save directory is provided.
            baseSaveDir = f"Results_{baseFileName}_{modelName}"
          else:
            # Use the provided save directory as the parent for per-dataset/model folders.
            baseSaveDir = os.path.join(saveDir, baseFileName, modelName)
          # Run the requested number of independent trials, creating subfolders when >1.
          for t in range(max(1, int(noOfTrials))):
            # Create a trial-specific subdirectory when multiple trials are requested.
            if (int(noOfTrials) > 1):
              trialSaveDir = os.path.join(baseSaveDir, f"Trial_{t + 1}")
            else:
              # Use the base save directory when only a single trial is requested.
              trialSaveDir = baseSaveDir
            # Ensure the trial save directory exists on disk.
            os.makedirs(trialSaveDir, exist_ok=True)
            # Print which trial is starting and where outputs will be saved.
            fprint(f"--> Trial {t + 1}/{noOfTrials}: saving to {trialSaveDir}")
            # Read the file and check the label column and if not, get the last column name.
            columns = pd.read_csv(trainFile, nrows=0).columns.tolist()
            if (labelColumn is None or labelColumn not in columns):
              # If the label column is not specified or not found, default to the last column in the CSV.
              labelColumn = columns[-1]
              fprint(f"Label column not specified or not found. Using last column: {labelColumn}")
            # Execute the per-CSV tabular pipeline for the current trial folder.
            RunTabularPipelineForCSV(
              modelName, None, configs, trialSaveDir, labelColumn,
              dropFirstColumn, trainPath=trainFile, testPath=testFile
            )
        except Exception as e:
          fprint(f"Error occurred while processing {trainFile}: {e}")
          import traceback

          # Print the full traceback for debugging purposes.
          traceback.print_exc()
    # Inform the user that the training phase has completed for all requested runs.
    fprint("Training phase completed for all models and CSV files.")

  # Execute only the statistical analysis phase when specified by the CLI argument.
  elif (phase == "statistical"):
    if (args.addFusedModel):
      # If the fused model flag is set, add a "FusedModel" entry to the list of models for analysis.
      modelsNames.append("FusedModel")
      fprint("Added 'FusedModel' to the list of models for statistical analysis.")
    if (args.fusedModelOnly):
      # If the fused model only flag is set, restrict the list of models to just "FusedModel".
      modelsNames = ["FusedModel"]
      fprint("Restricted statistical analysis to 'FusedModel' only.")
    # Iterate through configured models for performing statistical aggregation.
    for modelName in modelsNames:
      # Iterate through each CSV dataset to run statistics for their results.
      for csvFile in trainFiles:
        # Print a header announcing the statistical phase for the model and dataset.
        fprint(f"\n\n=== Running statistical phase for model: {modelName} on dataset: {csvFile} ===")
        # Determine the base file name for the dataset from its CSV path.
        baseFileName = os.path.splitext(os.path.basename(csvFile))[0]
        # Compute the base save directory for the dataset and model combination.
        if (saveDir is None):
          # Use a default results folder pattern when no explicit save directory is provided.
          baseSaveDir = f"Results_{baseFileName}_{modelName}"
        else:
          # Use the provided parent save directory to construct the per-dataset path.
          baseSaveDir = os.path.join(saveDir, baseFileName, modelName)
        # Run the statistical analysis phase using the assembled save directory path.
        RunStatisticalPhase(modelName, csvFile, configs, baseSaveDir)
    # Inform the user that the statistical phase completed for all requested runs.
    fprint("Statistical phase completed for all models and CSV files.")

  # Execute the fusion phase when specified by the CLI argument.
  elif (phase == "fused"):
    # Determine output fused model folder name from configs or default.
    outModelName = configs.get("FusedModelName", "FusedModel")
    # Experiments root where per-dataset folders are stored is saveDir by convention.
    experimentsRoot = saveDir if (saveDir is not None) else "Experiments"
    if (not os.path.isdir(experimentsRoot)):
      raise RuntimeError(f"Experiments directory for fusion not found: {experimentsRoot}")

    # Iterate over dataset subfolders and run fusion using `FuseModelsFromData`.
    for dataset in os.listdir(experimentsRoot):
      experimentPath = os.path.join(experimentsRoot, dataset)
      if (not os.path.isdir(experimentPath)):
        continue
      try:
        fprint(f"\n\n=== Running fusion for: {experimentPath} ===")
        FuseModelsFromData(experimentPath, modelsNames, outModelName=outModelName)
      except Exception as e:
        fprint(f"Warning: fusion failed for {experimentPath}: {e}")

    fprint("Fusion phase completed for all datasets.")

  # Execute the explainability phase when specified by the CLI argument.
  elif (phase == "explain"):
    # Iterate through each model configured in the project.
    for modelName in modelsNames:
      # Iterate through each CSV dataset discovered in the data directory.
      for csvFile in trainFiles:
        try:
          # Print a header announcing which model and dataset are being processed for explainability.
          fprint(f"\n\n=== Running explainability for model: {modelName} on dataset: {csvFile} ===")
          # Compute a base file name for the dataset from the CSV filename.
          baseFileName = os.path.splitext(os.path.basename(csvFile))[0]
          # Determine the base save directory for this dataset and model.
          if (saveDir is None):
            # Use a default results folder name when no explicit save directory is provided.
            baseSaveDir = f"Results_{baseFileName}_{modelName}"
          else:
            # Use the provided save directory as the parent for per-dataset/model folders.
            baseSaveDir = os.path.join(saveDir, baseFileName, modelName)

          # Search for the trials subdirectories under the base save directory to find all of them.
          trialDirs = []
          if (os.path.isdir(baseSaveDir)):
            for entry in os.listdir(baseSaveDir):
              entryPath = os.path.join(baseSaveDir, entry)
              if (os.path.isdir(entryPath) and entry.startswith("Trial_")):
                trialDirs.append(entryPath)
          # If no trial subdirectories are found, use the base save directory itself as the trial directory.
          if (len(trialDirs) == 0):
            trialDirs = [baseSaveDir]

          # Run the explainability phase for each trial directory found, which will look for artifacts in those directories.
          for t, trialDir in enumerate(trialDirs):
            # Ensure path exists (RunExplainabilityPhase will check), and announce the trial.
            fprint(f"--> Explainability for Trial {t + 1}/{len(trialDirs)}: looking for artifacts in {trialDir}")
            RunExplainabilityPhase(modelName, csvFile, configs, trialDir, labelColumn, dropFirstColumn)
        except Exception as e:
          # Print any exception that occurs while processing a model/dataset combination for explainability.
          fprint(f"Error occurred while processing for explainability: {e}")
    # Inform the user that the explainability phase has completed for all requested runs.
    fprint("Explainability phase completed for all models and CSV files.")
