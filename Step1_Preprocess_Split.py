# Import the os module for path operations.
import os
# Import the argparse module for command-line argument parsing.
import argparse
# Import the logging module for logging utilities.
import logging
# Import the joblib module for object serialization.
import joblib
# Import the matplotlib module for plotting.
import matplotlib
# Import the numpy module for numerical operations.
import numpy as np
# Import the pandas module for data manipulation.
import pandas as pd
# Import the Path class from pathlib for object-oriented path handling.
from pathlib import Path
# Import the LabelEncoder from sklearn for categorical encoding.
from sklearn.preprocessing import LabelEncoder
# Import train_test_split and StratifiedKFold for dataset splitting.
from sklearn.model_selection import train_test_split, StratifiedKFold
# Import the custom Logger utility.
from HMB.Utils import Logger
# Import the custom PlotClassDistribution utility.
from HMB.PlotsHelper import PlotClassDistribution
# Import the custom dataset helper utilities.
from HMB.DatasetsHelper import FindCsvFile, SafeReadCsv

# Set the matplotlib backend to Agg for non-interactive rendering.
matplotlib.use("Agg")
# Create the global logger instance.
logger = Logger()

# Define default columns to drop as a list.
DEFAULT_DROP_COLUMNS = [
  "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
  "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
  "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
  "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"
]


def MapTo6Class(attackStr):
  # Normalize the attack string to lowercase.
  s = str(attackStr).lower()
  # Map normal attacks.
  if ("normal" in s):
    return "Normal"
  # Map DDoS/DoS related attacks.
  if ("ddos" in s):
    return "DoS_DDoS"
  # Map fingerprinting or scan related attacks.
  if (("fingerprinting" in s) or ("scan" in s)):
    return "Information_Gathering"
  # Map MITM related attacks.
  if ("mitm" in s):
    return "MITM"
  # Map injection class attacks.
  if (("xss" in s) or ("sql" in s) or ("upload" in s)):
    return "Injection"
  # Map malware-like attacks.
  if (("backdoor" in s) or ("password" in s) or ("ransomware" in s)):
    return "Malware"
  # Default fallback mapping.
  return "Other"


def Preprocess(df, dropColumns=None, categoricalCols=None):
  # Use provided dropColumns or defaults.
  dropColumns = (dropColumns or DEFAULT_DROP_COLUMNS)
  # Log the action about to drop configured columns.
  logger.info("Dropping configured columns if they exist...")
  # Work on a copy of the dataframe to avoid modifying the caller object.
  df = df.copy()
  # Compute actual columns to drop that exist in the dataframe.
  colsToDrop = [c for c in dropColumns if c in df.columns]
  # Drop columns when any are present.
  logger.info(f"Columns to drop: {colsToDrop}")
  if (colsToDrop):
    df.drop(columns=colsToDrop, inplace=True)
    logger.info(f"Dropped columns: {colsToDrop}")
  else:
    logger.info("No configured columns found to drop.")

  # Log that rows with missing values and duplicates will be removed.
  logger.info("Dropping rows with any missing values and duplicates...")

  # Drop rows with any NaN values.
  logger.info(f"Initial shape before dropping: {df.shape}")
  df.dropna(axis=0, how="any", inplace=True)
  logger.info(f"Shape after dropping NaNs: {df.shape}")

  # Drop duplicate rows keeping the first occurrence.
  logger.info(f"Shape before dropping duplicates: {df.shape}")
  df.drop_duplicates(keep="first", inplace=True)
  logger.info(f"Shape after dropping duplicates: {df.shape}")

  # Ensure required label columns are present.
  if (("Attack_type" not in df.columns) or ("Attack_label" not in df.columns)):
    raise KeyError("Expected columns 'Attack_type' and 'Attack_label' in the dataset")

  # Extract 15-class labels and fit a LabelEncoder.
  y15 = df["Attack_type"].values
  le15 = LabelEncoder()
  y15Encoded = le15.fit_transform(y15)
  numClasses15 = len(le15.classes_)

  # Extract 2-class labels as provided by the dataset.
  y2 = df["Attack_label"].values

  # Map to 6-class labels and fit a LabelEncoder.
  logger.info("Mapping to 6-class labels...")
  df["Attack_type_6"] = df["Attack_type"].apply(MapTo6Class)
  le6 = LabelEncoder()
  y6Encoded = le6.fit_transform(df["Attack_type_6"].values)
  numClasses6 = len(le6.classes_)

  # Build the feature dataframe by dropping label columns.
  Xdf = df.drop(columns=[c for c in ["Attack_label", "Attack_type", "Attack_type_6"] if (c in df.columns)])

  # Detect object/string columns and combine with any provided categorical columns.
  detectedCat = [c for c, t in Xdf.dtypes.items() if t == "object"]
  categoricalCols = ((categoricalCols or []) + detectedCat)
  categoricalCols = list(dict.fromkeys(categoricalCols))
  categoricalCols = [c for c in categoricalCols if (c in Xdf.columns)]
  logger.info(f"Detected categorical columns: {detectedCat}")
  logger.info(f"Final categorical columns to encode (after combining with provided): {categoricalCols}")

  # One-hot encode categorical columns when present.
  if (categoricalCols):
    logger.info(f"Encoding categorical columns with get_dummies: {categoricalCols}")
    Xdf = pd.get_dummies(Xdf, columns=categoricalCols)
    logger.info(f"Shape after one-hot encoding: {Xdf.shape}")
  else:
    logger.info("No categorical columns found to encode.")

  # Convert all remaining entries to numeric and replace NaNs with zeros.
  logger.info("Converting feature dataframe to numeric and filling remaining NaNs with 0")
  Xdf = Xdf.apply(pd.to_numeric, errors="coerce").fillna(0)
  logger.info("Completed numeric conversion and NaN handling for features.")
  logger.info(f"Feature dataframe shape after processing: {Xdf.shape}")

  # Convert features to numpy array.
  X = Xdf.values

  # Log the shapes of produced arrays.
  logger.info(f"Shapes - X: {X.shape}  y2: {y2.shape}  y6: {y6Encoded.shape}  y15: {y15Encoded.shape}")

  # Log a sample of the processed feature dataframe.
  logger.info(f"Sample of processed feature dataframe:\n{Xdf.head()}")
  logger.info(f"Sample of 2-class labels (y2): {y2[:5]}")
  logger.info(f"Sample of 6-class encoded labels (y6): {y6Encoded[:5]} (classes: {le6.classes_})")
  logger.info(f"Sample of 15-class encoded labels (y15): {y15Encoded[:5]} (classes: {le15.classes_})")

  # Build metadata dictionary.
  meta = {
    "FeatureNames": list(Xdf.columns),
    "Le6Classes"  : list(le6.classes_),
    "Le15Classes" : list(le15.classes_),
    "NumClasses6" : numClasses6,
    "NumClasses15": numClasses15,
  }

  # Return features, the processed feature dataframe, labels, encoders and metadata.
  return X, Xdf, y2, y6Encoded, y15Encoded, le6, le15, meta


def SaveProcessed(outputDir, prefix, **kwargs):
  # Ensure the output directory exists.
  outputDir.mkdir(parents=True, exist_ok=True)
  # Iterate over provided objects and save appropriately.
  for name, obj in kwargs.items():
    # Build the output path using prefix and name.
    outPath = outputDir / f"{prefix}_{name}"
    # Save numpy arrays using compressed npz files.
    if (isinstance(obj, np.ndarray)):
      np.savez_compressed(str(outPath) + ".npz", obj=obj)
      logger.info(f"Saved numpy array: {outPath}.npz")
    # Save lists and dictionaries with joblib.
    elif (isinstance(obj, (list, dict))):
      joblib.dump(obj, str(outPath) + ".joblib")
      logger.info(f"Saved joblib object: {outPath}.joblib")
    else:
      # Fallback saving using joblib for arbitrary objects.
      joblib.dump(obj, str(outPath) + ".joblib")
      logger.info(f"Saved joblib object: {outPath}.joblib")


def NormalizeBoolColumns(df):
  # Convert actual boolean dtypes to integers.
  boolCols = list(df.select_dtypes(include=["bool"]).columns)
  if (boolCols):
    for c in boolCols:
      df[c] = df[c].astype(int)
  # Convert string columns that only contain boolean-like text to integers.
  objCols = list(df.select_dtypes(include=["object"]).columns)
  for c in objCols:
    uniques = set(df[c].dropna().astype(str).unique())
    lowered = {u.lower() for u in uniques}
    if (lowered <= {"true", "false"}):
      df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0}).astype(int)
  return df


def Main(args):
  # Convert input and output paths to Path objects.
  inputDir = Path(args.input_dir)
  outputDir = Path(args.output_dir)

  # Determine if a specific input file path was provided.
  csvPath = args.input_file
  # Use provided file or search the directory for a CSV file.
  if (csvPath):
    csvPath = Path(csvPath)
    # Raise an error if the provided file does not exist.
    if (not csvPath.exists()):
      raise FileNotFoundError(f"Provided input file does not exist: {csvPath}")
  else:
    csvPath = FindCsvFile(inputDir)

  # Read the CSV into a dataframe.
  df = SafeReadCsv(csvPath)

  # Preprocess the dataframe to extract features and labels.
  X, Xdf, y2, y6, y15, le6, le15, meta = Preprocess(df, categoricalCols=args.categorical_cols)

  # Ensure the output directory exists.
  outputDir.mkdir(parents=True, exist_ok=True)

  # Save label encoders and metadata.
  joblib.dump(le6, outputDir / "Le6.joblib")
  joblib.dump(le15, outputDir / "Le15.joblib")
  joblib.dump(meta, outputDir / "Meta.joblib")
  logger.info("Saved label encoders and metadata")

  # Build a features-only dataframe for CSV export.
  featuresDf = pd.DataFrame(X, columns=meta["FeatureNames"])

  # Normalize boolean-like columns in the features dataframe.
  featuresDf = NormalizeBoolColumns(featuresDf)
  # Ensure all feature columns are numeric (float32) for consistent storage.
  featuresDf = featuresDf.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

  # Plot class distribution bar plots for the dataset before persisting CSV outputs.
  try:
    PlotClassDistribution(
      y2, outputDir / "Distribution_Attack_label.pdf",
      title="Class Distribution: Attack_label (2-class)"
    )
    PlotClassDistribution(
      le6.inverse_transform(y6), outputDir / "Distribution_Attack_type_6.pdf",
      title="Class Distribution: Attack_type_6 (6-class)"
    )
    PlotClassDistribution(
      le15.inverse_transform(y15), outputDir / "Distribution_Attack_type_15.pdf",
      title="Class Distribution: Attack_type_15 (15-class)"
    )
  except Exception as e:
    logger.warning(f"Plotting class distributions failed: {e}")

  # Save a features-only CSV for convenience.
  featuresDf.to_csv(outputDir / "ProcessedFeatures.csv", index=False)

  # Build and save a processed CSV that includes only the Attack_label column.
  processedWithAttackLabel = featuresDf.copy()
  processedWithAttackLabel["Attack_label"] = y2
  processedWithAttackLabel = NormalizeBoolColumns(processedWithAttackLabel)
  processedWithAttackLabel.to_csv(outputDir / "ProcessedWithAttackLabel.csv", index=False)

  # Build and save a processed CSV that includes only the Attack_type_6 label.
  processedWithAttackType6 = featuresDf.copy()
  processedWithAttackType6["Attack_type_6"] = le6.inverse_transform(y6)
  processedWithAttackType6 = NormalizeBoolColumns(processedWithAttackType6)
  processedWithAttackType6.to_csv(outputDir / "ProcessedWithAttackType6.csv", index=False)

  # Build and save a processed CSV that includes only the Attack_type_15 label.
  processedWithAttackType15 = featuresDf.copy()
  processedWithAttackType15["Attack_type_15"] = le15.inverse_transform(y15)
  processedWithAttackType15 = NormalizeBoolColumns(processedWithAttackType15)
  processedWithAttackType15.to_csv(outputDir / "ProcessedWithAttackType15.csv", index=False)

  # Save feature names.
  joblib.dump(meta["FeatureNames"], outputDir / "FeatureNames.joblib")

  # Check if train/test split is requested.
  if (args.test_size is not None):
    testFolder = outputDir / "TrainTestSplit"
    os.makedirs(testFolder, exist_ok=True)
    # Log the train/test split action.
    logger.info(f"Performing train/test split with test_size={args.test_size}...")
    # Perform stratified train/test split using the 15-class labels.
    xTrain, xTest, y2Train, y2Test, y6Train, y6Test, y15Train, y15Test = train_test_split(
      X, y2, y6, y15, test_size=args.test_size, random_state=args.random_state, stratify=y15
    )
    # Save the train/test split arrays as NPZ files.
    SaveProcessed(
      testFolder,
      "Split",
      XTrain=xTrain,
      XTest=xTest,
      Y2Train=y2Train,
      Y2Test=y2Test,
      Y6Train=y6Train,
      Y6Test=y6Test,
      Y15Train=y15Train,
      Y15Test=y15Test
    )

    # Build a dataframe for the training split features.
    trainSplitDf = pd.DataFrame(xTrain, columns=meta["FeatureNames"])

    # Build and save the training split CSV with 2-class labels.
    trainAttackLabelDf = trainSplitDf.copy()
    # Assign the 2-class labels to the training split dataframe.
    trainAttackLabelDf["Attack_label"] = y2Train
    # Normalize boolean-like columns in the training split dataframe.
    trainAttackLabelDf = NormalizeBoolColumns(trainAttackLabelDf)
    # Save the training split dataframe with 2-class labels to a CSV file.
    trainAttackLabelDf.to_csv(testFolder / "SplitTrainAttackLabel.csv", index=False)

    # Build and save the training split CSV with 6-class labels.
    trainAttackType6Df = trainSplitDf.copy()
    # Assign the 6-class textual labels to the training split dataframe.
    trainAttackType6Df["Attack_type_6"] = le6.inverse_transform(y6Train)
    # Normalize boolean-like columns in the training split dataframe.
    trainAttackType6Df = NormalizeBoolColumns(trainAttackType6Df)
    # Save the training split dataframe with 6-class labels to a CSV file.
    trainAttackType6Df.to_csv(testFolder / "SplitTrainAttackType6.csv", index=False)

    # Build and save the training split CSV with 15-class labels.
    trainAttackType15Df = trainSplitDf.copy()
    # Assign the 15-class textual labels to the training split dataframe.
    trainAttackType15Df["Attack_type_15"] = le15.inverse_transform(y15Train)
    # Normalize boolean-like columns in the training split dataframe.
    trainAttackType15Df = NormalizeBoolColumns(trainAttackType15Df)
    # Save the training split dataframe with 15-class labels to a CSV file.
    trainAttackType15Df.to_csv(testFolder / "SplitTrainAttackType15.csv", index=False)

    # Build a dataframe for the testing split features.
    testSplitDf = pd.DataFrame(xTest, columns=meta["FeatureNames"])

    # Build and save the testing split CSV with 2-class labels.
    testAttackLabelDf = testSplitDf.copy()
    # Assign the 2-class labels to the testing split dataframe.
    testAttackLabelDf["Attack_label"] = y2Test
    # Normalize boolean-like columns in the testing split dataframe.
    testAttackLabelDf = NormalizeBoolColumns(testAttackLabelDf)
    # Save the testing split dataframe with 2-class labels to a CSV file.
    testAttackLabelDf.to_csv(testFolder / "SplitTestAttackLabel.csv", index=False)

    # Build and save the testing split CSV with 6-class labels.
    testAttackType6Df = testSplitDf.copy()
    # Assign the 6-class textual labels to the testing split dataframe.
    testAttackType6Df["Attack_type_6"] = le6.inverse_transform(y6Test)
    # Normalize boolean-like columns in the testing split dataframe.
    testAttackType6Df = NormalizeBoolColumns(testAttackType6Df)
    # Save the testing split dataframe with 6-class labels to a CSV file.
    testAttackType6Df.to_csv(testFolder / "SplitTestAttackType6.csv", index=False)

    # Build and save the testing split CSV with 15-class labels.
    testAttackType15Df = testSplitDf.copy()
    # Assign the 15-class textual labels to the testing split dataframe.
    testAttackType15Df["Attack_type_15"] = le15.inverse_transform(y15Test)
    # Normalize boolean-like columns in the testing split dataframe.
    testAttackType15Df = NormalizeBoolColumns(testAttackType15Df)
    # Save the testing split dataframe with 15-class labels to a CSV file.
    testAttackType15Df.to_csv(testFolder / "SplitTestAttackType15.csv", index=False)

    # Log the completion of saving train/test split files.
    logger.info("Saved train/test split NPZ and CSV files.")

  # Check if k-fold cross-validation split is requested.
  if (args.k_folds is not None):
    # Log the k-fold cross-validation split action.
    logger.info(f"Performing {args.k_folds}-fold cross-validation split...")
    # Initialize the stratified k-fold splitter.
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
    # Iterate through each fold to generate splits.
    for foldIndex, (trainIdx, testIdx) in enumerate(skf.split(X, y15)):
      foldFolder = outputDir / f"Fold_{foldIndex}"
      os.makedirs(foldFolder, exist_ok=True)
      # Initialize a dictionary to hold fold data for NPZ saving.
      foldData = {}
      # Assign training features for the current fold to the dictionary.
      foldData[f"XTrainFold{foldIndex}"] = X[trainIdx]
      # Assign testing features for the current fold to the dictionary.
      foldData[f"XTestFold{foldIndex}"] = X[testIdx]
      # Assign training 2-class labels for the current fold to the dictionary.
      foldData[f"Y2TrainFold{foldIndex}"] = y2[trainIdx]
      # Assign testing 2-class labels for the current fold to the dictionary.
      foldData[f"Y2TestFold{foldIndex}"] = y2[testIdx]
      # Assign training 6-class labels for the current fold to the dictionary.
      foldData[f"Y6TrainFold{foldIndex}"] = y6[trainIdx]
      # Assign testing 6-class labels for the current fold to the dictionary.
      foldData[f"Y6TestFold{foldIndex}"] = y6[testIdx]
      # Assign training 15-class labels for the current fold to the dictionary.
      foldData[f"Y15TrainFold{foldIndex}"] = y15[trainIdx]
      # Assign testing 15-class labels for the current fold to the dictionary.
      foldData[f"Y15TestFold{foldIndex}"] = y15[testIdx]

      # Build a dataframe for the current fold training features.
      trainFoldDf = pd.DataFrame(X[trainIdx], columns=meta["FeatureNames"])

      # Build and save the training fold CSV with 2-class labels.
      trainFoldAttackLabelDf = trainFoldDf.copy()
      # Assign the 2-class labels to the current fold training dataframe.
      trainFoldAttackLabelDf["Attack_label"] = y2[trainIdx]
      # Normalize boolean-like columns in the current fold training dataframe.
      trainFoldAttackLabelDf = NormalizeBoolColumns(trainFoldAttackLabelDf)
      # Save the current fold training dataframe with 2-class labels to a CSV file.
      trainFoldAttackLabelDf.to_csv(foldFolder / f"SplitTrainFold{foldIndex}AttackLabel.csv", index=False)

      # Build and save the training fold CSV with 6-class labels.
      trainFoldAttackType6Df = trainFoldDf.copy()
      # Assign the 6-class textual labels to the current fold training dataframe.
      trainFoldAttackType6Df["Attack_type_6"] = le6.inverse_transform(y6[trainIdx])
      # Normalize boolean-like columns in the current fold training dataframe.
      trainFoldAttackType6Df = NormalizeBoolColumns(trainFoldAttackType6Df)
      # Save the current fold training dataframe with 6-class labels to a CSV file.
      trainFoldAttackType6Df.to_csv(foldFolder / f"SplitTrainFold{foldIndex}AttackType6.csv", index=False)

      # Build and save the training fold CSV with 15-class labels.
      trainFoldAttackType15Df = trainFoldDf.copy()
      # Assign the 15-class textual labels to the current fold training dataframe.
      trainFoldAttackType15Df["Attack_type_15"] = le15.inverse_transform(y15[trainIdx])
      # Normalize boolean-like columns in the current fold training dataframe.
      trainFoldAttackType15Df = NormalizeBoolColumns(trainFoldAttackType15Df)
      # Save the current fold training dataframe with 15-class labels to a CSV file.
      trainFoldAttackType15Df.to_csv(foldFolder / f"SplitTrainFold{foldIndex}AttackType15.csv", index=False)

      # Build a dataframe for the current fold testing features.
      testFoldDf = pd.DataFrame(X[testIdx], columns=meta["FeatureNames"])

      # Build and save the testing fold CSV with 2-class labels.
      testFoldAttackLabelDf = testFoldDf.copy()
      # Assign the 2-class labels to the current fold testing dataframe.
      testFoldAttackLabelDf["Attack_label"] = y2[testIdx]
      # Normalize boolean-like columns in the current fold testing dataframe.
      testFoldAttackLabelDf = NormalizeBoolColumns(testFoldAttackLabelDf)
      # Save the current fold testing dataframe with 2-class labels to a CSV file.
      testFoldAttackLabelDf.to_csv(foldFolder / f"SplitTestFold{foldIndex}AttackLabel.csv", index=False)

      # Build and save the testing fold CSV with 6-class labels.
      testFoldAttackType6Df = testFoldDf.copy()
      # Assign the 6-class textual labels to the current fold testing dataframe.
      testFoldAttackType6Df["Attack_type_6"] = le6.inverse_transform(y6[testIdx])
      # Normalize boolean-like columns in the current fold testing dataframe.
      testFoldAttackType6Df = NormalizeBoolColumns(testFoldAttackType6Df)
      # Save the current fold testing dataframe with 6-class labels to a CSV file.
      testFoldAttackType6Df.to_csv(foldFolder / f"SplitTestFold{foldIndex}AttackType6.csv", index=False)

      # Build and save the testing fold CSV with 15-class labels.
      testFoldAttackType15Df = testFoldDf.copy()
      # Assign the 15-class textual labels to the current fold testing dataframe.
      testFoldAttackType15Df["Attack_type_15"] = le15.inverse_transform(y15[testIdx])
      # Normalize boolean-like columns in the current fold testing dataframe.
      testFoldAttackType15Df = NormalizeBoolColumns(testFoldAttackType15Df)
      # Save the current fold testing dataframe with 15-class labels to a CSV file.
      testFoldAttackType15Df.to_csv(foldFolder / f"SplitTestFold{foldIndex}AttackType15.csv", index=False)

      # Save all k-fold split arrays as NPZ files.
      SaveProcessed(foldFolder, "Split", **foldData)

    # Log the completion of saving k-fold split files.
    logger.info(f"Saved {args.k_folds}-fold cross-validation split NPZ and CSV files.")

  # Log completion message with the output directory path.
  logger.info(f"All preprocessing complete. Files saved to: {outputDir}")


if (__name__ == "__main__"):
  # Create an argument parser for command line execution.
  parser = argparse.ArgumentParser(description="Preprocess Edge-IIoTset selected dataset")
  # Add argument for input directory to search for CSV files.
  parser.add_argument("--input_dir", type=str, default=".", help="Directory to search for input CSV files")
  # Add optional argument for a specific input file path.
  parser.add_argument(
    "--input_file",
    type=str,
    default=None,
    help="Path to a specific CSV file (overrides input_dir search)"
  )
  # Add argument for output directory to save processed outputs.
  parser.add_argument("--output_dir", type=str, default="ProcessedData", help="Directory to save processed outputs")
  # Add argument for test set proportion.
  parser.add_argument(
    "--test_size",
    type=float,
    default=None,
    help="Test set proportion for train/test split (e.g., 0.2). If None, train/test split is skipped."
  )
  # Add argument for k-fold cross-validation splits.
  parser.add_argument(
    "--k_folds",
    type=int,
    default=None,
    help="Number of folds for k-fold cross-validation (e.g., 5). If None, k-fold split is skipped."
  )
  # Add argument for random seed.
  parser.add_argument("--random_state", type=int, default=42, help="Random seed for splits")
  # Add optional list of categorical columns to force one-hot encoding.
  parser.add_argument(
    "--categorical_cols",
    nargs="*",
    default=None,
    help="List of categorical columns to one-hot (optional)"
  )
  # Parse the provided arguments.
  args = parser.parse_args()

  # Run the main processing pipeline with parsed arguments.
  Main(args)
