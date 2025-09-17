#!/usr/bin/env python3
"""
Script to create a balanced dataset with specific requirements:
- Balanced Left (0) and Right (1) Coronary: ~100k each
- For coronary arteries: 75% diagnostic (stent_presence=0) and 25% with stent (stent_presence=1)
- Include ALL Graft samples (class 3)
- Proper train/val split by patient ID to prevent data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse
import cv2
import os
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def validate_video(video_path, min_frames=5, min_size_kb=50):
    """
    Check if a video file is valid.
    
    Args:
        video_path: Path to video file
        min_frames: Minimum number of frames required (default: 5)
        min_size_kb: Minimum file size in KB (default: 50KB)
    
    Returns:
        tuple: (is_valid, frame_count, error_msg)
    """
    # Check if file exists
    if not os.path.exists(video_path):
        return False, 0, "File not found"
    
    # Check file size (quick check before opening)
    file_size_kb = os.path.getsize(video_path) / 1024
    if file_size_kb < min_size_kb:
        return False, 0, f"File too small ({file_size_kb:.1f}KB)"
    
    # Try to open video and check properties
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, 0, "Cannot open video"
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        cap.release()
        
        # Validate properties
        if frame_count == 0:
            return False, frame_count, f"Video has no frames: {video_path} (frame_count=0)"
        elif frame_count < min_frames:
            return False, frame_count, f"Too few frames ({frame_count})"
        if width <= 0 or height <= 0:
            return False, frame_count, f"Invalid dimensions ({width}x{height})"
        if fps <= 0:
            return False, frame_count, f"Invalid FPS ({fps})"
        
        return True, frame_count, None
        
    except Exception as e:
        return False, 0, f"Error: {str(e)}"


def balance_coronary_samples(df, target_per_class=100000, stent_ratio=0.25):
    """
    Balance Left and Right Coronary samples with specific stent distribution.
    Keep ALL other classes except class 2.
    
    Args:
        df: DataFrame with main_structure_class and stent_presence_class columns
        target_per_class: Target number of samples for each coronary class
        stent_ratio: Ratio of samples with stent (stent_presence_class=1)
    
    Returns:
        Balanced DataFrame
    """
    balanced_dfs = []
    
    # Define the mapping for clarity
    structure_mapping = {
        0: "Left Coronary",
        1: "Right Coronary",
        2: "Other (EXCLUDED)",
        3: "Graft",
        4: "Catheter",
        5: "Femoral",
        6: "LV",
        7: "TAVR",
        8: "Aorta",
        9: "Radial",
        10: "TEE probe",
        11: "Pigtail"
    }
    
    print("\n" + "="*60)
    print("Balancing Dataset")
    print("="*60)
    
    # First, exclude class 2
    df = df[df['main_structure_class'] != 2].copy()
    print(f"Excluded class 2 (Other)")
    
    # Process Left and Right Coronary (classes 0 and 1)
    for coronary_class in [0, 1]:
        class_name = structure_mapping.get(coronary_class, f"Class {coronary_class}")
        print(f"\nProcessing {class_name} (class {coronary_class}):")
        
        # Get all samples for this coronary class
        class_df = df[df['main_structure_class'] == coronary_class].copy()
        print(f"  Total available: {len(class_df):,}")
        
        if 'stent_presence_class' not in class_df.columns:
            print(f"  Warning: stent_presence_class not found, using all samples")
            # If no stent column, just sample the target amount
            if len(class_df) > target_per_class:
                class_df = class_df.sample(n=target_per_class, random_state=42)
            balanced_dfs.append(class_df)
            continue
        
        # Separate by stent presence
        no_stent = class_df[class_df['stent_presence_class'] == 0]
        with_stent = class_df[class_df['stent_presence_class'] == 1]
        
        print(f"  - Without stent (diagnostic): {len(no_stent):,}")
        print(f"  - With stent: {len(with_stent):,}")
        
        # Calculate target numbers
        target_with_stent = int(target_per_class * stent_ratio)
        target_no_stent = target_per_class - target_with_stent
        
        print(f"  Target distribution:")
        print(f"  - Without stent: {target_no_stent:,} (75%)")
        print(f"  - With stent: {target_with_stent:,} (25%)")
        
        # Sample from each group
        sampled_no_stent = no_stent
        sampled_with_stent = with_stent
        
        # Sample diagnostic (no stent) cases
        if len(no_stent) > target_no_stent:
            sampled_no_stent = no_stent.sample(n=target_no_stent, random_state=42)
            print(f"  Sampled {target_no_stent:,} from {len(no_stent):,} diagnostic cases")
        else:
            print(f"  Using all {len(no_stent):,} diagnostic cases (less than target)")
        
        # Sample stent cases
        if len(with_stent) > target_with_stent:
            sampled_with_stent = with_stent.sample(n=target_with_stent, random_state=42)
            print(f"  Sampled {target_with_stent:,} from {len(with_stent):,} stent cases")
        else:
            print(f"  Using all {len(with_stent):,} stent cases (less than target)")
        
        # Combine
        class_balanced = pd.concat([sampled_no_stent, sampled_with_stent])
        print(f"  Final count for {class_name}: {len(class_balanced):,}")
        
        balanced_dfs.append(class_balanced)
    
    # Add ALL samples from other classes (except 2 which is already excluded)
    all_other_classes = sorted(df['main_structure_class'].unique())
    other_classes = [c for c in all_other_classes if c not in [0, 1, 2]]  # Exclude 0, 1 (already processed) and 2
    
    print(f"\nAdding ALL samples from other classes:")
    for other_class in other_classes:
        class_name = structure_mapping.get(other_class, f"Class {other_class}")
        other_df = df[df['main_structure_class'] == other_class].copy()
        print(f"  {class_name} (class {other_class}): {len(other_df):,} samples")
        balanced_dfs.append(other_df)
    
    # Combine all balanced classes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the combined dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df


def main():
    parser = argparse.ArgumentParser(description="Create balanced dataset with specific requirements")
    parser.add_argument(
        "--input_file",
        type=str,
        default="data/deepcoro_split_dataset_clean.parquet",
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/deepcoro_balanced_dataset.parquet",
        help="Output parquet file path"
    )
    parser.add_argument(
        "--target_coronary",
        type=int,
        default=100000,
        help="Target samples per coronary class (default: 100000)"
    )
    parser.add_argument(
        "--stent_ratio",
        type=float,
        default=0.25,
        help="Ratio of samples with stent for coronary classes (default: 0.25)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training set ratio (default: 0.8)"
    )
    parser.add_argument(
        "--patient_id_col",
        type=str,
        default="PatientID",
        help="Column name for patient ID"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--skip_video_validation",
        action="store_true",
        help="Skip video validation (faster but not recommended)"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=4,
        help="Number of threads for parallel video validation"
    )
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input_file}")
    df = pd.read_parquet(args.input_file)
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Total samples: {len(df):,}")
    
    # Show original distribution
    if 'main_structure_class' in df.columns:
        print("\nOriginal main_structure_class distribution:")
        print(df['main_structure_class'].value_counts().sort_index())
    else:
        print("ERROR: main_structure_class column not found!")
        return
    
    # Show stent distribution if available
    if 'stent_presence_class' in df.columns:
        print("\nOriginal stent_presence_class distribution:")
        print(df['stent_presence_class'].value_counts())
    
    # Balance the dataset
    balanced_df = balance_coronary_samples(
        df, 
        target_per_class=args.target_coronary,
        stent_ratio=args.stent_ratio
    )
    
    print("\n" + "="*60)
    print("Balanced Dataset Summary")
    print("="*60)
    print(f"Total samples after balancing: {len(balanced_df):,}")
    
    print("\nBalanced main_structure_class distribution:")
    print(balanced_df['main_structure_class'].value_counts().sort_index())
    
    if 'stent_presence_class' in balanced_df.columns:
        print("\nStent distribution by class:")
        for cls in sorted(balanced_df['main_structure_class'].unique()):
            class_df = balanced_df[balanced_df['main_structure_class'] == cls]
            if 'stent_presence_class' in class_df.columns:
                stent_dist = class_df['stent_presence_class'].value_counts()
                total = len(class_df)
                print(f"\nClass {cls}:")
                for stent_val, count in stent_dist.items():
                    pct = (count/total)*100
                    print(f"  Stent={stent_val}: {count:,} ({pct:.1f}%)")
    
    # Validate videos if not skipped
    if not args.skip_video_validation:
        print("\n" + "="*60)
        print("Validating Video Files")
        print("="*60)
        
        video_col = 'FileName' if 'FileName' in balanced_df.columns else None
        if video_col:
            print(f"Found video path column: {video_col}")
            
            valid_mask = []
            bad_videos = []
            
            print(f"Checking video files using {args.num_threads} threads...")
            
            video_paths = balanced_df[video_col].tolist()
            indices = balanced_df.index.tolist()
            
            def validate_video_with_index(video_path, idx):
                is_valid, frames, error = validate_video(video_path)
                return idx, is_valid, frames, error, video_path
            
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                future_to_video = {
                    executor.submit(validate_video_with_index, video_path, idx): (video_path, idx)
                    for video_path, idx in zip(video_paths, indices)
                }
                
                for future in tqdm(concurrent.futures.as_completed(future_to_video),
                                 total=len(future_to_video),
                                 desc="Validating videos"):
                    idx, is_valid, frames, error, video_path = future.result()
                    
                    if is_valid:
                        valid_mask.append(idx)
                    else:
                        bad_videos.append((video_path, error))
            
            # Apply the mask
            df_before = len(balanced_df)
            balanced_df = balanced_df.loc[valid_mask].copy()
            df_after = len(balanced_df)
            
            print(f"\nRemoved {df_before - df_after} bad videos")
            print(f"Dataset shape after video validation: {balanced_df.shape}")
            
            if bad_videos and len(bad_videos) <= 20:
                print(f"\nBad videos found:")
                for path, error in bad_videos[:20]:
                    print(f"  {path}: {error}")
    
    # Create train/validation split by patient ID
    print("\n" + "="*60)
    print("Creating Train/Validation Split")
    print("="*60)
    
    # Find patient ID column
    patient_col = None
    possible_names = [args.patient_id_col, 'PatientID', 'Patient_ID', 'patient_id', 
                      'PATIENT_ID', 'PatientId', 'patient', 'Patient', 'MRN', 'mrn',
                      'patient_id_anon']
    
    for col_name in possible_names:
        if col_name in balanced_df.columns:
            patient_col = col_name
            print(f"Found patient ID column: {patient_col}")
            break
    
    if patient_col is None:
        print("Warning: Patient ID column not found. Using row-based splitting...")
        balanced_df['Split'] = 'TRAIN'
        train_size = int(len(balanced_df) * args.train_ratio)
        balanced_df.loc[train_size:, 'Split'] = 'VAL'
    else:
        # Get unique patient IDs
        unique_patients = balanced_df[patient_col].unique()
        n_patients = len(unique_patients)
        print(f"Number of unique patients: {n_patients:,}")
        
        # Split patients into train/val
        train_patients, val_patients = train_test_split(
            unique_patients, 
            train_size=args.train_ratio,
            random_state=args.seed,
            shuffle=True
        )
        
        print(f"Train patients: {len(train_patients):,} ({len(train_patients)/n_patients*100:.1f}%)")
        print(f"Validation patients: {len(val_patients):,} ({len(val_patients)/n_patients*100:.1f}%)")
        
        # Create Split column
        balanced_df['Split'] = 'VAL'  # Default to validation
        balanced_df.loc[balanced_df[patient_col].isin(train_patients), 'Split'] = 'TRAIN'
        
        # Verify no patient overlap
        train_patients_check = balanced_df[balanced_df['Split'] == 'TRAIN'][patient_col].unique()
        val_patients_check = balanced_df[balanced_df['Split'] == 'VAL'][patient_col].unique()
        overlap = set(train_patients_check) & set(val_patients_check)
        
        if overlap:
            print(f"WARNING: Found {len(overlap)} patients in both train and val!")
        else:
            print("✓ No patient overlap between train and validation sets")
    
    # Print split statistics
    print("\n" + "="*60)
    print("Final Dataset Statistics")
    print("="*60)
    
    print("\nSplit distribution:")
    print(balanced_df['Split'].value_counts())
    
    print("\nClass distribution by split:")
    for split in ['TRAIN', 'VAL']:
        split_df = balanced_df[balanced_df['Split'] == split]
        print(f"\n{split} split ({len(split_df):,} samples):")
        print(split_df['main_structure_class'].value_counts().sort_index())
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the balanced dataset
    print(f"\nSaving to: {args.output_file}")
    balanced_df.to_parquet(args.output_file, index=False)
    
    print(f"\n✅ Successfully created balanced dataset!")
    print(f"   Total samples: {len(balanced_df):,}")
    print(f"   File saved to: {args.output_file}")
    
    # Final summary
    print("\n" + "="*60)
    print("Summary of Balanced Dataset:")
    print("="*60)
    structure_mapping = {
        0: "Left Coronary",
        1: "Right Coronary",
        3: "Graft",
        4: "Catheter",
        5: "Femoral",
        6: "LV",
        7: "TAVR",
        8: "Aorta",
        9: "Radial",
        10: "TEE probe",
        11: "Pigtail"
    }
    summary = balanced_df['main_structure_class'].value_counts().sort_index()
    for cls, count in summary.items():
        pct = (count/len(balanced_df))*100
        class_name = structure_mapping.get(cls, f"Class {cls}")
        print(f"{class_name:15} (class {cls}): {count:7,} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()