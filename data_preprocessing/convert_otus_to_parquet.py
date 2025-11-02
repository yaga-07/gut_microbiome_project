#!/usr/bin/env python3
"""
Script to convert otus.97.allinfo file into a queryable parquet file.
Extracts 97_* ID, DNA sequence, and bacteria taxonomy information.
"""

import pandas as pd
import re
from pathlib import Path


def extract_97_id(id_string):
    """Extract the 97_* ID from the ID string."""
    match = re.search(r'97_(\d+)', id_string)
    if match:
        return f"97_{match.group(1)}"
    return None


def process_otus_file(input_file, output_file, chunk_size=10000):
    """
    Process the otus.97.allinfo file and convert to parquet.
    
    Args:
        input_file: Path to the input otus.97.allinfo file
        output_file: Path to the output parquet file
        chunk_size: Number of lines to process at a time
    """
    print(f"Processing {input_file}...")
    
    data_chunks = []
    total_lines = 0
    skipped_lines = 0
    
    with open(input_file, 'r') as f:
        chunk_data = []
        
        for line_num, line in enumerate(f, 1):
            try:
                # Split by tabs
                parts = line.strip().split('\t')
                
                if len(parts) < 9:
                    print(f"Warning: Line {line_num} has fewer than 9 columns, skipping")
                    skipped_lines += 1
                    continue
                
                # Extract the required fields
                id_string = parts[0]
                dna_sequence = parts[6]  # Column 7 (0-indexed)
                bacteria_taxonomy = parts[8]  # Column 9 (0-indexed)
                
                # Extract 97_* ID
                otu_97_id = extract_97_id(id_string)
                
                if otu_97_id is None:
                    print(f"Warning: Line {line_num} has no 97_* ID, skipping")
                    skipped_lines += 1
                    continue
                
                # Add to chunk
                chunk_data.append({
                    'otu_97_id': otu_97_id,
                    'dna_sequence': dna_sequence,
                    'bacteria_taxonomy': bacteria_taxonomy
                })
                
                total_lines += 1
                
                # Process chunk if it reaches chunk_size
                if len(chunk_data) >= chunk_size:
                    data_chunks.append(pd.DataFrame(chunk_data))
                    chunk_data = []
                    
                    if total_lines % 50000 == 0:
                        print(f"Processed {total_lines:,} lines...")
                
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                skipped_lines += 1
                continue
        
        # Process remaining data
        if chunk_data:
            data_chunks.append(pd.DataFrame(chunk_data))
    
    print(f"\nCombining {len(data_chunks)} chunks...")
    
    # Combine all chunks
    if data_chunks:
        df = pd.concat(data_chunks, ignore_index=True)
        
        print(f"\nDataFrame created:")
        print(f"  Total records: {len(df):,}")
        print(f"  Unique 97_* IDs: {df['otu_97_id'].nunique():,}")
        print(f"  Skipped lines: {skipped_lines:,}")
        print(f"\nDataFrame info:")
        print(df.info())
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Save to parquet
        print(f"\nSaving to {output_file}...")
        df.to_parquet(output_file, index=False, compression='snappy')
        
        print(f"✓ Successfully saved parquet file!")
        
        # Show file size
        output_path = Path(output_file)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")
        
        return df
    else:
        print("No data to process!")
        return None


def query_parquet(parquet_file, otu_97_id):
    """
    Query the parquet file for a specific 97_* ID.
    
    Args:
        parquet_file: Path to the parquet file
        otu_97_id: The 97_* ID to search for (e.g., '97_42874')
    
    Returns:
        DataFrame with matching records
    """
    df = pd.read_parquet(parquet_file)
    result = df[df['otu_97_id'] == otu_97_id]
    return result


if __name__ == "__main__":
    # Set up file paths
    input_file = "otus.97.allinfo"
    output_file = "otus_97_queryable.parquet"
    
    # Process the file
    df = process_otus_file(input_file, output_file)
    
    if df is not None:
        # Show some example queries
        print("\n" + "="*60)
        print("EXAMPLE QUERIES")
        print("="*60)
        
        # Get a few sample IDs
        sample_ids = df['otu_97_id'].head(3).tolist()
        
        for sample_id in sample_ids:
            print(f"\nQuerying for {sample_id}:")
            result = query_parquet(output_file, sample_id)
            if not result.empty:
                print(f"  DNA Sequence length: {len(result.iloc[0]['dna_sequence'])} bp")
                print(f"  Bacteria: {result.iloc[0]['bacteria_taxonomy']}")
            else:
                print(f"  Not found")
        
        print("\n" + "="*60)
        print("To query the parquet file in your own code:")
        print("="*60)
        print("""
import pandas as pd

# Load the parquet file
df = pd.read_parquet('otus_97_queryable.parquet')

# Query for a specific 97_* ID
result = df[df['otu_97_id'] == '97_42874']

# Access the data
if not result.empty:
    dna_seq = result.iloc[0]['dna_sequence']
    bacteria = result.iloc[0]['bacteria_taxonomy']
    print(f"DNA: {dna_seq}")
    print(f"Bacteria: {bacteria}")
""")

