"""
Convert samples-otus.97.mapped file to Parquet format with streaming approach.

This script processes a large FASTA-like file containing OTU data and converts it
to a queryable Parquet format, extracting only the 97_* OTU IDs.
"""

import re
from pathlib import Path
from typing import Iterator, Dict, List
import pyarrow as pa
import pyarrow.parquet as pq


def parse_header(line: str) -> Dict[str, str]:
    """
    Parse header line to extract SRR and SRS IDs.
    
    Example: >SRR2459896.SRS1074972  66481   23845   497
    Returns: {'srr_id': 'SRR2459896', 'srs_id': 'SRS1074972'}
    """
    if not line.startswith('>'):
        raise ValueError(f"Invalid header line: {line}")
    
    # Remove '>' and split by whitespace
    parts = line[1:].split()[0]  # Get first part: SRR2459896.SRS1074972
    
    # Split by '.' to get SRR and SRS
    ids = parts.split('.')
    if len(ids) != 2:
        raise ValueError(f"Unexpected header format: {line}")
    
    return {
        'srr_id': ids[0],
        'srs_id': ids[1]
    }


def extract_97_otu_id(otu_string: str) -> str:
    """
    Extract the OTU ID that starts with 97_ from semicolon-separated IDs.
    
    Example: 90_246;96_8626;97_10374 -> 97_10374
    """
    ids = otu_string.split(';')
    for otu_id in ids:
        if otu_id.startswith('97_'):
            return otu_id
    return None  # No 97_ ID found


def parse_data_line(line: str) -> tuple:
    """
    Parse data line to extract 97_* OTU ID and abundance.
    
    Example: 90_246;96_8626;97_10374  4920
    Returns: ('97_10374', 4920)
    """
    parts = line.strip().split()
    if len(parts) != 2:
        return None, None
    
    otu_string, abundance = parts
    otu_97_id = extract_97_otu_id(otu_string)
    
    if otu_97_id is None:
        return None, None
    
    try:
        abundance = int(abundance)
    except ValueError:
        return None, None
    
    return otu_97_id, abundance


def stream_parse_file(file_path: Path, batch_size: int = 100000) -> Iterator[List[Dict]]:
    """
    Stream parse the OTU file and yield batches of records.
    
    Args:
        file_path: Path to the samples-otus.97.mapped file
        batch_size: Number of records to accumulate before yielding
        
    Yields:
        List of dictionaries with keys: srr_id, srs_id, otu_id, abundance
    """
    batch = []
    current_header = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('>'):
                # New sample header
                current_header = parse_header(line)
            else:
                # Data line
                if current_header is None:
                    continue  # Skip data lines without a header
                
                otu_id, abundance = parse_data_line(line)
                
                if otu_id is not None:
                    record = {
                        'srr_id': current_header['srr_id'],
                        'srs_id': current_header['srs_id'],
                        'otu_id': otu_id,
                        'abundance': abundance
                    }
                    batch.append(record)
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
    
    # Yield remaining records
    if batch:
        yield batch


def convert_to_parquet(
    input_file: Path,
    output_file: Path,
    batch_size: int = 100000
):
    """
    Convert the OTU file to Parquet format with streaming.
    
    Args:
        input_file: Path to samples-otus.97.mapped
        output_file: Path to output Parquet file
        batch_size: Number of records per batch
    """
    # Define schema
    schema = pa.schema([
        ('srr_id', pa.string()),
        ('srs_id', pa.string()),
        ('otu_id', pa.string()),
        ('abundance', pa.int64())
    ])
    
    writer = None
    total_records = 0
    
    try:
        for batch_idx, batch in enumerate(stream_parse_file(input_file, batch_size)):
            # Convert batch to PyArrow Table
            table = pa.Table.from_pylist(batch, schema=schema)
            
            if writer is None:
                # Create writer on first batch
                writer = pq.ParquetWriter(output_file, schema)
            
            writer.write_table(table)
            total_records += len(batch)
            
            # Progress update
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches, {total_records:,} records")
        
        print(f"\nConversion complete!")
        print(f"Total records written: {total_records:,}")
        print(f"Output file: {output_file}")
        
    finally:
        if writer is not None:
            writer.close()


def query_example(parquet_file: Path, srs_id: str = None, srr_id: str = None):
    """
    Example queries on the Parquet file.
    
    Args:
        parquet_file: Path to the Parquet file
        srs_id: SRS ID to filter by (optional)
        srr_id: SRR ID to filter by (optional)
    """
    import pandas as pd
    
    # Read Parquet file
    df = pd.read_parquet(parquet_file)
    
    print(f"\nTotal records in file: {len(df):,}")
    print(f"Unique SRR IDs: {df['srr_id'].nunique():,}")
    print(f"Unique SRS IDs: {df['srs_id'].nunique():,}")
    print(f"Unique OTU IDs: {df['otu_id'].nunique():,}")
    
    if srs_id:
        filtered = df[df['srs_id'] == srs_id]
        print(f"\nRecords for SRS_ID={srs_id}: {len(filtered):,}")
        print(filtered.head(10))
    
    if srr_id:
        filtered = df[df['srr_id'] == srr_id]
        print(f"\nRecords for SRR_ID={srr_id}: {len(filtered):,}")
        print(filtered.head(10))


if __name__ == '__main__':
    # File paths
    input_file = Path('data_preprocessing/samples-otus.97.mapped')
    output_file = Path('data_preprocessing/samples-otus-97.parquet')
    
    print("Starting conversion of OTU file to Parquet format...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"File size: {input_file.stat().st_size / (1024**2):.2f} MB")
    print()
    
    # Convert to Parquet
    convert_to_parquet(input_file, output_file, batch_size=100000)
    
    # Show file size comparison
    output_size = output_file.stat().st_size / (1024**2)
    input_size = input_file.stat().st_size / (1024**2)
    compression_ratio = (1 - output_size / input_size) * 100
    
    print(f"\nFile size comparison:")
    print(f"Input:  {input_size:.2f} MB")
    print(f"Output: {output_size:.2f} MB")
    print(f"Compression: {compression_ratio:.1f}%")
    
    # Example queries (comment out if file is too large to load fully)
    # print("\n" + "="*60)
    # print("Example queries:")
    # print("="*60)
    # query_example(output_file, srr_id='SRR2459896')

