from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd
from typing import Optional, List
import argparse
from tqdm import tqdm

SRS_TO_OTU_FILE_PATH = Path("mapref_data/samples-otus-97.parquet")
OTU_TO_DNA_FILE_PATH = Path("mapref_data/otus_97_to_dna.parquet")
OUT_DIR = Path("dna_sequences")

class Query:
    def __init__(self, srs_id: str, srs_to_otu_file: Path, otu_to_dna_file: Path):
        self.srs_id = srs_id
        self.srs_to_otu_file = srs_to_otu_file
        self.otu_to_dna_file = otu_to_dna_file
        
    def get_otus_from_srs(self, srs_id: str) -> List[str]:
        """
        Get OTUs from the srs_to_otu_file for a given SRS ID.
        Args:
            srs_id: SRS ID to get OTUs from the srs_to_otu_file
            
        Returns:
            List of OTU IDs
        """
        
        print(f"Getting OTUs for SRS {srs_id}")
        filters = [('srs_id', '=', srs_id)]
        table = pq.read_table(self.srs_to_otu_file, filters=filters)
        return table.to_pandas()['otu_id'].tolist()
    
    def get_dna_from_otu(self, otu_id: str) -> str:
        """
        Get DNA from the otu_to_dna_file for a given OTU ID.
        Args:
            otu_id: OTU ID to get DNA from the otu_to_dna_file
            
        Returns:
            DNA sequence
        """
        filters = [('otu_97_id', '=', otu_id)]
        table = pq.read_table(self.otu_to_dna_file, filters=filters).to_pandas()
        dna_seq = table.iloc[0]['dna_sequence']
        return dna_seq
    
    def get_dna_from_srs(self, srs_id):
        """
        Get DNA from the srs_to_otu_file and otu_to_dna_file for a given SRS ID.
        Args:
            srs_id: SRS ID to get DNA from the srs_to_otu_file and otu_to_dna_file
            
        Returns:
            Dictionary of OTU IDs and DNA sequences
        """
        print(f"Getting DNA for SRS {srs_id}")
        otus = self.get_otus_from_srs(srs_id)
        srs_dna_map = {}
        print(f"Getting DNA for {len(otus)} OTUs")
        for otu in tqdm(otus):
            dna_seq = self.get_dna_from_otu(otu)
            srs_dna_map[otu] = dna_seq
        return srs_dna_map
    
    def store_csv(self, output_dir: Path, srs_dna_map: dict):
        """
        Store the DNA sequences for a given SRS ID in a CSV file.
        Args:
            output_dir: Directory to store the CSV file
            srs_dna_map: Dictionary of OTU IDs and DNA sequences
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Storing CSV for SRS {self.srs_id}")
        df = pd.DataFrame(srs_dna_map.items(), columns=['otu_id', 'dna_sequence'])
        df.to_csv(output_dir / f"{self.srs_id}.csv", index=False)
        print("Processing complete!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--srs_id", type=str, required=True)
 
    args = parser.parse_args()
    query = Query(args.srs_id, SRS_TO_OTU_FILE_PATH, OTU_TO_DNA_FILE_PATH)
    srs_dna_map = query.get_dna_from_srs(args.srs_id)
    query.store_csv(OUT_DIR, srs_dna_map)
        
        