import os
import zipfile
import pandas as pd
from pathlib import Path
from lxml import etree as ET
from typing import Iterator, Optional, List, Dict, Any
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import duckdb
import polars as pl
from datetime import datetime, timedelta

class DuckDBManager:
    """Manager class for handling DuckDB operations with all data"""
    def __init__(
        self,
        base_path: str,
        read_only: bool = False,
        data_source: str = "burning_glass",
        data_mode: str = 'pd',
    ):
        self.base_path = Path(base_path)
        
        if data_source not in ["burning_glass", "nlx"]:
            raise ValueError("data_source must be either 'burning_glass' or 'nlx'")
        self.data_source = data_source

        if data_mode not in ['pd', 'pl']:
            raise ValueError("data_mode must be either 'pd' (pandas) or 'pl' (polars)")
        self.data_mode = data_mode

        # Determine processed path logic
        if self.data_source == "burning_glass":
            self.processed_path = self.base_path / "processed"
            self.processed_path.mkdir(exist_ok=True)
        else:
            self.processed_path = self.base_path

        # Find DuckDB file in base_path or use provided filename
        duckdb_files = list(self.base_path.glob("*.duckdb"))
        if duckdb_files:
            db_path = duckdb_files[0]
        else:
            db_path = self.base_path / "all_jobs.duckdb"

        self.conn = duckdb.connect(str(db_path), read_only=read_only)
        self._setup_duckdb()
    
    def _setup_duckdb(self):
        """Setup DuckDB with settings for large datasets"""
        self.conn.execute("""
            PRAGMA threads=4;
            PRAGMA enable_progress_bar=1;
            PRAGMA memory_limit='4GB';
        """)
    
    def _xml_to_dict(self, elem) -> dict:
        """Convert XML element to dictionary with type inference"""
        result = {}
        for child in elem:
            if child.text:
                text = child.text.strip()
                # Basic type inference
                if child.tag in ['JobID'] and text.isdigit():
                    result[child.tag] = int(text)
                elif child.tag in ['JobDate']:
                    try:
                        result[child.tag] = pd.to_datetime(text).date()
                    except:
                        result[child.tag] = text
                else:
                    result[child.tag] = text
        return result

    def stream_xml_from_zip(self, zip_path: Path) -> Iterator[dict]:
        """Stream XML records from ZIP file with better error handling"""
        xml_filename = zip_path.stem + '.xml'
        xml_path = zip_path.parent / xml_filename
        
        # Prefer XML file if it exists (faster than ZIP extraction)
        if xml_path.exists():
            with open(xml_path, 'rb') as xml_file:
                yield from self._parse_xml_stream(xml_file)
        else:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                try:
                    with zf.open(xml_filename) as xml_file:
                        yield from self._parse_xml_stream(xml_file)
                except KeyError:
                    print(f"Warning: {xml_filename} not found in {zip_path.name}")
                    return

    def _parse_xml_stream(self, xml_file) -> Iterator[dict]:
        """Parse XML stream with memory-efficient approach"""
        context = ET.iterparse(xml_file, events=('start', 'end'), recover=True)
        context = iter(context)
        try:
            event, root = next(context)
            for event, elem in context:
                if event == 'end' and elem.tag == 'Job':
                    yield self._xml_to_dict(elem)
                    elem.clear()
                    # Clear root periodically to free memory
                    if len(root) > 1000:
                        root.clear()
        except StopIteration: # Empty or malformed XML, just return
            print(f"Warning: {xml_file.name} is empty or malformed")
            return
        except ET.XMLSyntaxError as e:
            print(f"XML parsing error: {e}")
            return

    def process_in_chunks(self, zip_path: Path, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
        """Process data in manageable chunks"""
        chunk = []
        for record in self.stream_xml_from_zip(zip_path):
            chunk.append(record)
            if len(chunk) >= chunk_size:
                yield pd.DataFrame(chunk)
                chunk = []

        if chunk:  # Don't forget the last chunk
            yield pd.DataFrame(chunk)

    def convert_to_optimized_parquet(self, zip_path: Path, 
                                   partition_cols: List[str] = None,
                                   compression: str = 'zstd') -> Path:
        """Convert to Parquet with better compression and partitioning"""
        parquet_path = self.processed_path / f"{zip_path.stem}.parquet"
        
        if parquet_path.exists():
            print(f"✓ {parquet_path.name} already exists, skipping conversion")
            return parquet_path

        print(f"Converting {zip_path.name} to optimized Parquet...")
        
        # Collect data in chunks
        chunks = []
        for chunk_df in self.process_in_chunks(zip_path, chunk_size=5000):
            if len(chunk_df) > 0:
                # Add derived columns for better querying
                if 'JobDate' in chunk_df.columns:
                    chunk_df['year'] = pd.to_datetime(chunk_df['JobDate']).dt.year
                    chunk_df['month'] = pd.to_datetime(chunk_df['JobDate']).dt.month
                chunks.append(chunk_df)

        if not chunks:
            return parquet_path

        full_df = pd.concat(chunks, ignore_index=True, sort=False)
        
        # Write with better compression
        full_df.to_parquet(
            parquet_path, 
            engine='pyarrow', 
            index=False,
            compression=compression,
            # Use dictionary encoding for categorical columns
            use_dictionary=['ConsolidatedONET', 'CleanJobTitle'] if 'ConsolidatedONET' in full_df.columns else None
        )
        
        print(f"✓ Converted")

        return parquet_path

    def query_date_range(self, start_date: str, end_date: str,
                         columns: Optional[list] = None) -> pd.DataFrame:
        """Query data across multiple files for a date range"""
        # Always use DuckDB
        select_cols = ", ".join(columns) if columns else "*"
        query = f"""
            SELECT {select_cols}
            FROM all_jobs 
            WHERE JobDate BETWEEN '{start_date}' AND '{end_date}'
        """
        return self.query_with_duckdb(query)

    def query_onet_codes(self, onet_codes: list, columns: Optional[list] = None) -> pd.DataFrame:
        """Query data for specific O*NET codes by scanning processed Parquet files"""
        # Always use DuckDB
        return self.smart_query_onet_codes(onet_codes, columns=columns)
    
    def query_onet_codes_balanced(self, onet_codes: list, columns: Optional[list] = None, default_size: int = 300) -> pd.DataFrame:
        """Advanced filtering scheme that returns a dataframe with balanced distribution per ONET code"""
        # Get all matching jobs first
        all_jobs = self.query_onet_codes(onet_codes, columns)
        
        if all_jobs.empty:
            return pd.DataFrame()

        # Ensure we have at least `default_size` jobs per O*NET code
        counts = all_jobs['ConsolidatedONET'].value_counts()
        valid_onet_codes = counts[counts >= default_size].index.tolist()

        if not valid_onet_codes:
            return pd.DataFrame()
            
        # Filter to only include valid O*NET codes
        filtered_df = all_jobs[all_jobs['ConsolidatedONET'].isin(valid_onet_codes)]
        
        # Ensure distribution is consistent across O*NET codes
        consistent_df = filtered_df.groupby('ConsolidatedONET').apply(
            lambda x: x.sample(default_size, random_state=42)
        ).reset_index(drop=True)
        
        return consistent_df if not consistent_df.empty else pd.DataFrame()
    
    def check_parquet_exists(self, parquet_path: Path) -> bool:
        """Check if a Parquet file exists in the processed directory"""
        return (self.base_path / parquet_path).exists()

    def create_duckdb_views(self):
        """Create optimized views in DuckDB for common queries"""
        # Always use DuckDB
        # Create a view that unions all parquet files
        parquet_files = list(self.processed_path.glob("*.parquet"))
        if not parquet_files:
            return
            
        # Get schema information from all files to find common columns
        all_columns = set()
        file_schemas = {}
        
        print(f"Found {len(parquet_files)} parquet files")
        for f in parquet_files:
            try:
                # Use DuckDB to describe the parquet file structure
                schema_query = f"DESCRIBE SELECT * FROM '{f}' LIMIT 0"
                schema_result = self.conn.execute(schema_query).fetchall()
                columns = [row[0] for row in schema_result]
                file_schemas[f] = columns  # Keep as list to preserve order
                all_columns.update(columns)
                print(f"File {f.name}: {len(columns)} columns - {columns[:5]}...")
            except Exception as e:
                print(f"Warning: Could not read schema for {f.name}: {e}")
                continue
        
        if not all_columns:
            print("Warning: No valid parquet files found")
            return
            
        # Find common columns across all files (intersection)
        common_columns = set(all_columns)
        for file_cols in file_schemas.values():
            common_columns.intersection_update(file_cols)
        
        # Remove source_file from common columns if it exists (we'll add it back)
        common_columns.discard('source_file')
        common_columns = sorted(list(common_columns))
        
        print(f"Common columns across all files: {len(common_columns)} - {common_columns[:10]}...")
        
        if not common_columns:
            print("Warning: No common columns found across parquet files")
            # Show what columns each file has for debugging
            for f, cols in file_schemas.items():
                print(f"  {f.name}: {cols}")
            return
        
        # Build union query with only common columns plus source_file
        common_cols_str = ", ".join(common_columns)
        union_queries = []
        
        for f in parquet_files:
            if f in file_schemas:  # Only include files we could read
                query = f"SELECT {common_cols_str}, '{f.stem}' as source_file FROM '{f}'"
                union_queries.append(query)
                print(f"Query for {f.name}: {len(common_columns) + 1} columns")
        
        if not union_queries:
            print("Warning: No valid files to union")
            return
            
        union_query = " UNION ALL ".join(union_queries)
        print(f"Final union query has {len(union_queries)} parts")
        
        try:
            self.conn.execute(f"""
                CREATE OR REPLACE VIEW all_jobs AS
                {union_query}
            """)
            print("Successfully created all_jobs view")
        except Exception as e:
            print(f"Failed to create view: {e}")
            # Print first query for debugging
            if union_queries:
                print(f"First query: {union_queries[0]}")
            raise
        
        if self.data_source == "burning_glass":
            # Create materialized aggregation tables for faster queries
            self.conn.execute("""
                CREATE OR REPLACE TABLE job_counts_by_onet AS
                SELECT 
                    ConsolidatedONET,
                    COUNT(*) as job_count,
                    MIN(JobDate) as earliest_date,
                    MAX(JobDate) as latest_date,
                    COUNT(DISTINCT YEAR(CAST(JobDate AS DATE))) as years_active
                FROM all_jobs 
                WHERE ConsolidatedONET IS NOT NULL
                GROUP BY ConsolidatedONET
            """)
            print("Created job_counts_by_onet table for fast ONET code queries")
        else:
            # Create a view for NLX data
            self.conn.execute("""
                CREATE OR REPLACE VIEW nlx_jobsby_onet_state AS 
                SELECT
                    classifications_onet_code,
                    COUNT(*) as job_count,
                    MIN(date_compiled) as earliest_date,
                    MAX(date_compiled) as latest_date,
                FROM all_jobs 
                WHERE classifications_onet_code IS NOT NULL
                GROUP BY classifications_onet_code
            """)
            print("Created nlx_jobsby_onet_state view for fast NLX ONET code queries")

    def query_with_duckdb(self, query: str):
        """Execute SQL query using DuckDB, returning pandas or polars DataFrame based on self.data_mode"""
        try:
            if self.data_mode == 'pl':
                return self.conn.execute(query).pl()
            else:
                return self.conn.execute(query).df()
        except Exception as e:
            print(f"SQL query failed: {e}")
            if self.data_mode == 'pl':
                return pl.DataFrame()
            else:
                return pd.DataFrame()

    def smart_query_onet_codes(self, onet_codes: List[str], 
                             date_range: Optional[tuple] = None,
                             sample_size: Optional[int] = None,
                             columns: Optional[List[str]] = None):
        """Smart querying that chooses best method based on data size and requirements"""
        # Always use DuckDB
        # Build SQL query
        select_cols = ", ".join(columns) if columns else "*"
        # Initialize where conditions list - nothing for now but might populate from pg. 49/34
        where_conditions = []
        # Add ONET codes condition with proper SQL quoting
        if onet_codes:
            quoted_codes = [f"'{code}'" for code in onet_codes]
            onet_condition = f"ConsolidatedONET IN ({','.join(quoted_codes)})"
            where_conditions.append(onet_condition)
        # Add date range condition if specified
        if date_range:
            start_date, end_date = date_range
            where_conditions.append(f"JobDate BETWEEN '{start_date}' AND '{end_date}'")
        # Build the WHERE clause
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        # Base query
        query = f"""
            SELECT {select_cols}
            FROM all_jobs 
            WHERE {where_clause}
        """
        # Add sampling if specified (DuckDB syntax)
        if sample_size:
            query += f" ORDER BY RANDOM() LIMIT {sample_size}"
        return self.query_with_duckdb(query)

    def query_to_parquet(self, query: str, filename: str, analysis_folder: str = "Analysis") -> str:
        """
        Execute SQL query and save results directly to parquet file in Analysis folder.
        
        Args:
            query: SQL query to execute
            filename: Name of the parquet file (without extension)
            analysis_folder: Name of the analysis folder (default: "Analysis")
            
        Returns:
            str: Full path to the saved parquet file
        """
        # Create Analysis folder if it doesn't exist
        analysis_path = self.base_path / analysis_folder
        analysis_path.mkdir(exist_ok=True)
        
        # Ensure filename has .parquet extension
        if not filename.endswith('.parquet'):
            filename += '.parquet'

        # Check if file already exists
        if (analysis_path / filename).exists():
            print(f"File already exists: {analysis_path / filename}")
            return str(analysis_path / filename)

        output_path = analysis_path / filename
        
        # Use DuckDB's COPY command to write directly to parquet
        copy_query = f"""
            COPY (
                {query}
            ) TO '{output_path}' (FORMAT 'parquet');
        """
        
        try:
            self.conn.execute(copy_query)
            print(f"Query results saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error saving query to parquet: {e}")
            raise

    def get_total_parquet_size(self) -> float:
        """Calculate total size of all Parquet files in the processed directory"""
        total_size = 0.0
        for parquet_file in self.processed_path.glob("*.parquet"):
            total_size += parquet_file.stat().st_size / (1024**2)
        return total_size

    def catalog_data(self) -> pd.DataFrame:
        """Create a catalog of all available data files - enhanced version"""
        include_folder_suffixes = ["", "6", "7"]  # Allows '20166', '20077', '2017', etc.
        exclude_folders = ["Analysis", "processed", "sample_output"]

        catalog = []

        for year_dir in self.base_path.glob("*/"):
            folder_name = year_dir.name
            try:
                if folder_name in exclude_folders:
                    continue
                if not year_dir.is_dir() or not any(folder_name.endswith(suffix) for suffix in include_folder_suffixes):
                    continue
            except Exception as e:
                print(f"Skipping folder {folder_name}: {e}")
                continue
                
            for zip_file in year_dir.glob("*.zip"):
                parts = zip_file.stem.split('_')
                try:
                    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
                        start_date = pd.to_datetime(parts[-2])
                        end_date = pd.to_datetime(parts[-1])
                    elif len(parts) >= 2 and parts[-1].isdigit():
                        start_date = pd.to_datetime(parts[-1])
                        end_date = start_date
                    else:
                        raise ValueError("Filename does not contain parseable date(s)")
                    year = start_date.year
                except Exception as e:
                    print(f"Skipping file {zip_file.name}: cannot parse dates ({e})")
                    continue

                # Enhanced catalog with additional metadata
                catalog.append({
                    'folder_name': folder_name,
                    'year': year,
                    'file_path': zip_file,
                    'start_date': start_date,
                    'end_date': end_date,
                    'size_mb': zip_file.stat().st_size / (1024**2),
                    'processed': (self.processed_path / f"{zip_file.stem}.parquet").exists(),
                    'week_of_year': start_date.isocalendar()[1],  # Week number
                    'quarter': (start_date.month - 1) // 3 + 1,   # Quarter
                    'has_xml_counterpart': (zip_file.parent / f"{zip_file.stem}.xml").exists(),
                    'compression_ratio': None  # Will be filled if XML exists
                })
                
                # Calculate compression ratio if XML exists
                xml_file = zip_file.parent / f"{zip_file.stem}.xml"
                if xml_file.exists():
                    catalog[-1]['compression_ratio'] = xml_file.stat().st_size / zip_file.stat().st_size

        df = pd.DataFrame(catalog).sort_values(['year', 'start_date'])
        
        # Add some useful derived columns
        if not df.empty:
            df['days_covered'] = (df['end_date'] - df['start_date']).dt.days + 1
            df['mb_per_day'] = df['size_mb'] / df['days_covered']
            
        return df
    
    def get_data_summary(self) -> Dict[str, Any]:
        if self.data_source == "nlx":
            return self._get_nlx_data_summary()
        else:
            return self._get_burning_glass_data_summary()
        
    def _get_nlx_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the NLX dataset"""
        summary = {}

        # Overall stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total_jobs,
                COUNT(DISTINCT classifications_onet_code) as unique_onet_codes,
                MIN(date_compiled) as earliest_date,
                MAX(date_compiled) as latest_date,
                COUNT(DISTINCT source_file) as total_files
            FROM all_jobs
        """).fetchone()

        summary['overview'] = {
            'total_jobs': result[0],
            'unique_onet_codes': result[1],
            'date_range': (result[2], result[3]),
            'total_files': result[4]
        }

        # Top ONET codes
        top_onet = self.conn.execute("""
            SELECT classifications_onet_code, job_count
            FROM nlx_jobsby_onet_state
            ORDER BY job_count DESC
            LIMIT 10
        """).df()
        summary['top_onet_codes'] = top_onet

        # Jobs per year
        yearly_counts = self.conn.execute("""
            SELECT 
                YEAR(CAST(date_compiled AS DATE)) as year,
                COUNT(*) as job_count
            FROM all_jobs
            GROUP BY YEAR(CAST(date_compiled AS DATE))
            ORDER BY year
        """).df()
        summary['yearly_distribution'] = yearly_counts

        # Total range of dates for which there is data
        date_range = self.conn.execute("""
            SELECT 
                MIN(date_compiled) as earliest_date,
                MAX(date_compiled) as latest_date
            FROM all_jobs
        """).fetchone()
        summary['date_range'] = {
            'earliest_date': date_range[0],
            'latest_date': date_range[1]
        }

        return summary

    def _get_burning_glass_data_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the dataset"""
        # Always use DuckDB
        summary = {}
        
        # Overall stats
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total_jobs,
                COUNT(DISTINCT ConsolidatedONET) as unique_onet_codes,
                MIN(JobDate) as earliest_date,
                MAX(JobDate) as latest_date,
                COUNT(DISTINCT source_file) as total_files
            FROM all_jobs
        """).fetchone()
        
        summary['overview'] = {
            'total_jobs': result[0],
            'unique_onet_codes': result[1], 
            'date_range': (result[2], result[3]),
            'total_files': result[4]
        }
        
        # Top ONET codes
        top_onet = self.conn.execute("""
            SELECT ConsolidatedONET, job_count
            FROM job_counts_by_onet 
            ORDER BY job_count DESC 
            LIMIT 10
        """).df()
        summary['top_onet_codes'] = top_onet
        
        # Jobs per year
        yearly_counts = self.conn.execute("""
            SELECT 
                YEAR(CAST(JobDate AS DATE)) as year,
                COUNT(*) as job_count
            FROM all_jobs
            GROUP BY YEAR(CAST(JobDate AS DATE))
            ORDER BY year
        """).df()
        summary['yearly_distribution'] = yearly_counts
        
        return summary

    def cleanup_xml_files(self, dry_run: bool = True):
        """Remove XML files if ZIP versions exist (space saving)"""
        xml_files = []
        for year_dir in self.base_path.iterdir():
            try:
                xml_files.extend(year_dir.rglob("*.xml"))
            except OSError as e:
                print(f"Skipping corrupted directory {year_dir}: {e}")
                continue

        savings = 0
        files_to_remove = []
        
        for xml_file in xml_files:
            zip_file = xml_file.with_suffix('.zip')
            if zip_file.exists():
                size_diff = xml_file.stat().st_size - zip_file.stat().st_size
                savings += size_diff
                files_to_remove.append((xml_file, size_diff))
        
        print(f"Found {len(files_to_remove)} XML files with ZIP counterparts")
        print(f"Potential space savings: {savings / (1024**3):.2f} GB")
        
        if not dry_run:
            for xml_file, _ in files_to_remove:
                xml_file.unlink()
            print("XML files removed!")
        else:
            print("Dry run - no files actually removed")