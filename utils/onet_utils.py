"""
O*NET Code Utilities

This module provides utilities for working with O*NET occupation codes, including
format conversion between numeric and standard formats, and discipline assignment.

O*NET codes can be represented in two formats:
- Numeric format: '17214100' (8 digits)
- Standard format: '17-2141.00' (XX-XXXX.XX)
"""

from typing import Optional, Dict, List
import json
from pathlib import Path


def convert_onet_numeric_to_standard(code: Optional[str]) -> Optional[str]:
    """
    Convert numeric O*NET code to standard format.

    Transforms an 8-digit numeric code (e.g., '17214100') into the standard
    O*NET format (e.g., '17-2141.00') with dashes and decimal points.

    Parameters
    ----------
    code : str or None
        Numeric O*NET code as an 8-character string

    Returns
    -------
    str or None
        Standard format O*NET code (XX-XXXX.XX) or None if invalid input

    Examples
    --------
    >>> convert_onet_numeric_to_standard('17214100')
    '17-2141.00'

    >>> convert_onet_numeric_to_standard(None)
    None

    >>> convert_onet_numeric_to_standard('invalid')
    None

    Notes
    -----
    The function validates that the input is exactly 8 characters long.
    Invalid inputs return None rather than raising exceptions.
    """
    if code is None or len(str(code)) != 8:
        return None

    code_str = str(code)
    # Format: XX-XXXX.XX
    return f"{code_str[:2]}-{code_str[2:6]}.{code_str[6:]}"


def convert_onet_standard_to_numeric(code: Optional[str]) -> Optional[str]:
    """
    Convert standard O*NET code to numeric format.

    Transforms a standard O*NET code (e.g., '17-2141.00') into the numeric
    8-digit format (e.g., '17214100') by removing dashes and decimal points.

    Parameters
    ----------
    code : str or None
        Standard format O*NET code (XX-XXXX.XX)

    Returns
    -------
    str or None
        Numeric O*NET code as 8-character string or None if invalid input

    Examples
    --------
    >>> convert_onet_standard_to_numeric('17-2141.00')
    '17214100'

    >>> convert_onet_standard_to_numeric(None)
    None

    Notes
    -----
    This function performs the inverse operation of convert_onet_numeric_to_standard.
    The two functions can be chained for round-trip conversion.
    """
    if code is None:
        return None
    return str(code).replace('-', '').replace('.', '')


def load_discipline_codes(config_path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load O*NET codes for engineering disciplines from configuration.

    Reads the mechanical_terms.json configuration file and extracts the O*NET
    codes for civil, electrical, and mechanical engineering disciplines. Returns
    both standard and numeric formats for easy lookup.

    Parameters
    ----------
    config_path : Path or None, optional
        Path to the mechanical_terms.json configuration file.
        If None, uses the default path relative to this module.

    Returns
    -------
    dict
        Dictionary with keys 'civil', 'electrical', 'mechanical', each containing
        a dict with 'standard' and 'numeric' lists of codes.

    Examples
    --------
    >>> codes = load_discipline_codes()
    >>> codes['mechanical']['standard']
    ['17-2141.00']
    >>> codes['mechanical']['numeric']
    ['17214100']

    Notes
    -----
    The returned dictionary has the structure:
    {
        'civil': {
            'standard': ['17-2051.00', '17-2051.01'],
            'numeric': ['17205100', '17205101']
        },
        'electrical': {...},
        'mechanical': {...}
    }
    """
    if config_path is None:
        # Default path: config/mechanical_terms.json relative to this file
        config_path = Path(__file__).parent.parent / 'config' / 'mechanical_terms.json'

    with open(config_path, 'r') as f:
        config = json.load(f)

    onet_codes = config['onet_codes']

    # Convert to both formats for easy lookup
    discipline_codes = {}
    for discipline, standard_codes in onet_codes.items():
        discipline_codes[discipline] = {
            'standard': standard_codes,
            'numeric': [convert_onet_standard_to_numeric(code) for code in standard_codes]
        }

    return discipline_codes


def assign_discipline(
    onet_code: str,
    discipline_map: Optional[Dict[str, List[str]]] = None,
    format: str = 'numeric'
) -> str:
    """
    Assign engineering discipline based on O*NET code.

    Maps an O*NET code to its corresponding engineering discipline (Civil,
    Electrical, or Mechanical). Codes not matching any discipline return 'Other'.

    Parameters
    ----------
    onet_code : str
        O*NET code in either numeric or standard format
    discipline_map : dict or None, optional
        Pre-loaded discipline code mapping from load_discipline_codes().
        If None, loads from default configuration.
    format : {'numeric', 'standard'}, default 'numeric'
        Format of the input onet_code

    Returns
    -------
    str
        Discipline name: 'Civil', 'Electrical', 'Mechanical', or 'Other'

    Examples
    --------
    >>> assign_discipline('17214100')
    'Mechanical'

    >>> assign_discipline('17-2141.00', format='standard')
    'Mechanical'

    >>> assign_discipline('99999999')
    'Other'

    Notes
    -----
    The discipline mapping is based on the onet_codes configuration in
    mechanical_terms.json. This function is optimized for batch processing
    by allowing a pre-loaded discipline_map to be passed in.
    """
    if discipline_map is None:
        discipline_map = load_discipline_codes()

    # Normalize code to string
    code_str = str(onet_code)

    # Check each discipline
    for discipline, codes in discipline_map.items():
        if code_str in codes[format]:
            return discipline.capitalize()

    return 'Other'


def create_discipline_lookup_expression(
    discipline_map: Optional[Dict[str, List[str]]] = None,
    format: str = 'numeric'
) -> Dict[str, str]:
    """
    Create a Polars-friendly lookup dictionary for discipline assignment.

    Generates a flat dictionary mapping O*NET codes to discipline names,
    suitable for use with Polars .replace() or similar operations.

    Parameters
    ----------
    discipline_map : dict or None, optional
        Pre-loaded discipline code mapping from load_discipline_codes().
        If None, loads from default configuration.
    format : {'numeric', 'standard'}, default 'numeric'
        Format of O*NET codes to include in lookup

    Returns
    -------
    dict
        Flat dictionary: {onet_code: discipline_name}

    Examples
    --------
    >>> lookup = create_discipline_lookup_expression()
    >>> lookup['17214100']
    'Mechanical'

    >>> import polars as pl
    >>> df = df.with_columns(
    ...     pl.col('ConsolidatedONET').replace(lookup).alias('Discipline')
    ... )

    Notes
    -----
    This function is optimized for vectorized operations in Polars DataFrames.
    It's more efficient than applying assign_discipline() row-by-row.
    """
    if discipline_map is None:
        discipline_map = load_discipline_codes()

    lookup = {}
    for discipline, codes in discipline_map.items():
        for code in codes[format]:
            lookup[code] = discipline.capitalize()

    return lookup


# Example usage and testing
if __name__ == "__main__":
    # Test conversions
    print("Testing O*NET code conversions:")
    print("-" * 50)

    # Mechanical Engineering code
    numeric_code = '17214100'
    standard_code = '17-2141.00'

    print(f"Numeric to Standard: {numeric_code} -> {convert_onet_numeric_to_standard(numeric_code)}")
    print(f"Standard to Numeric: {standard_code} -> {convert_onet_standard_to_numeric(standard_code)}")

    # Round-trip test
    round_trip = convert_onet_standard_to_numeric(
        convert_onet_numeric_to_standard(numeric_code)
    )
    print(f"Round-trip test: {numeric_code} -> {round_trip} (Match: {numeric_code == round_trip})")

    print("\nTesting discipline assignment:")
    print("-" * 50)

    # Load codes
    codes = load_discipline_codes()
    print(f"Loaded codes for {len(codes)} disciplines")

    # Test assignments
    test_codes = [
        ('17214100', 'Mechanical'),
        ('17205100', 'Civil'),
        ('17207100', 'Electrical'),
        ('99999999', 'Other')
    ]

    for code, expected in test_codes:
        result = assign_discipline(code, codes)
        status = "✓" if result == expected else "✗"
        print(f"{status} {code} -> {result} (expected: {expected})")

    print("\nTesting lookup dictionary:")
    print("-" * 50)
    lookup = create_discipline_lookup_expression(codes)
    print(f"Created lookup with {len(lookup)} entries")
    for code, discipline in list(lookup.items())[:3]:
        print(f"  {code}: {discipline}")
