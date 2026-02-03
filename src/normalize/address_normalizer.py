"""
Address normalization for ProviderVerify.

Standardizes provider addresses using libpostal and US Census data
to parse components and normalize city/state/ZIP information.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import usaddress
from phonenumbers import PhoneNumberFormat, PhoneNumberType
import phonenumbers

logger = logging.getLogger(__name__)

# Note: libpostal installation may require system dependencies
# This implementation provides fallback using usaddress and regex patterns
try:
    import libpostal
    LIBPOSTAL_AVAILABLE = True
except ImportError:
    LIBPOSTAL_AVAILABLE = False
    logger.warning("libpostal not available, using fallback address normalization")


class AddressNormalizer:
    """
    Normalizes provider addresses for consistent matching.
    
    Parses addresses into components and standardizes city, state, and ZIP codes.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize address normalizer with configuration.
        
        Args:
            config: Configuration dictionary with normalization rules
        """
        self.config = config
        self.standardize_city_state = config.get("standardize_city_state", True)
        self.normalize_street = config.get("normalize_street", True)
        self.zip_regex = config.get("zip_regex", r"\d{5}(-\d{4})?")
        
        # US state abbreviations mapping
        self.state_abbreviations = {
            "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
            "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
            "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
            "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
            "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
            "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
            "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
            "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
            "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
            "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
            "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
            "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
            "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC"
        }
        
        # Street direction abbreviations
        self.street_directions = {
            "north": "N", "south": "S", "east": "E", "west": "W",
            "northeast": "NE", "northwest": "NW", "southeast": "SE", "southwest": "SW"
        }
        
        # Street type abbreviations
        self.street_types = {
            "street": "St", "avenue": "Ave", "boulevard": "Blvd", "drive": "Dr",
            "lane": "Ln", "road": "Rd", "court": "Ct", "place": "Pl",
            "square": "Sq", "terrace": "Ter", "highway": "Hwy", "circle": "Cir"
        }
        
        # Compile regex patterns
        self.zip_pattern = re.compile(self.zip_regex)
        self.phone_pattern = re.compile(r"[^\d]")
        self.punctuation_pattern = re.compile(r"[^\w\s]")
        self.whitespace_pattern = re.compile(r"\s+")
        
        logger.info("Initialized AddressNormalizer")
    
    def parse_address(self, address: str) -> Dict[str, str]:
        """
        Parse address into components using usaddress.
        
        Args:
            address: Raw address string
            
        Returns:
            Dictionary with address components
        """
        if pd.isna(address) or not isinstance(address, str):
            return {
                "street_number": "",
                "street_name": "",
                "street_type": "",
                "city": "",
                "state": "",
                "zipcode": "",
                "address_line_1": "",
                "address_line_2": ""
            }
        
        try:
            # Parse using usaddress
            parsed = usaddress.tag(address)[0]
            
            # Extract components
            components = {
                "street_number": parsed.get("AddressNumber", ""),
                "street_name": parsed.get("StreetName", ""),
                "street_type": parsed.get("StreetNamePostType", ""),
                "city": parsed.get("PlaceName", ""),
                "state": parsed.get("StateName", ""),
                "zipcode": parsed.get("ZipCode", ""),
                "address_line_1": "",
                "address_line_2": ""
            }
            
            # Build address lines
            street_parts = []
            if components["street_number"]:
                street_parts.append(components["street_number"])
            if components["street_name"]:
                street_parts.append(components["street_name"])
            if components["street_type"]:
                street_parts.append(components["street_type"])
            
            components["address_line_1"] = " ".join(street_parts)
            
            return components
            
        except Exception as e:
            logger.warning(f"Failed to parse address '{address}': {e}")
            return {
                "street_number": "",
                "street_name": "",
                "street_type": "",
                "city": "",
                "state": "",
                "zipcode": "",
                "address_line_1": address,
                "address_line_2": ""
            }
    
    def normalize_city(self, city: str) -> str:
        """
        Normalize city name.
        
        Args:
            city: Raw city name
            
        Returns:
            Normalized city name
        """
        if pd.isna(city) or not isinstance(city, str):
            return ""
        
        # Convert to title case and remove punctuation
        city = self.punctuation_pattern.sub(' ', city)
        city = self.whitespace_pattern.sub(' ', city).strip()
        city = city.title()
        
        return city
    
    def normalize_state(self, state: str) -> str:
        """
        Normalize state name/abbreviation.
        
        Args:
            state: Raw state name or abbreviation
            
        Returns:
            Normalized 2-letter state abbreviation
        """
        if pd.isna(state) or not isinstance(state, str):
            return ""
        
        state = state.strip().lower()
        
        # Check if it's already a 2-letter abbreviation
        if len(state) == 2 and state.isalpha():
            return state.upper()
        
        # Map full state name to abbreviation
        if state in self.state_abbreviations:
            return self.state_abbreviations[state]
        
        return ""
    
    def normalize_zipcode(self, zipcode: str) -> str:
        """
        Normalize ZIP code to 5-digit format.
        
        Args:
            zipcode: Raw ZIP code
            
        Returns:
            Normalized 5-digit ZIP code
        """
        if pd.isna(zipcode) or not isinstance(zipcode, str):
            return ""
        
        # Extract 5-digit ZIP using regex
        match = self.zip_pattern.search(zipcode)
        if match:
            return match.group(0)[:5]  # Return first 5 digits
        
        return ""
    
    def normalize_street(self, street: str) -> str:
        """
        Normalize street name.
        
        Args:
            street: Raw street name
            
        Returns:
            Normalized street name
        """
        if pd.isna(street) or not isinstance(street, str):
            return ""
        
        # Convert to lowercase and split
        street = street.lower().strip()
        tokens = street.split()
        
        normalized_tokens = []
        for token in tokens:
            # Normalize directions
            if token in self.street_directions:
                normalized_tokens.append(self.street_directions[token])
            # Normalize street types
            elif token in self.street_types:
                normalized_tokens.append(self.street_types[token])
            else:
                normalized_tokens.append(token.title())
        
        return " ".join(normalized_tokens)
    
    def normalize_address(self, address: str) -> Dict[str, str]:
        """
        Normalize complete address.
        
        Args:
            address: Raw address string
            
        Returns:
            Dictionary with normalized address components
        """
        # Parse address
        components = self.parse_address(address)
        
        # Normalize each component
        normalized = {
            "street_number": components["street_number"],
            "street_name": self.normalize_street(components["street_name"]),
            "street_type": components["street_type"].title(),
            "city_norm": self.normalize_city(components["city"]),
            "state_norm": self.normalize_state(components["state"]),
            "zip_norm": self.normalize_zipcode(components["zipcode"]),
            "address_line_1_norm": self.normalize_street(components["address_line_1"]),
            "address_line_2_norm": components["address_line_2"]
        }
        
        return normalized
    
    def normalize_phone(self, phone: str, default_country: str = "US") -> str:
        """
        Normalize phone number to E164 format.
        
        Args:
            phone: Raw phone number
            default_country: Default country code
            
        Returns:
            Normalized phone number in E164 format
        """
        if pd.isna(phone) or not isinstance(phone, str):
            return ""
        
        try:
            # Parse phone number
            parsed_phone = phonenumbers.parse(phone, default_country)
            
            # Validate phone number
            if not phonenumbers.is_valid_number(parsed_phone):
                return ""
            
            # Format to E164
            return phonenumbers.format_number(parsed_phone, PhoneNumberFormat.E164)
            
        except Exception as e:
            logger.warning(f"Failed to normalize phone '{phone}': {e}")
            return ""
    
    def normalize_email(self, email: str) -> str:
        """
        Normalize email address.
        
        Args:
            email: Raw email address
            
        Returns:
            Normalized email address (lowercase)
        """
        if pd.isna(email) or not isinstance(email, str):
            return ""
        
        email = email.strip().lower()
        
        # Basic email validation
        if "@" in email and "." in email.split("@")[1]:
            return email
        else:
            return ""
    
    def calculate_address_similarity(self, address1: Dict[str, str], address2: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate similarity metrics between two normalized addresses.
        
        Args:
            address1: First normalized address components
            address2: Second normalized address components
            
        Returns:
            Dictionary with similarity scores
        """
        similarities = {}
        
        # Street similarity
        street1 = f"{address1.get('street_number', '')} {address1.get('street_name', '')} {address1.get('street_type', '')}".strip()
        street2 = f"{address2.get('street_number', '')} {address2.get('street_name', '')} {address2.get('street_type', '')}".strip()
        
        similarities["street_match"] = 1.0 if street1.lower() == street2.lower() else 0.0
        
        # City match
        city1 = address1.get("city_norm", "")
        city2 = address2.get("city_norm", "")
        similarities["city_match"] = 1.0 if city1.lower() == city2.lower() else 0.0
        
        # State match
        state1 = address1.get("state_norm", "")
        state2 = address2.get("state_norm", "")
        similarities["state_match"] = 1.0 if state1 == state2 else 0.0
        
        # ZIP match
        zip1 = address1.get("zip_norm", "")
        zip2 = address2.get("zip_norm", "")
        similarities["zip_match"] = 1.0 if zip1 == zip2 else 0.0
        
        # Overall location match (city+state or ZIP)
        location_match = 0.0
        if (city1 and city2 and state1 and state2 and 
            city1.lower() == city2.lower() and state1 == state2):
            location_match = 1.0
        elif zip1 and zip2 and zip1 == zip2:
            location_match = 1.0
        
        similarities["location_match"] = location_match
        
        return similarities
    
    def normalize_dataframe(self, df: pd.DataFrame, 
                          address_column: str = "address",
                          phone_column: str = "phone",
                          email_column: str = "email") -> pd.DataFrame:
        """
        Normalize addresses, phones, and emails in a DataFrame.
        
        Args:
            df: Input DataFrame
            address_column: Column with addresses
            phone_column: Column with phone numbers
            email_column: Column with email addresses
            
        Returns:
            DataFrame with normalized address/contact columns
        """
        result_df = df.copy()
        
        # Normalize addresses
        if address_column in df.columns:
            address_components = df[address_column].apply(self.normalize_address)
            address_df = pd.DataFrame(address_components.tolist())
            
            for col in address_df.columns:
                result_df[col] = address_df[col]
        
        # Normalize phone numbers
        if phone_column in df.columns:
            result_df[f"{phone_column}_norm"] = df[phone_column].apply(self.normalize_phone)
        
        # Normalize email addresses
        if email_column in df.columns:
            result_df[f"{email_column}_norm"] = df[email_column].apply(self.normalize_email)
        
        logger.info(f"Normalized addresses and contact info for {len(result_df)} records")
        return result_df


def normalize_provider_addresses(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Convenience function to normalize provider addresses in a DataFrame.
    
    Args:
        df: Provider DataFrame
        config: Normalization configuration
        
    Returns:
        DataFrame with normalized addresses
    """
    normalizer = AddressNormalizer(config)
    return normalizer.normalize_dataframe(df)
