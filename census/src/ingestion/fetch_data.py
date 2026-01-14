def __init__(self, api_key: str = "70e45fe92b3388139fe141a2db2d6d6bab65d94d"):
    self.api_key = api_key
    self.base_url = 'https://api.census.gov/data/2022/acs/acs5/profile'
    try:
        self.variable_descriptions = self._get_variable_descriptions()
    except Exception as e:
        print(f"Warning: Could not fetch variable descriptions: {e}")
        self.variable_descriptions = self._get_fallback_descriptions()
    
def _get_variable_descriptions(self) -> Dict[str, str]:
    """Fetch variable descriptions from Census API"""
    try:
        url = f"{self.base_url}/variables.json"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            variables_json = response.json()
            variables = variables_json['variables']
            variable_dict = {var: variables[var]['label'] for var in variables}
            return variable_dict
        else:
            print(f"Warning: API returned status code {response.status_code}")
            return self._get_fallback_descriptions()
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Warning: Error fetching variable descriptions: {e}")
        return self._get_fallback_descriptions()


def _get_fallback_descriptions(self) -> Dict[str, str]:
    """Fallback descriptions for common variables"""
    return {
        "NAME": "Geographic Name",
        "DP05_0001E": "Total Population",
        "DP05_0002E": "Male Population", 
        "DP05_0003E": "Female Population",
        "DP03_0052PE": "Households $10k-$15k (%)",
        "DP03_0053PE": "Households <$10k (%)",
        "DP03_0054PE": "Households $15k-$25k (%)",
        "DP03_0055PE": "Households $25k-$35k (%)",
        "DP03_0056PE": "Households $35k-$50k (%)",
        "DP03_0057PE": "Households $50k-$75k (%)",
        "DP03_0058PE": "Households $75k-$100k (%)",
        "DP03_0059PE": "Households $100k-$150k (%)",
        "DP03_0097PE": "Private Health Insurance (%)",
        "DP03_0129E": "People Below Poverty Level"
    }

def _get_var_description(self, var: str) -> str:
    """Safely get variable description"""
    if hasattr(self, 'variable_descriptions') and var in self.variable_descriptions:
        return self.variable_descriptions[var]
    return var

def _shorten_variable_name(self, var_name: str, max_length: int = 50) -> str:
    """
    Automatically shorten long variable names for better visualization display.
    
    Args:
        var_name: Original variable name/description
        max_length: Maximum allowed length for the shortened name
        
    Returns:
        Shortened, more readable variable name
    """
    # If already short enough, return as-is
    if len(var_name) <= max_length:
        return var_name
        
    # Common patterns to remove or replace
    replacements = {
        # Remove redundant phrases
        r'\bEstimate!!\b': '',
        r'\bTotal!!\b': '',
        r'\bHouseholds!!\b': 'HH',
        r'\bPopulation\b': 'Pop',
        r'\bCharacteristics\b': 'Chars',
        r'\bDemographic\b': 'Demo',
        r'\bEconomic\b': 'Econ',
        r'\bEducational\b': 'Edu',
        r'\bAttainment\b': 'Attain',
        r'\bEmployment\b': 'Employ',
        r'\bOccupation\b': 'Occup',
        r'\bIndustry\b': 'Ind',
        r'\bTransportation\b': 'Transport',
        r'\bCommuting\b': 'Commute',
        r'\bManagement\b': 'Mgmt',
        r'\bProfessional\b': 'Prof',
        r'\bService\b': 'Svc',
        r'\bProduction\b': 'Prod',
        r'\bConstruction\b': 'Constr',
        r'\bMaintenance\b': 'Maint',
        # Income brackets shortening
        r'\$(\d+),(\d+)': r'$\1k',  # $10,000 -> $10k
        r'(\d+) years': r'\1y',      # 25 years -> 25y
        r' and over': '+',           # and over -> +
        r' to ': '-',                # to -> -
        # Remove extra spaces and punctuation
        r'!!+': ' ',                 # Multiple !! -> single space
        r'\s+': ' ',                 # Multiple spaces -> single space
        r'^\s+|\s+$': '',           # Trim spaces
    }
    
    shortened = var_name
    for pattern, replacement in replacements.items():
        shortened = re.sub(pattern, replacement, shortened, flags=re.IGNORECASE)
    
    # If still too long, use more aggressive shortening
    if len(shortened) > max_length:
        # Try to keep the most important parts
        parts = shortened.split(' ')
        if len(parts) > 1:
            # Keep first few words and last word if it contains key info
            important_parts = []
            for i, part in enumerate(parts):
                if i < 3:  # Keep first 3 parts
                    important_parts.append(part)
                elif any(keyword in part.lower() for keyword in ['%', '$', 'income', 'age', 'year']):
                    important_parts.append(part)
                
                # Check if we're getting close to limit
                temp_name = ' '.join(important_parts)
                if len(temp_name) > max_length - 10:  # Leave some buffer
                    break
                    
            shortened = ' '.join(important_parts)
    
    # Final truncation if still too long
    if len(shortened) > max_length:
        shortened = shortened[:max_length-3] + '...'
        
    return shortened.strip()

def add_county_names_from_metadata(self, df):
    """
    Adds a 'County' column to the DataFrame by extracting the county name from the 'NAME' column.
    """
    df = df.loc[:, ~df.columns.duplicated()]
    df['County'] = df['NAME'].str.extract(r'Census Tract [^;]+; ([^;]+) County')
    return df
def fetch_data(self, variables: List[str], geography: str = 'county', 
                state: str = '01', county: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch data from Census API
    
    Args:
        variables: List of variable codes
        geography: Geographic level (tract, county, state)
        state: State FIPS code
        county: County FIPS code (optional)
    """
    params = {
        "get": ",".join(["NAME"] + variables),
        "for": f"{geography}:*",
        "key": self.api_key
    }
    
    # Fix the 'in' parameter format to match working notebook
    if geography == 'tract':
        params["in"] = f"state:{state}"
    elif geography == 'county':
        params["in"] = f"state:{state}"
    else:
        params["in"] = f"state:{state}"
        
    response = requests.get(self.base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        return df
    else:
        print(f"API URL: {self.base_url}")
        print(f"API Params: {params}")
        raise Exception(f"API error {response.status_code}: {response.text}")
