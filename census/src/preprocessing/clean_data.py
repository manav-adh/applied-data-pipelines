def clean_data(self, df: pd.DataFrame, rename_columns: bool = True) -> pd.DataFrame:
    """Clean and prepare data for visualization"""
    df_clean = df.copy()
    
    if rename_columns:
        # Create rename dictionary from available descriptions
        rename_dict = {}
        for col in df_clean.columns:
            if hasattr(self, 'variable_descriptions') and col in self.variable_descriptions:
                # Use shortened names for better display
                original_desc = self.variable_descriptions[col]
                shortened_desc = self._shorten_variable_name(original_desc)
                rename_dict[col] = shortened_desc
            else:
                rename_dict[col] = col
        
        df_clean = df_clean.rename(columns=rename_dict)
    
    # Extract county name from NAME column
    if 'NAME' in df_clean.columns or any('name' in col.lower() for col in df_clean.columns):
        name_col = 'NAME' if 'NAME' in df_clean.columns else [col for col in df_clean.columns if 'name' in col.lower()][0]
        df_clean['County'] = df_clean[name_col].str.extract(r'Census Tract [^;]+; ([^;]+) County')
    
    # Convert numeric columns
    for col in df_clean.columns:
        if col not in ['NAME', 'County', 'state', 'county', 'tract']:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    return df_clean

def detect_variable_type(self, variables: list[str]) -> dict[str, str]:
    """Detect the type of variables for appropriate visualization"""
    variable_types = {}
        
    for var in variables:
        if 'PE' in var:
            variable_types[var] = 'percentage'
        elif 'E' in var:
            variable_types[var] = 'estimate'
        elif 'M' in var:
            variable_types[var] = 'margin_of_error'
        else:
            variable_types[var] = 'unknown'
        
    return variable_types

