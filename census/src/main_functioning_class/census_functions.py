# %%
import pandas as pd 
import requests
import numpy as np
import geopandas as gpd
import folium
import requests
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import List, Dict, Optional, Self, Union
import warnings
warnings.filterwarnings('ignore')

class NevadaCensusVisualizer:
    """
    Automated visualization system for Nevada Census data.
    Takes any variables and visualization type and returns appropriate plots.
    """
    "To look for specific states change the state number in the fetch_data function"
    
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

    def create_visualization(self, variables: list[str], viztype: str, county_filter : Optional[str] = None, **kwargs) -> Union[plt.figure, go.Figure]:
        df = self.fetch_data(variables)
        df = self.add_county_names_from_metadata(df)
        df_clean = self.clean_data(df)
        #filter by county if specified 
        if county_filter: 
            df_clean = df_clean[df_clean['County'] == county_filter]
            df = df[df['County'] == county_filter]
        #detect variable types 
        var_types = self.detect_variable_type(variables)

        #create appropriate viz 
        if viztype.lower() == 'bar':
            return self.create_bar_chart(df_clean, variables, var_types, **kwargs)
        elif viztype.lower() == 'box':
            return self._create_box_plot(df_clean, variables, var_types, **kwargs)
        elif viztype.lower() == 'choropleth':
            return self._create_choropleth(df_clean, variables, var_types, **kwargs)
        elif viztype.lower() == 'scatter':
            return self._create_scatter_plot(df_clean, variables, var_types, **kwargs)
        elif viztype.lower() == 'heatmap':
            return self._create_heatmap(df_clean, variables, var_types, **kwargs)
        else:
            raise ValueError(f"Unsupported visualization type: {viztype}")   

    def create_bar_chart(self, df: pd.DataFrame, variables : list[str], vartypes = Dict[str, str], **kwargs) -> plt.Figure:
        fig, ax = plt.subplots(figsize = kwargs.get('figsize', (12,6)))
        column_names = [self._shorten_variable_name(self.variable_descriptions.get(var, var)) for var in variables]
        #group by county and aggregate 
        if 'County' in df.columns:
            agg_df = df.groupby('County')[column_names].mean().reset_index()
            if len(variables) > 1:
                bottom = np.zeros(len(agg_df))
                for var in column_names:
                    if var in agg_df.columns: 
                        ax.bar(agg_df['County'], agg_df[var], bottom=bottom, label = var)
                        bottom += agg_df[var]
                ax.legend()
            else:
                var = column_names[0]
                ax.bar(agg_df['County'], agg_df[var])

            
            ax.set_xlabel('County')
            ax.set_ylabel(column_names[0])
            ax.set_title(f'{column_names[0]} by County')
            plt.xticks(rotation= 55, ha = 'right')
        plt.tight_layout()
        return fig
    
    def _create_box_plot(self, df: pd.DataFrame, variables: List[str], var_types: Dict[str,str], **kwargs ) -> plt.figure:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10,6)))
        column_names = [self._shorten_variable_name(self.variable_descriptions.get(var, var)) for var in variables]
        if 'County' in df.columns:
            melt_vars = [var for var in column_names if var in df.columns]
            melted_df = df.melt(id_vars = ['County'],
                                value_vars= melt_vars,
                                var_name = 'Variable',
                                value_name = 'Value')
            sns.boxplot(data=melted_df, x='Variable', y='Value', hue='County', ax=ax, fill=True)
            plt.xticks(rotation = 45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def _create_choropleth(self, df: pd.DataFrame, variables: list[str], var_types: dict[str, str], **kwargs) -> go.Figure:
        # Path to your shapefile (do not include the .shp extension, just the base name)
        shapefile_path = 'tl_2024_01_tract/tl_2024_01_tract.shp'

        # Load the shapefile
        gdf = gpd.read_file(shapefile_path)
        print(gdf['GEOID'].head())

        if all(col in df.columns for col in ['state', 'county', 'tract']):
            # Clean and format GEOID components
            df['state'] = df['state'].astype(str).str.zfill(2)
            df['county'] = df['county'].astype(str).str.zfill(3) 
            df['tract'] = df['tract'].astype(str).str.zfill(6)
            
            # Create GEOID: state(2) + county(3) + tract(6) = 11 digits
            df['GEOID'] = df['state'] + df['county'] + df['tract']
        
        print("Census data GEOID sample:", df['GEOID'].head())
        print("GEOID lengths - Shapefile:", gdf['GEOID'].str.len().unique())
        print("GEOID lengths - Census:", df['GEOID'].str.len().unique())
        
        # Ensure both are strings for matching
        gdf['GEOID'] = gdf['GEOID'].astype(str)
        df['GEOID'] = df['GEOID'].astype(str)
        
        # Check for matches
        matches = set(gdf['GEOID']) & set(df['GEOID'])
        print(f"Number of matching GEOIDs: {len(matches)} out of {len(df)} census tracts and {len(gdf)} shapefile tracts")
        
        # Get shortened variable name for better display
        original_var = self._get_var_description(variables[0])
        color_var = self._shorten_variable_name(original_var)
        
        # Find the actual column name in the dataframe (which should be the shortened version)
        actual_column = None
        for col in df.columns:
            if col == color_var or self._shorten_variable_name(col) == color_var:
                actual_column = col
                break
        
        if actual_column is None:
            # Fallback to original logic
            actual_column = original_var
            color_var = original_var
        
        print("Color variable (shortened):", color_var)
        print("Actual column name:", actual_column)
        
        gdf['GEOID'] = gdf['GEOID'].astype(str)
        geojson = json.loads(gdf.to_json())

        if actual_column not in df.columns: 
            raise ValueError(f"Variable {actual_column} not found in data. Available columns: {list(df.columns)}")
            
        df[actual_column] = pd.to_numeric(df[actual_column], errors='coerce')
        df = df.dropna(subset=[actual_column])
        df = df[df[actual_column] >= 0]

        fig = px.choropleth(
            df,
            geojson=geojson,
            locations='GEOID',
            featureidkey="properties.GEOID",
            color=actual_column,
            color_continuous_scale=kwargs.get('color_scale', "Viridis"),
            title=color_var,  # Use shortened name for title
            labels={actual_column: color_var}  # Use shortened name in legend
        )

        
        fig.update_geos(fitbounds = 'locations', visible = False)
        return fig


# Example usage
variables = ["DP05_0001E"]

def quick_visualize(variables: List[str], viz_type: str, 
                   county: Optional[str] = None, **kwargs):
    visualizer = NevadaCensusVisualizer()
    return visualizer.create_visualization(variables, viz_type, county, **kwargs)


