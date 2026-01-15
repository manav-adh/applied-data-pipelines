# Nevada Census Data Visualization Automation

This system provides automated visualization capabilities for Nevada Census data, allowing you to create professional visualizations with minimal code.

## Quick Start

```python
from automated_visualization import quick_visualize

# Create a bar chart of population by county
fig = quick_visualize(["DP05_0001E"], "bar")
plt.show()

# Create a choropleth map of health insurance coverage
fig = quick_visualize(["DP03_0097PE"], "choropleth")
fig.show()
```

##  Supported Visualization Types

1. **Bar Chart** (`"bar"`) - Compare values across counties
2. **Box Plot** (`"box"`) - Show distribution of values
3. **Choropleth Map** (`"choropleth"`) - Geographic visualization by census tract
4. **Scatter Plot** (`"scatter"`) - Show relationships between variables
5. **Heatmap** (`"heatmap"`) - Correlation matrix visualization

## Core Functions

### 1. `quick_visualize(variables, viz_type, county=None, **kwargs)`

**Parameters:**
- `variables` (list): Census variable codes (e.g., `["DP05_0001E", "DP05_0002E"]`)
- `viz_type` (str): Visualization type (`"bar"`, `"box"`, `"choropleth"`, `"scatter"`, `"heatmap"`)
- `county` (str, optional): County name to filter data (e.g., `"Clark"`, `"Washoe"`)
- `**kwargs`: Additional parameters (e.g., `figsize=(12, 8)`, `color_scale="Reds"`)

**Returns:** Matplotlib figure or Plotly figure object

### 2. `get_available_variables()`

Returns a dictionary of all available Census variables and their descriptions.

### 3. `search_variables(keyword)`

Search for variables containing a specific keyword.

## Examples

### Population Analysis
```python
# Total population by county
fig = quick_visualize(["DP05_0001E"], "bar")

# Male vs Female population
fig = quick_visualize(["DP05_0002E", "DP05_0003E"], "bar")

# Population distribution across counties
fig = quick_visualize(["DP05_0001E"], "box")
```

### Income Analysis
```python
# Income brackets by county
income_vars = ["DP03_0054PE", "DP03_0055PE", "DP03_0056PE", "DP03_0057PE"]
fig = quick_visualize(income_vars, "bar")

# Income distribution in Clark County
fig = quick_visualize(income_vars, "box", county="Clark")
```

### Geographic Analysis
```python
# Health insurance coverage by census tract
fig = quick_visualize(["DP03_0097PE"], "choropleth")

# Poverty levels with custom color scale
fig = quick_visualize(["DP03_0129E"], "choropleth", color_scale="Reds")
```

### Relationship Analysis
```python
# Income vs Health Insurance correlation
fig = quick_visualize(["DP03_0057PE", "DP03_0097PE"], "scatter")

# Correlation matrix of multiple variables
vars = ["DP05_0001E", "DP03_0054PE", "DP03_0055PE", "DP03_0097PE"]
fig = quick_visualize(vars, "heatmap")
```

## Advanced Usage

### Using the Class Directly
```python
from automated_visualization import NevadaCensusVisualizer

visualizer = NevadaCensusVisualizer()

# Fetch data
df = visualizer.fetch_data(["DP05_0001E", "DP05_0002E"])

# Clean data
df_clean = visualizer.clean_data(df)

# Create visualization
fig = visualizer.create_visualization(["DP05_0001E"], "bar")
```

### Custom Parameters
```python
# Custom figure size
fig = quick_visualize(["DP05_0001E"], "bar", figsize=(15, 8))

# Custom color scale for choropleth
fig = quick_visualize(["DP03_0097PE"], "choropleth", color_scale="Blues")

# County-specific analysis
fig = quick_visualize(["DP03_0054PE"], "box", county="Washoe")
```

##  Common Variable Codes

### Population Variables
- `DP05_0001E` - Total Population
- `DP05_0002E` - Male Population
- `DP05_0003E` - Female Population

### Income Variables
- `DP03_0054PE` - Households $15k-$25k (%)
- `DP03_0055PE` - Households $25k-$35k (%)
- `DP03_0056PE` - Households $35k-$50k (%)
- `DP03_0057PE` - Households $50k-$75k (%)
- `DP03_0058PE` - Households $75k-$100k (%)
- `DP03_0059PE` - Households $100k-$150k (%)

### Health & Social Variables
- `DP03_0097PE` - Private Health Insurance (%)
- `DP03_0129E` - People Below Poverty Level

##  Finding Variables

```python
from automated_visualization import search_variables

# Search for income-related variables
income_vars = search_variables("income")
print(income_vars)

# Search for health-related variables
health_vars = search_variables("health")
print(health_vars)
```

##  File Requirements

- `automated_visualization.py` - Main automation system
- `nevada_tracts_2022.json` - GeoJSON file for choropleth maps (from shapefile)
- Census API key (included in code)

## Installation

1. Ensure all required packages are installed:
```bash
pip install pandas numpy matplotlib seaborn plotly geopandas requests
```

2. Place the `automated_visualization.py` file in your working directory

3. Ensure the GeoJSON file is available for choropleth maps

## Customization

### Adding New Visualization Types
```python
class NevadaCensusVisualizer:
    def _create_custom_viz(self, df, variables, var_types, **kwargs):
        # Your custom visualization logic here
        pass
    
    def create_visualization(self, variables, viz_type, county_filter=None, **kwargs):
        # Add your new type to the if-elif chain
        elif viz_type.lower() == 'custom':
            return self._create_custom_viz(df_clean, variables, var_types, **kwargs)
```

### Custom Data Processing
```python
# Override the clean_data method for custom processing
def clean_data(self, df, rename_columns=True):
    df_clean = super().clean_data(df, rename_columns)
    # Add your custom processing here
    return df_clean
```

## Error Handling

The system includes comprehensive error handling for:
- Invalid variable codes
- API connection issues
- Missing GeoJSON files
- Data processing errors
- Unsupported visualization types

##  Output Formats

- **Matplotlib figures**: For bar, box, scatter, and heatmap visualizations
- **Plotly figures**: For choropleth maps (interactive)

##  Batch Processing

```python
# Generate multiple visualizations
variable_groups = {
    "Population": ["DP05_0001E", "DP05_0002E", "DP05_0003E"],
    "Income": ["DP03_0054PE", "DP03_0055PE", "DP03_0056PE"],
    "Health": ["DP03_0097PE"]
}

for group_name, variables in variable_groups.items():
    for viz_type in ["bar", "box", "heatmap"]:
        fig = quick_visualize(variables, viz_type)
        # Save or display figure
```

## Best Practices

1. **Variable Selection**: Use percentage variables (ending in 'PE') for comparisons
2. **Visualization Choice**: 
   - Use 'bar' for county comparisons
   - Use 'box' for distribution analysis
   - Use 'choropleth' for geographic patterns
   - Use 'scatter' for relationships
   - Use 'heatmap' for correlations
3. **County Filtering**: Use county filters for detailed analysis
4. **Data Validation**: Always check the output for reasonable values

##  Support

For issues or questions:
1. Check the error messages for specific issues
2. Verify variable codes are valid
3. Ensure GeoJSON file is present for choropleth maps
4. Check API key validity

## Updates

The system automatically:
- Fetches variable descriptions from the Census API
- Handles data type conversions
- Applies appropriate aggregations
- Generates meaningful titles and labels
<img width="468" height="641" alt="image" src="https://github.com/user-attachments/assets/69137480-6b8f-49b7-bff0-4ab7f5e5f7d2" />
