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

def create_bar_chart(df: pd.DataFrame, variables : list[str], vartypes = Dict[str, str], **kwargs) -> plt.Figure:
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
