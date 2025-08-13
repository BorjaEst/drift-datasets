# Plotting and Visualization

The `drift_datasets` library includes plotting capabilities to visualize datasets and drift patterns.

## Basic Usage

```python
import drift_datasets

# Create or load a dataset
dataset = drift_datasets.create_dataset(config)

# Create a basic plot
fig = drift_datasets.plot_dataset(dataset)

# Create a scatter plot with drift points highlighted
fig = drift_datasets.plot_dataset(
    dataset,
    plot_type="scatter",
    show_drift_points=True,
    color_by_target=True
)

# Create a time series plot showing features over time
fig = drift_datasets.plot_dataset(
    dataset,
    plot_type="time_series",
    show_drift_points=True,
    show_concept_segments=True
)

# Save plot to file
fig = drift_datasets.plot_dataset(
    dataset,
    save_path="my_dataset_plot.png"
)
```

## Plot Types

### Auto Selection (`plot_type="auto"`)

Automatically selects the best plot type based on the dataset characteristics:

- 1 feature → time series
- 2 features → scatter plot  
- 3+ features & <10k samples → pairwise plot
- 3+ features & ≥10k samples → time series

### Scatter Plot (`plot_type="scatter"`)

Creates a 2D scatter plot using the first two features. Best for:

- 2D datasets
- Understanding feature relationships
- Visualizing class separation

### Time Series (`plot_type="time_series"`)  

Shows features as lines over sample index (time). Best for:

- High-dimensional datasets
- Understanding temporal patterns
- Visualizing drift evolution over time

### Pairwise Plot (`plot_type="pairplot"`)

Creates a matrix of pairwise scatter plots and histograms. Best for:

- 3-4 dimensional datasets
- Comprehensive feature analysis
- Small to medium datasets (<10k samples)

## Visualization Options

### Drift Point Annotations (`show_drift_points=True`)

- Adds vertical lines at drift points
- Color-coded by drift type
- Includes drift type labels

### Concept Segments (`show_concept_segments=True`)

- Highlights different concept regions with background colors
- Useful for understanding concept stability periods

### Target-based Coloring (`color_by_target=True`)

- Colors points/lines by target variable
- Works best with categorical targets with few classes (<10)
- Helps visualize class separation and concept drift effects

### Feature Selection (`features=["feature_1", "feature_2"]`)

- Plot only specific features
- Useful for focusing on particular aspects of the data

### Custom Styling

- `figsize=(width, height)` - Set figure size
- `save_path="filename.png"` - Save to file

## Examples

### Basic Drift Visualization

```python
# Highlight drift points and concept segments
fig = drift_datasets.plot_dataset(
    dataset,
    show_drift_points=True,
    show_concept_segments=True,
    color_by_target=True
)
```

### Feature-specific Analysis

```python
# Plot only specific features
fig = drift_datasets.plot_dataset(
    dataset,
    features=["temperature", "humidity"],
    plot_type="scatter",
    color_by_target=True
)
```

### High-dimensional Dataset

```python
# For datasets with many features
fig = drift_datasets.plot_dataset(
    dataset,
    plot_type="time_series",  # Shows all features as time series
    show_drift_points=True
)
```

### Export for Publications

```python
# Create publication-ready plots
fig = drift_datasets.plot_dataset(
    dataset,
    plot_type="scatter",
    figsize=(10, 8),
    save_path="publication_figure.png"
)
```

## Integration with Analysis Workflow

The plotting function integrates seamlessly with other library features:

```python
# Analyze and visualize dataset splits
train_data, test_data = dataset.split_temporal(ratio=0.7)

# Plot training data
train_fig = drift_datasets.plot_dataset(
    train_data, 
    show_drift_points=True,
    color_by_target=True
)

# Plot test data  
test_fig = drift_datasets.plot_dataset(
    test_data,
    show_drift_points=True,
    color_by_target=True
)

# Visualize drift detection features only
modeling_features = dataset.get_drift_detection_features()
fig = drift_datasets.plot_dataset(
    dataset,
    features=list(modeling_features.columns),
    show_drift_points=True
)
```

## Return Value

All plotting functions return a `matplotlib.figure.Figure` object, which can be:

- Displayed in Jupyter notebooks
- Saved to various formats (PNG, PDF, SVG, etc.)
- Further customized with matplotlib
- Embedded in applications

```python
fig = drift_datasets.plot_dataset(dataset)

# Display in Jupyter
fig.show()

# Save manually
fig.savefig("my_plot.pdf", dpi=300, bbox_inches='tight')

# Access axes for customization
axes = fig.get_axes()[0]
axes.set_title("My Custom Title")
```
