import matplotlib.pyplot as plt
from cycler import cycler

class PlotConfig:
    def __init__(self):

        self.colors = {
            'primary': '#4878CF',
            'secondary': '#B47CC7',
            'accent1': '#D65F5F',
            'accent2': '#55A868',
            'neutral': '#6A994E',
            'light': '#E9EDC9',
            'dark': '#264653',
            'real_data': '#4878CF',
            'generated_data': '#B47CC7',
            'loss_curve': '#55A868',
            'true_contour': 'black',
        }        
        self.setup_style()
    
    def setup_style(self):
        """Apply the consistent plotting style using matplotlib rcParams."""
        plt.style.use('seaborn-v0_8-muted')

        colors_for_cycle = ['#001C7F',
                             '#7600A1',
                             '#009E73',
                             '#8172B2',
                             '#017517',
                             '#006374',
                             '#7A68A6',
                             '#467821',
                             '#0072B2',
                             '#B47CC7',
                             '#6A994E',
                             '#2E86AB',
                             '#CC79A7',
                             '#55A868',
                            ]

        # Update rcParams
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            # --- Use the special list ---
            'axes.prop_cycle': cycler('color', colors_for_cycle),
            # ---
            # --- Set the DEFAULT colormap ---
            'image.cmap': 'coolwarm',
        })

# --- Initialize Configuration ---
# Create a single instance of the configuration class when the module is imported
plot_config = PlotConfig()
colors = plot_config.colors

def get_color_palette():
    """Returns the configured color palette."""
    return colors

def get_default_figsize():
    """Returns the configured default figure size."""
    return tuple(plot_config.setup_style().get('figure.figsize', (12, 8)))
