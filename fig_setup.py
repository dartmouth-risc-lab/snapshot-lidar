import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib import rc
from pylab import rcParams


def fig_setup():
    XS = 10
    SMALL = 14
    MEDIUM = 15
    TITLE = 24
    XL = 20
    
    plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": TITLE,
    "axes.labelsize": MEDIUM,
    'axes.titlesize':XL,
    "xtick.labelsize": SMALL,
    "ytick.labelsize": SMALL,
    "legend.fontsize": XS,
    # "xtick.bottom": "false",
    # "ytick.left": "false",
    # "xtick.labelbottom": "false",
    # "ytick.labelleft": "false",
    "text.usetex": "true",
    "pdf.fonttype": 42

    })
