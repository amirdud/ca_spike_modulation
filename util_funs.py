import numpy as np


def find_nearest(x, target, type='below'):
    '''
    find the closest value and its index above/below
        a certain number'

    x: np.array of values
    target: target value to find the closest value to it
    type: find the closest below/above the target value

    val: the closest value
    idx: the index of the closest value

    '''

    if type == 'above':
        inidices = np.where(x >= target)[0]
        val = np.min(x[inidices])
        idx = inidices[0]

    elif type == 'below':
        inidices = np.where(x <= target)[0]
        val = np.max(x[inidices])
        idx = np.argmax(x[inidices])

    return val, idx


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def find_graph_intersections(f, g):
    indices = np.argwhere(np.diff(np.sign(f - g))).flatten()
    return indices


def convert_var_to_numpy(var):
    var_np = {}
    for key_1 in var:
        if isinstance(var[key_1], dict):
            var_np[key_1] = {}

            for key_2 in var[key_1]:
                var_np[key_1][key_2] = np.array(var[key_1][key_2])

        else:
            var_np[key_1] = np.array(var[key_1])

    return var_np


def get_shifted_time(t, t_ref):
    return t - t_ref


# https://gist.github.com/dmeliza/3251476
from matplotlib.offsetbox import AnchoredOffsetbox

class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, barcolor="black", barwidth=None,
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, ec=barcolor, lw=barwidth, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx, minimumdescent=False)
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)


def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """

    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex: ax.xaxis.set_visible(False)
    if hidey: ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)

    return sb