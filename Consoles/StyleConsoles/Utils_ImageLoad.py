from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from Consoles.Consoles8Gafchromic.Concept8GafMeasurementComparison import GafImage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


def load_image(folder_path, image, background_subtraction=True, normalization=True, position=None):
    # Automatic assigning of Analyzer, background, normalization
    if image[-4:] == '.bmp':
        OriginalGaf = GafImage(folder_path / image)
        OriginalGaf.load_image()
        # '''
        if 'matrix211024_006.bmp' in image:
            OriginalGaf.image = OriginalGaf.image[::-1]
            OriginalGaf.image = OriginalGaf.image[:, ::-1]
        # '''
        OriginalGaf.transform_to_normed(max_n=1e5)
        print('Gafchromic Image detected and loaded - note that no Analyzer instance will be returned!')
        return OriginalGaf
    elif '_230924' in folder_path.name:
        mapping = Path('../../Files/mapping.xlsx')
        data = pd.read_excel(mapping, header=1)
        channel_assignment = [int(k[-3:]) - 1 for k in data['direction_2']]
        readout, position_parser = (lambda x, y: ams_channel_assignment_readout(x, y, channel_assignment),
                                    standard_position)
        A = Analyzer((1, 128), 0.5, 0.0, readout=readout,
                     position_parser=position_parser, voltage_parser=standard_voltage, current_parser=standard_current)
        # Correct sizing of the arrays
        if 'Array3' in image:
            A.diode_size = (0.25, 0.5)
            A.diode_size = (0.17, 0.4)
            A.diode_spacing = (0.08, 0.1)
        else:
            A.diode_size = (0.5, 0.5)
            A.diode_size = (0.4, 0.4)
            A.diode_spacing = (0.1, 0.1)
        # Dark Subtraction - correct file assignment
        dark_path = folder_path
        if 'Array3_Logo' in image:
            dark = ['Array3_VoltageScan_dark_nA_1.8_x_0.0_y_40.0.csv']
        elif 'Array3' in image:
            dark = ['Array3_VoltageScan_dark_nA_1.0_x_0.0_y_40.0.csv']
        else:
            dark = ['voltage_scan_no_beam_nA_1.8000000000000005_x_20.0_y_70.0.csv',
                         'd2_1n_5s_flat_calib_nA_1.8000000000000007_x_20.0_y_70.0.csv']
        # Norm Assignment
        norm_path = folder_path
        if 'Array3' in image:
            norm = ['Array3_DiffuserYScan']
        else:
            norm = ['uniformity_scan_']
        norm_module = normalization_from_translated_array_v2

    elif '_111024' in folder_path.name:
        pass
    elif '_211024' in folder_path.name:
        pass
    elif '_221024' in folder_path.name:
        mapping = Path('../../Files/mapping.xlsx')
        data = pd.read_excel(mapping, header=1)
        channel_assignment = [int(k[-3:]) - 1 for k in data['direction_2']]
        readout = lambda x, y: ams_2line_readout(x, y, channel_assignment=channel_assignment)

        A = Analyzer((2, 64), (0.4, 0.4), (0.1, 0.1), readout=readout,
                     diode_offset=[[0, - 0.25], np.zeros(64)], position_parser=standard_position,
                     voltage_parser=standard_voltage, current_parser=standard_current)
        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')

        dark = ['2Line_DarkVoltageScan_200_ um_0_nA_nA_1.9_x_22.0_y_66.625.csv']

        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
        norm = ['2Line_YScan_']
        norm_module = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
            list_of_files, instance, method, align_lines=True)

    elif '_211124' in folder_path.name:
        mapping = Path('../../Files/mapping.xlsx')
        direction1 = pd.read_excel(mapping, header=1)
        direction1 = np.array([int(k[-3:]) for k in direction1['direction_1']])
        direction2 = pd.read_excel(mapping, header=1)
        direction2 = np.array([int(k[-3:]) for k in direction2['direction_2']])

        mapping = Path('../../Files/Mapping_SmallMatrix2.xlsx')
        data2 = pd.read_excel(mapping, header=None)
        mapping_map = data2.to_numpy().flatten()
        translated_mapping = np.array([direction2[np.argwhere(direction1 == i)[0][0]] - 1 for i in mapping_map])

        readout, position_parser = lambda x, y: ams_2D_assignment_readout(x, y,
                                                                          channel_assignment=translated_mapping), standard_position

        A = Analyzer((11, 11), 0.4, 0.1, readout=readout)

        dark_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
        dark = ['12_2DSmall_miscshape_']

        norm_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_211124/')
        norm = '8_2DSmall_yscan_'
        norm_module = lambda list_of_files, instance, method='least_squares': normalization_from_translated_array_v3(
            list_of_files, instance, method, align_lines=True)
    elif '_260325' in folder_path.name:
        pass
    elif '_19062024' in folder_path.name:
        readout, position_parser = ams_constant_signal_readout, standard_position

        A = Analyzer((1, 64), 0.4, 0.1, readout=readout,
                     position_parser=position_parser, voltage_parser=standard_voltage)

        # Dark Subtraction - correct file assignment
        dark_path = folder_path
        dark = ['d2_1n_3s_beam_all_without_diffuser_dark.csv']

        # Norm Assignment
        norm_path = folder_path
        norm = ['5s_flat_calib_']

        norm_module = simple_normalization
    else:
        print('The given measurement / path combination is not registered. Please recheck the input!')
        return None

    # Filtering for correct files - Logo would be found in Array3_Logo...
    if image == 'Logo':
        A.set_measurement(folder_path, image, blacklist=['png', 'Array3'])
    else:
        A.set_measurement(folder_path, image)

    A.load_measurement()

    if background_subtraction:
        A.set_dark_measurement(dark_path, dark)
    if normalization:
        A.normalization(norm_path, norm, normalization_module=norm_module)

    A.update_measurement(dark=background_subtraction, factor=normalization)

    if '_19062024' in folder_path.name:
        A.create_map(inverse=[False, False])
    else:
        A.create_map(inverse=[True, False])

    if len(A.maps) > 1 and position is None:
        A.maps = [A.maps[0]]
    elif len(A.maps) > 1 and position is not None:
        A.maps = [i for i in A.maps if i['position'] == position]
    return A


def add_diode_geometry_indicator(ax, analyzer, position='upper right', fig=None):
    if fig is None:
        return

    # Force rendering to get correct sizes
    fig.canvas.draw()

    # Get figure size in pixels and DPI
    fig_width_px = fig.get_size_inches()[0] * fig.dpi
    fig_height_px = fig.get_size_inches()[1] * fig.dpi

    # Calculate pitch
    pitch_x = analyzer.diode_size[0] + analyzer.diode_spacing[0]
    pitch_y = analyzer.diode_size[1] + analyzer.diode_spacing[1]

    # Get axis limits
    x_lims = ax.get_xlim()
    y_lims = ax.get_ylim()

    # Calculate scaling (units per pixel)
    scale_x = (x_lims[1] - x_lims[0]) / fig_width_px
    scale_y = (y_lims[1] - y_lims[0]) / fig_height_px

    # Determine desired indicator size relative to plot size
    desired_size_fraction = 0.12  # 12% of plot size
    reference_size = min(x_lims[1] - x_lims[0], y_lims[1] - y_lims[0]) * desired_size_fraction

    # Calculate scaling factor for indicator
    scale_factor = reference_size / max(pitch_x, pitch_y)

    # Scale pitch and diode size
    scaled_pitch_x = pitch_x * scale_factor
    scaled_pitch_y = pitch_y * scale_factor
    scaled_diode_x = analyzer.diode_size[0] * scale_factor
    scaled_diode_y = analyzer.diode_size[1] * scale_factor
    scaled_spacing_x = analyzer.diode_spacing[0] * scale_factor
    scaled_spacing_y = analyzer.diode_spacing[1] * scale_factor

    # Calculate text for pitch values
    pitch_text_x = f'{pitch_x:.2f} mm'
    pitch_text_y = f'{pitch_y:.2f} mm'

    # Get text size in data coordinates
    renderer = fig.canvas.get_renderer()
    text_size_scaling = 0.15  # Adjust this factor if needed

    # Calculate text heights and widths in data coordinates
    bbox_x = ax.text(0, 0, pitch_text_x, fontsize=8).get_window_extent(renderer=renderer)
    bbox_y = ax.text(0, 0, pitch_text_y, fontsize=8).get_window_extent(renderer=renderer)
    text_width = bbox_x.width * (x_lims[1] - x_lims[0]) / fig_width_px
    text_height = bbox_x.height * (y_lims[1] - y_lims[0]) / fig_height_px

    # Clear temporary text objects
    for text in ax.texts[:]:
        text.remove()

    # Calculate padding based on text size
    corner_padding = max(reference_size * 0.2, text_width * 0.7) * 0.9  # Ensure enough space for text
    text_offset = max(text_height * 1.2, reference_size * 0.15)  # Space between arrow and text

    if position == 'upper right':
        base_x = x_lims[1] - scaled_pitch_x - corner_padding
        base_y = y_lims[1] - scaled_pitch_y - corner_padding
    else:  # 'upper left'
        base_x = x_lims[0] + corner_padding
        base_y = y_lims[1] - scaled_pitch_y - corner_padding

    # Draw outer grey square (pitch) with light fill
    outer_square = plt.Rectangle((base_x, base_y), scaled_pitch_x, scaled_pitch_y,
                                 facecolor='lightgrey', edgecolor='grey', linewidth=1, alpha=1)
    ax.add_patch(outer_square)

    # Draw inner golden square (diode size) with light fill
    inner_x = base_x + scaled_spacing_x / 2
    inner_y = base_y + scaled_spacing_y / 2
    inner_square = plt.Rectangle((inner_x, inner_y),
                                 scaled_diode_x, scaled_diode_y,
                                 facecolor='gold', edgecolor='gold', linewidth=1, alpha=1)
    ax.add_patch(inner_square)

    # Add arrows and text with adjusted positions
    arrow_props = dict(arrowstyle='<->', color='grey')

    # Horizontal arrow and text
    arrow_y = base_y - text_offset
    ax.annotate('', xy=(base_x, arrow_y),
                xytext=(base_x + scaled_pitch_x, arrow_y),
                arrowprops=arrow_props)

    # Position text with enough space from arrow and square
    text_y_pos = arrow_y - text_offset
    # Ensure text doesn't go below axis
    if text_y_pos > y_lims[0] + text_height:  # Check if text would be visible
        ax.text(base_x + scaled_pitch_x / 2, text_y_pos,
                pitch_text_x, ha='center', va='top', fontsize=8)

    # Vertical arrow and text
    arrow_x = base_x - text_offset
    ax.annotate('', xy=(arrow_x, base_y),
                xytext=(arrow_x, base_y + scaled_pitch_y),
                arrowprops=arrow_props)

    # Position text with enough space from arrow and square
    text_x_pos = arrow_x - text_offset
    if position == 'upper right':
        if text_x_pos > x_lims[0] + text_width:  # Check if text would be visible
            ax.text(text_x_pos, base_y + scaled_pitch_y / 2,
                    pitch_text_y, ha='right', va='center', rotation=90, fontsize=8)
    else:
        if text_x_pos > x_lims[0] + text_width:  # Check if text would be visible
            ax.text(text_x_pos, base_y + scaled_pitch_y / 2,
                    pitch_text_y, ha='right', va='center', rotation=90, fontsize=8)


def add_png_icon(ax, instance, location='top right', zoom=96/400, translation=None, background=False):
    """
    Adds a PNG icon to a specified location within a Matplotlib axis, with options for customization such as
    zoom level, translations, and background.

    This function facilitates displaying icons onto a plot based on the attributes of the given `instance`.
    It uses preset paths to specific icons, matched with particular geometry configurations. Icons can be
    aligned in different positions on the plot, zoomed to a given scale, and supplemented with arrows to
    indicate translation directions. Optional background shading can also be added for visual clarity.

    :param ax: Matplotlib axis on which the icon will be added.
    :param instance: Object containing geometry attributes like diode_dimension and diode_size, used to select
                     the appropriate icon.
    :param location: String specifying the location on the axis where the icon will appear. Supports 'top right',
                     'top left', 'bottom left', and 'bottom right'. Defaults to 'top right'.
    :param zoom: Float defining the scaling level of the icon. Defaults to 96/400.
    :param translation: List of dimensions ('x' or 'y') in which directional arrows will be added to indicate movement.
                        Defaults to None.
    :param background: Boolean indicating whether to draw a white rectangle as a background box behind the icon.
                       Defaults to False.
    :return: None if the geometry specified in `instance` does not correspond to any predefined icon. Otherwise,
             modifies the axis in place by adding the icon and optional visual elements.
    """
    if instance.diode_dimension[0] == 11 and instance.diode_dimension[1] == 11 and instance.diode_size[0] == 0.8:
        icon_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/11x11Big.png')
    elif instance.diode_dimension[0] == 11 and instance.diode_dimension[1] == 11 and instance.diode_size[0] == 0.4:
        icon_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/11x11.png')
    elif instance.diode_dimension[0] == 1 and instance.diode_dimension[1] == 128 and instance.diode_size[0] == 0.4:
        icon_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/1x128.png')
    elif instance.diode_dimension[0] == 1 and instance.diode_dimension[1] == 128 and instance.diode_size[0] == 0.17:
        icon_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/1x128Small.png')
    elif instance.diode_dimension[0] == 2 and instance.diode_dimension[1] == 64 and instance.diode_size[0] == 0.4:
        icon_path = Path('/Users/nico_brosda/Cyrce_Messungen/Style/2x64.png')
    else:
        print('The given geometry is not exisiting a icon to add into graph')
        return None

    # Load the PNG image using PIL
    with Image.open(icon_path) as image:
        # Wrap it for matplotlib
        im = OffsetImage(image, zoom=zoom)

        # Define location anchor and box alignment
        location_map = {
            'top right': ((0.98, 0.98), (1, 1)),
            'top left': ((0.02, 0.98), (0, 1)),
            'bottom left': ((0.02, 0.02), (0, 0)),
            'bottom right': ((0.98, 0.02), (1, 0)),
        }
        (xy, box_alignment) = location_map.get(location, ((1, 1), (1, 1)))

        # Create and add icon box
        ab = AnnotationBbox(
            im,
            xy=xy,
            xycoords='axes fraction',
            box_alignment=box_alignment,
            frameon=False,
            pad=0
        )
        ax.add_artist(ab)

        # Force rendering to get accurate icon position
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Get icon bounding box in display (pixel) coords
        icon_bbox_display = ab.get_window_extent(renderer)

        # Convert to axes fraction coordinates
        trans = ax.transAxes.inverted()
        icon_bbox_axes = trans.transform(icon_bbox_display)

        # Extract edges and center
        (x0, y0), (x1, y1) = icon_bbox_axes
        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2

        # Arrow size scaling based on axis size
        ax_width_px = ax.get_window_extent().width
        ax_height_px = ax.get_window_extent().height
        arrow_length = 0.1  # in axes fraction
        arrow_lw = max(1.5, 4 * ((x1 - x0) + (y1 - y0)))

        # Add arrows if requested
        if translation:
            if 'x' in translation:
                if 'left' in location:
                    start = (x1, y_center)
                    end = (x1 + arrow_length, y_center)
                else:
                    start = (x0, y_center)
                    end = (x0 - arrow_length, y_center)
                ax.annotate(
                    '', xy=end, xytext=start,
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='red', lw=arrow_lw)
                )
            if 'y' in translation:
                if 'bottom' in location:
                    start = (x_center, y1)
                    end = (x_center, y1 + arrow_length)
                else:
                    start = (x_center, y0)
                    end = (x_center, y0 - arrow_length)
                ax.annotate(
                    '', xy=end, xytext=start,
                    xycoords='axes fraction',
                    textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='red', lw=arrow_lw)
                )

            # Draw white background box behind icon
            if background:
                width = x1 - x0
                height = y1 - y0
                rect = Rectangle(
                    (x0, y0), width, height,
                    transform=ax.transAxes,
                    facecolor='white', edgecolor='none', alpha=0.75,
                    zorder=ab.zorder - 1  # behind the icon
                )
                ax.add_patch(rect)
        return x0, y0, x1, y1


def add_image(ax, image_path, location='top right', zoom=96/400, background=False, align_corner=(0, 0)):
    """
    Adds an image to a Matplotlib Axes object at a specified location with optional
    background settings.

    This function allows placing an image on a plot, with features such as location
    control, scaling, and background rendering to enhance visualization.

    :param ax: A Matplotlib Axes object where the image will be added.
    :param image_path: The path to the image file (must be a PNG file).
    :type image_path: str
    :param location: Either a string specifying the general location ('top right',
        'top left', 'bottom right', 'bottom left') or a tuple of coordinates in
        axes fraction (default is 'top right').
    :type location: str or tuple[float, float]
    :param zoom: Scaling factor for the image, the default is 96/400 for adjusting
        icon sizes.
    :type zoom: float
    :param background: If True, draws a semi-transparent white background behind
        the image to improve visibility (default is False).
    :type background: bool
    :param align_corner: Defines the alignment of the image corner relative to
        the position if `location` is provided as a tuple. Defaults to (0, 0).
    :type align_corner: tuple[float, float]
    :return: Image corner coordinates in axes fraction.
    """

    # Load the PNG image using PIL
    with Image.open(image_path) as image:
        # Wrap it for matplotlib
        im = OffsetImage(image, zoom=zoom)

        if isinstance(location, str):
            # Define location anchor and box alignment
            location_map = {
                'top right': ((0.98, 0.98), (1, 1)),
                'top left': ((0.02, 0.98), (0, 1)),
                'bottom left': ((0.02, 0.02), (0, 0)),
                'bottom right': ((0.98, 0.02), (1, 0)),
            }
            (xy, box_alignment) = location_map.get(location, ((1, 1), (1, 1)))
        else:
            xy = location
            box_alignment = align_corner

        # Create and add icon box
        ab = AnnotationBbox(
            im,
            xy=xy,
            xycoords='axes fraction',
            box_alignment=box_alignment,
            frameon=False,
            pad=0
        )
        ax.add_artist(ab)

        # Force rendering to get accurate icon position
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Get icon bounding box in display (pixel) coords
        icon_bbox_display = ab.get_window_extent(renderer)

        # Convert to axes fraction coordinates
        trans = ax.transAxes.inverted()
        icon_bbox_axes = trans.transform(icon_bbox_display)

        # Extract edges and center
        (x0, y0), (x1, y1) = icon_bbox_axes

        # Draw white background box behind icon
        if background:
            width = x1 - x0
            height = y1 - y0
            rect = Rectangle(
                (x0, y0), width, height,
                transform=ax.transAxes,
                facecolor='white', edgecolor='none', alpha=0.75,
                zorder=ab.zorder - 1  # behind the icon
            )
            ax.add_patch(rect)
        return x0, y0, x1, y1