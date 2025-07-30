from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout
from Consoles.Consoles8Gafchromic.Concept8GafMeasurementComparison import GafImage


def load_image(folder_path, image, background_subtraction=True, normalization=True, position=None):
    # Automatic assigning of Analyzer, background, normalization
    if image[-4:] == '.bmp':
        OriginalGaf = GafImage(folder_path / image)
        OriginalGaf.load_image()
        # '''
        if 'matrix211024_006.bmp' in image:
            print('yes')
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
        pass
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