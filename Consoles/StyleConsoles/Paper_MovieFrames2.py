from EvaluationSoftware.movie_modules import *


# ---------------------------------------------------------------------------------------------------------------------
folder_path = Path('/Users/nico_brosda/Cyrce_Messungen/matrix_221024/')
analyzer = quick_movie_wrap(folder_path)
analyzer.scale = 'nano'
crit = '2DLarge_MovieBeamChanges2_'
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", "black", "red", "yellow"])
output_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/ResultsMovies/MatrixArray/')
plot_size = (18*cm, 9.3/1.2419*cm)
save_format = '.png'
dpi = 300
results_path = Path(f'/Users/nico_brosda/Cyrce_Messungen/Style/Paper/Frames{crit}/')

frame_select = [883, 2*4, 3]
fbunch = 500
# frame_select = None
zero_frame = 860
graph_multipanel_frames(analyzer, folder_path, crit, frame_select[0], frame_select[1], frame_select[2],
                        fbunch, zero_frame=zero_frame, plot_size=plot_size, save_format=save_format,
                        output_path=results_path, dpi=dpi, title_text=r'\textbf{(a)} s timescale - average beamspot subtracted',
                        intensity_limits=None)


frame_select = [869, 2*4, 5e-3]
fbunch = 1
# frame_select = None
zero_frame = None
graph_multipanel_frames(analyzer, folder_path, crit, frame_select[0], frame_select[1], frame_select[2],
                        fbunch, zero_frame=zero_frame, plot_size=plot_size, save_format=save_format, info=False,
                        output_path=results_path, dpi=dpi, title_text=r'\textbf{(b)} ms timescale - beam spot development',
                        intensity_limits=[0, 4.1*4.31])
# '''

