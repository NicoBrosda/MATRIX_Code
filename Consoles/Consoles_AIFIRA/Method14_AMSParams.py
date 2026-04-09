import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from EvaluationSoftware.main import *
from EvaluationSoftware.readout_modules import ams_channel_assignment_readout


# --------------------------------------------------
# Global setup (mapping, Analyzer)
# --------------------------------------------------
mapping = Path('../../Files/mapping.xlsx')
mapping_df = pd.read_excel(mapping, header=1)
channel_assignment = [int(k[-3:]) - 1 for k in mapping_df['direction_2']]

readout = lambda x, y: ams_fast_avg(x, y, channel_assignment=channel_assignment)

A = Analyzer((1, 128), 0.5, 0.0, readout=readout)
A.scale = 'pico'


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def sanitize_name(name):
    return str(name).replace(' ', '_').replace('/', '_').replace('\\', '_')


def load_signals_once(analyzer, folder_path, files):
    folder_path = Path(folder_path)
    loaded = {}

    for file_name in files:
        data = analyzer.readout(folder_path / file_name, analyzer)
        loaded[file_name] = {
            'signal': np.asarray(data['signal'][0], dtype=float),
            'std': np.asarray(data['std'][0], dtype=float),
        }

    return loaded


def reduce_channels(signal, std, mode='mean', channel_index=None):
    if mode == 'max':
        ind = np.argmax(signal, axis=0)
        return signal[ind], std[ind]
    elif mode == 'min':
        ind = np.argmin(signal, axis=0)
        return signal[ind], std[ind]
    elif mode == 'mean':
        return np.mean(signal, axis=0), np.mean(std, axis=0)
    elif mode == 'channel':
        if channel_index is None:
            raise ValueError("channel_index must be provided when mode='channel'")
        return signal[channel_index], std[channel_index]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def build_channel_series(loaded_data, files, mode='mean', channel_index=None):
    cache_signal = []
    cache_std = []

    for file_name in files:
        signal = loaded_data[file_name]['signal']
        std = loaded_data[file_name]['std']

        reduced_signal, reduced_std = reduce_channels(
            signal, std, mode=mode, channel_index=channel_index
        )
        cache_signal.append(reduced_signal)
        cache_std.append(reduced_std)

    return np.asarray(cache_signal, dtype=float), np.asarray(cache_std, dtype=float)


def plot_signals(
    signal_array,
    params,
    x_values=None,
    std_array=None,
    results_path=None,
    save_name=None,
    std=False,
    title=None,
    legend=False,
    close_after_save=False,
    predict_func=None,
    predict_param_input=None,
    annotate_ratios=True,
    ratio_fmt_data="{:.2f}$\\,\\%$",
    ratio_fmt_pred="{:.2f}$\\,\\%$",
    base_fontsize=11,
    logy=False,
):
    fig, ax = plt.subplots()

    if x_values is None:
        x = np.arange(len(params), dtype=float)
    else:
        x = np.asarray(x_values, dtype=float)
        if len(x) != len(params):
            raise ValueError("x_values must have the same length as params")
    y = np.asarray(signal_array, dtype=float)

    # Main data with optional error bars
    if std_array is not None:
        yerr = np.asarray(std_array, dtype=float)
        ax.errorbar(
            x, y, yerr=yerr,
            linestyle='--',
            color='k',
            marker='x',
            markersize=8,
            capsize=4,
            elinewidth=1.2,
            label='Data'
        )
    else:
        ax.plot(
            x, y,
            linestyle='--',
            color='k',
            marker='x',
            markersize=8,
            label='Data'
        )

    # X axis: numeric positions with fixed labels
    ax.set_xticks(x)
    ax.xaxis.set_major_locator(ticker.FixedLocator(x))

    # only overwrite labels if they are really different from numeric values
    if params is not None:
        ax.set_xticklabels(params)

    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel('AMS parameter')
    ax.set_ylabel('Circuit response Std (LSB)' if std else 'Circuit response (LSB)')

    if title is not None:
        ax.set_title(title)

    # Prediction line
    y_pred = None
    if predict_func is not None:
        if predict_param_input is None:
            pred_params = x
        else:
            pred_params = np.asarray(predict_param_input, dtype=float)

        if len(pred_params) != len(x):
            raise ValueError("predict_param_input must have the same length as params")

        y_pred = np.asarray([predict_func(y[0], p) for p in pred_params], dtype=float)

        ax.plot(
            x, y_pred,
            linestyle='-',
            linewidth=1.5,
            color='tab:red',
            marker='o',
            markersize=5,
            alpha=0.9,
            label='Prediction'
        )

    # Ratio labels: stacked above data point (data above pred), with border-aware alignment
    if annotate_ratios and len(y) > 1 and y[0] != 0:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        xspan = x_max - x_min if x_max != x_min else 1.0
        yspan = y_max - y_min if y_max != y_min else 1.0

        line_gap_points = base_fontsize * 0.7
        line_gap_data = (line_gap_points / 72.0) * yspan

        for i in range(1, len(y)):
            ratio_data = y[i] / y[0] * 100.0
            txt_data = ratio_fmt_data.format(ratio_data)

            has_pred = (y_pred is not None and y_pred[0] != 0)
            if has_pred:
                ratio_p = y_pred[i] / y_pred[0] * 100.0
                txt_pred = ratio_fmt_pred.format(ratio_p)
            else:
                txt_pred = ""

            y_base = y[i]
            if std_array is not None:
                y_base = max(y[i] + std_array[i], y_pred[i])

            x_anch = x[i]
            y_anch_pred = y_base + 0.01 * yspan
            y_anch_data = y_anch_pred + line_gap_data * 0.5

            x_rel_pred = (x_anch - x_min) / xspan

            ha = 'center'
            x_shift = 0.0
            if x_rel_pred > 0.9:
                ha = 'right'
                x_shift = -0.01 * xspan
            elif x_rel_pred < 0.1:
                ha = 'left'
                x_shift = 0.01 * xspan

            x_anch += x_shift

            y_margin = 0.02 * yspan
            if y_anch_data > y_max - y_margin:
                delta = y_anch_data - (y_max - y_margin)
                y_anch_data -= delta
                y_anch_pred -= delta

            if has_pred:
                ax.text(
                    x_anch, y_anch_pred,
                    f"{txt_pred}",
                    ha=ha, va='bottom',
                    fontsize=base_fontsize,
                    color='tab:red',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8),
                    zorder=4,
                )

            ax.text(
                x_anch, y_anch_data,
                f"{txt_data}",
                ha=ha, va='bottom',
                fontsize=base_fontsize,
                color='k',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8),
                zorder=4,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if logy:
        ax.set_yscale('log')

    if legend and len(ax.get_legend_handles_labels()[1]) > 0:
        ax.legend()

    if results_path is not None:
        if save_name is None:
            save_name = title if title is not None else 'plot'
        if logy:
            just_save(results_path, save_name, save_format='.png', legend=legend, fig=fig, axes=[ax])

        else:
            format_save(results_path, save_name, save_format='.png', legend=legend, fig=fig, axes=[ax])

    if close_after_save:
        plt.close(fig)

    return fig, ax


def plot_summary_bundle(
    loaded_data,
    params,
    files,
    results_path,
    base_name='Signal',
    x_values=None,
    random_channels=None,
    random_channel_count=None,
    seed=None,
    use_std=False,
    close_after_save=False,
    predict_func=None,
    predict_param_input=None,
    annotate_ratios=True,
    logy=False,
):
    if random_channels is None:
        random_channels = []

    if random_channel_count is not None:
        n_channels = loaded_data[files[0]]['signal'].shape[0]
        rng = np.random.default_rng(seed)
        random_channels = rng.choice(n_channels, size=random_channel_count, replace=False).tolist()

    plot_specs = [
        ('max', None, f'{base_name}_Max'),
        ('min', None, f'{base_name}_Min'),
        ('mean', None, f'{base_name}_Mean'),
    ] + [('channel', ch, f'{base_name}_Channel_{ch}') for ch in random_channels]

    figures = []

    for mode, channel_index, plot_title in plot_specs:
        signals, stds = build_channel_series(
            loaded_data,
            files,
            mode=mode,
            channel_index=channel_index
        )

        y = stds if use_std else signals
        yerr = None if use_std else stds
        save_name = sanitize_name(plot_title)

        fig, ax = plot_signals(
            signal_array=y,
            params=params,
            x_values=x_values,
            std_array=yerr,
            results_path=results_path,
            save_name=save_name,
            std=use_std,
            title=plot_title,
            legend=(predict_func is not None),
            close_after_save=close_after_save,
            predict_func=predict_func,
            predict_param_input=predict_param_input,
            annotate_ratios=annotate_ratios,
            logy=logy,
        )
        figures.append((fig, ax))

    return figures


# --------------------------------------------------
# High-level entry point for one case
# --------------------------------------------------
def run_case(
    folder_path,
    results_path,
    params,
    files,
    base_name,
    predict_func,
    param_values=None,
    random_channels=None,
    random_channel_count=None,
    random_seed=42,
    plot_std_directly=False,
    annotate_ratios=True,
    close_after_save=False,
    logy=False,
):
    folder_path = Path(folder_path)
    results_path = Path(results_path)

    loaded_data = load_signals_once(A, folder_path, files)

    plot_summary_bundle(
        loaded_data=loaded_data,
        params=params,
        x_values=param_values,
        files=files,
        results_path=results_path,
        base_name=base_name,
        random_channels=random_channels,
        random_channel_count=random_channel_count,
        seed=random_seed,
        use_std=plot_std_directly,
        close_after_save=close_after_save,
        predict_func=predict_func,
        predict_param_input=param_values,
        annotate_ratios=annotate_ratios,
        logy=logy,
    )