from Plot_Methods.plot_standards import *


def signal_ndown(delta, alpha=1, transfer=1/2):
    return 1 + (transfer*(1+alpha)-1) * delta


def signal_n(delta, alpha=1, transfer=1/2):
    return alpha + (transfer - alpha) * delta


def signal_nup(delta, alpha=1, transfer=1/2):
    return transfer * alpha * delta


# First plot ---------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
delta = np.linspace(0, 3, 100)
c = sns.color_palette("tab10")
ax.plot(delta, signal_nup(delta), label='Diode n+1', c=c[1])
ax.plot(delta, signal_n(delta), label=r'Diode n (Edge $\alpha = 1$)', c=c[2])
ax.plot(delta, signal_ndown(delta), label='Diode n-1', c=c[3])
ax.plot(1, 1, ls='-', c='k', label='T=0.5')

Tx = 0.02
ax.plot(delta, signal_nup(delta, transfer=1/2 - Tx), ls='--', c=c[1])
ax.plot(delta, signal_n(delta, transfer=1/2 - Tx), ls='--', c=c[2])
ax.plot(delta, signal_ndown(delta, transfer=1/2 - Tx), ls='--', c=c[3])
ax.plot(1, 1, ls='--', c='k', label='T=0.48')

Tx = 0.05
ax.plot(delta, signal_nup(delta, transfer=1/2 - Tx), ls='-.', c=c[1])
ax.plot(delta, signal_n(delta, transfer=1/2 - Tx), ls='-.', c=c[2])
ax.plot(delta, signal_ndown(delta, transfer=1/2 - Tx), ls='-.', c=c[3])
ax.plot(1, 1, ls='-.', c='k', label='T=0.45')

Tx = 0.1
ax.plot(delta, signal_nup(delta, transfer=1/2 - Tx), ls=':', c=c[1])
ax.plot(delta, signal_n(delta, transfer=1/2 - Tx), ls=':', c=c[2])
ax.plot(delta, signal_ndown(delta, transfer=1/2 - Tx), ls=':', c=c[3])
ax.plot(1, 1, ls=':', c='k', label='T=0.4')

ax.axhline(0, c='grey', ls='-')
ax.legend()
ax.set_xlabel(r'Ratio $\delta = E/R$')
ax.set_ylabel(r'Signal ($R \cdot B$)')
format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/', '_alpha1_', legend=False)

# Second plot ---------------------------------------------------------------------------------------------------------
for delta in np.arange(0, 3, 0.25):
    fig, ax = plt.subplots()
    plot_size = (41 * cm / 3, fullsize_plot[1])
    fig.set_size_inches(plot_size)
    alpha = np.linspace(0, 1, 100)
    ax.set_ylim(-0.1, 1.1)
    c = sns.color_palette("tab10")
    ax.plot(alpha, signal_nup(delta, alpha=alpha), label='Diode n+1', c=c[1])
    ax.plot(alpha, signal_n(delta, alpha=alpha), label=r'Diode n (Edge)', c=c[2])
    ax.plot(alpha, signal_ndown(delta, alpha=alpha), label='Diode n-1', c=c[3])
    # ax.plot(1, 1, ls='-', c='k', label='T=0.5')

    '''
    Tx = 0.02
    ax.plot(alpha, signal_nup(delta, alpha=alpha, transfer=1/2 - Tx), ls='--', c=c[1])
    ax.plot(alpha, signal_n(delta, alpha=alpha, transfer=1/2 - Tx), ls='--', c=c[2])
    ax.plot(alpha, signal_ndown(delta, alpha=alpha, transfer=1/2 - Tx), ls='--', c=c[3])
    ax.plot(1, 1, ls='--', c='k', label='T=0.48')

    Tx = 0.05
    ax.plot(alpha, signal_nup(delta, alpha=alpha, transfer=1/2 - Tx), ls='-.', c=c[1])
    ax.plot(alpha, signal_n(delta, alpha=alpha, transfer=1/2 - Tx), ls='-.', c=c[2])
    ax.plot(alpha, signal_ndown(delta, alpha=alpha, transfer=1/2 - Tx), ls='-.', c=c[3])
    ax.plot(1, 1, ls='-.', c='k', label='T=0.45')

    Tx = 0.1
    ax.plot(alpha, signal_nup(delta, alpha=alpha, transfer=1/2 - Tx), ls=':', c=c[1])
    ax.plot(alpha, signal_n(delta, alpha=alpha, transfer=1/2 - Tx), ls=':', c=c[2])
    ax.plot(alpha, signal_ndown(delta, alpha=alpha, transfer=1/2 - Tx), ls=':', c=c[3])
    ax.plot(1, 1, ls=':', c='k', label='T=0.4')
    '''

    ax.axhline(0, c='grey', ls='-', zorder=-1)
    ax.text(*transform_axis_to_data_coordinates(ax, [0.04, 0.91]), r'$\delta = $'+str(delta), fontsize=15, c='k', bbox={'facecolor': 'white', 'alpha': 0.85})
    if delta == 0:
        ax.legend()
    ax.set_xlabel(r'Edge overlay factor $\alpha$')
    ax.set_ylabel(r'Signal ($R \cdot B$)')
    plt.pause(0.0001)
    format_save('/Users/nico_brosda/Desktop/Cyrce_Messungen.nosync/Results_19062024/ProtoEmission/', '_1delta'+str(delta)+'_', legend=False, plot_size=plot_size)
