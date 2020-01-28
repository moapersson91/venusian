"""
Import the determined boundary crossings and simply plot them.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from extra import plot_venus_bs_imb, import_dicts, pre_post_average_location
import spiceypy as spice
from irfpy.vexpvat import vexspice as vspice
vspice.init()
spice.furnsh('/Volumes/VenusDataStorage/VExData/spice/kernels/lsk/NAIF0010.TLS')


def plot_boundary(user_dicts, boundary_name):
    # Setup variables
    users = ['a', 'n', 'v']
    users_colors = {'a': 'C0', 'n': 'C1', 'v': 'C2'}
    users_names = {'a': 'Alexander', 'n': 'Nicolas', 'v': 'Viktor'}
    orbits = range(1, 3190)
    r_v = 6052  # km, Venus radii

    boundary_location = ['inbound', 'outbound']
    boundary_prepost = ['pre', 'post']

    fig, AX = plt.subplots(3, 1, figsize=(16, 10))
    users_orbits = {}
    for b_l in boundary_location:
        for user_id in range(3):
            user_orbits = 0
            pos_x = []
            pos_r = []
            pos_diff = []
            for orbit in orbits:
                timesDt = []
                for b_pp in boundary_prepost:
                    boundary = b_l + ' ' + b_pp + ' ' + boundary_name
                    try:
                        timesDt.append(user_dicts[user_id][orbit][boundary])
                    except KeyError:
                        continue
                if len(timesDt) < 2:
                    continue
                px, pr, pdiff = pre_post_average_location(timesDt)
                pos_x.append(px)
                pos_r.append(pr)
                pos_diff.append(pdiff)
                user_orbits += 1
            users_orbits[b_l + '_' + users[user_id]] = user_orbits
            users_orbits[b_l + '_' + users[user_id] + '_' + 'pdiff'] = np.array(pos_diff)
            AX[user_id].plot(pos_x, pos_r, '.', color=users_colors[users[user_id]])

    for idx, ax in enumerate(AX.ravel()):
        plot_venus_bs_imb(ax)  # Add the theoretical boundaries from Martinecz et al., 2008, and the Venus disk
        ax.set_ylim(0., 6.)
        ax.set_xlim(-4., 2.)
        ax.set_aspect('equal')
        ax.set_title(boundary_name + ' averages')
        ax.set_ylabel('{}\nVSO R [Rv]'.format(users_names[users[idx]]))
    AX[-1].set_xlabel('VSO X [Rv]')

    figname = 'processed_data/ven01/ven01-{}-positions.png'.format(boundary_name)
    fig.savefig(figname, bbox_inches='tight')

    for b_l in boundary_location:
        for user_id in range(3):
            print(b_l + '_' + users[user_id], users_orbits[b_l + '_' + users[user_id]])

    # Plot statistics of how much pre and post differs for each user
    fig, AX = plt.subplots(3, 1, figsize=(16, 10))
    for user_id in range(3):
        pdiff = []
        for b_l in boundary_location:
            pdiff.extend(users_orbits[b_l + '_' + users[user_id] + '_' + 'pdiff'] * r_v)
        AX[user_id].hist(pdiff, bins=40)
        AX[user_id].axvline(x=np.mean(pdiff), c='k', label='mean {:.4}'.format(np.mean(pdiff)))
        AX[user_id].axvline(x=np.median(pdiff), c='r', label='median {:.4}'.format(np.median(pdiff)))

    for idx, ax in enumerate(AX.ravel()):
        ax.set_title(boundary_name + ' ' + users_names[users[idx]])
        ax.set_ylabel('Frequency')
        ax.legend()
    AX[-1].set_xlabel('Difference in position [km]')

    figname = 'processed_data/ven01/ven01-{}-prepostdiff-histogram.png'.format(boundary_name)
    fig.savefig(figname, bbox_inches='tight')


def main():
    # import user dictionaries
    user_dicts = import_dicts()

    tic0 = time.time()
    print('Plotting Bow Shock...')
    plot_boundary(user_dicts, boundary_name='bow shock')
    print('Bow shock plotted in {:.4} s'.format(time.time() - tic0))

    tic0 = time.time()
    print('Plotting Ion composition boundary...')
    plot_boundary(user_dicts, boundary_name='ion composition boundary')
    print('Ion composition boundary plotted in {:.4} s'.format(time.time() - tic0))


if __name__ == '__main__':
    main()
