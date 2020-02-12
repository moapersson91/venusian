"""
Extract the determined boundaries but compare the boundaries determined for
the same orbits between the different users. How much do they differ and why?
"""

import numpy as np
import datetime
import time
import sys
import matplotlib.pyplot as plt
import extra
import spiceypy as spice
from irfpy.vexpvat import vexspice as vspice
vspice.init()
spice.furnsh('/Volumes/VenusDataStorage/VExData/spice/kernels/lsk/NAIF0010.TLS')


def pre_post_average_location(timesDt):
    import numpy as np
    from irfpy.vexpvat import vexspice as vspice
    r_v = 6052  # km, Venus radii
    position_vex = []
    for timeDt in timesDt:
        posvel = vspice.get_posvel(timeDt, target='VEX', origin='VENUS', frame='VSO')
        position_vex.append(posvel[:3] / r_v)  # Venus radii unit
    position_vex = np.array(position_vex)
    pos_diff = np.abs(np.sqrt(np.sum(position_vex[0, :]**2)) - np.sqrt(np.sum(position_vex[1, :]**2)))
    pos_x = np.nanmean(position_vex[:, 0])
    pos_y = np.nanmean(position_vex[:, 1])
    pos_z = np.nanmean(position_vex[:, 2])
    pos_r = np.sqrt(pos_y**2 + pos_z**2)
    return pos_x, pos_y, pos_z, pos_r, pos_diff


def main():
    user_dicts = extra.import_dicts()
    users = ['a', 'n', 'v']
    orbits = np.arange(0, 3190)
    plim = 0.5
    boundary_location = ['inbound', 'outbound']
    boundary_prepost = ['pre', 'post']
    boundaries = ['bow shock', 'ion composition boundary']
    # Pre-define dictionary
    users_locations = {}
    for user in users:
        users_locations[user] = {}
        for boundary_name in boundaries:
            users_locations[user][boundary_name] = {}
            for b_l in boundary_location:
                users_locations[user][boundary_name][b_l] = {}

    # Loop over data to import locations
    for boundary_name in boundaries:
        position_difference_a_n = []
        position_difference_a_v = []
        position_difference_n_v = []

        for b_l in boundary_location:
            for orbit in orbits:
                print('Orbit number: ', orbit)
                pos_x = []
                pos_y = []
                pos_z = []
                pos_r = []
                pos_diff = []
                user_id = []
                for user in users:
                    timesDt = []
                    for b_pp in boundary_prepost:
                        boundary = b_l + ' ' + b_pp + ' ' + boundary_name
                        try:
                            timesDt.append(user_dicts[user][orbit][boundary])
                        except KeyError:
                            continue
                    if len(timesDt) < 2:
                        continue
                    px, py, pz, pr, pdiff = extra.pre_post_average_location(timesDt)
                    if pdiff < plim:
                        pos_x.append(px)
                        pos_y.append(py)
                        pos_z.append(pz)
                        pos_r.append(pr)
                        pos_diff.append(pdiff)
                        user_id.append(user)

                if len(user_id) == 2:
                    if ('a' in user_id) & ('n' in user_id):
                        position_difference_a_n.append(np.array([[pos_x[0] - pos_x[1]], [pos_y[0] - pos_y[1]], [pos_z[0] - pos_z[1]], [pos_r[0] - pos_r[1]]]))

                    elif ('a' in user_id) & ('v' in user_id):
                        position_difference_a_v.append(np.array([[pos_x[0] - pos_x[1]], [pos_y[0] - pos_y[1]], [pos_z[0] - pos_z[1]], [pos_r[0] - pos_r[1]]]))

                    elif ('n' in user_id) & ('v' in user_id):
                        position_difference_n_v.append(np.array([[pos_x[0] - pos_x[1]], [pos_y[0] - pos_y[1]], [pos_z[0] - pos_z[1]], [pos_r[0] - pos_r[1]]]))

                try:
                    print(np.array(position_difference_a_n).shape)
                    print(np.array(position_difference_a_v)[:, :-1, 0].shape)
                    print(np.array(position_difference_a_v).shape)
                    print(np.array(position_difference_n_v).shape)
                except IndexError:
                    continue

        position_difference_a_n = np.array(position_difference_a_n)
        position_difference_a_v = np.array(position_difference_a_v)
        position_difference_n_v = np.array(position_difference_n_v)
        # Compare the plotted locations and values of the users
        fig, ax = plt.subplots()
        r_diff = np.sqrt(np.sum(position_difference_a_n[:, :-1, 0]**2, axis=1))
        ax.hist(r_diff, bins=40, color='k')
        ax.set_yscale('log')
        fig.savefig('processed_data/ven03/ven03-r-diff-a-n.png', bbox_inches='tight')

        fig, ax = plt.subplots()
        r_diff = np.sqrt(np.sum(position_difference_a_v[:, :-1, 0]**2, axis=1))
        ax.hist(r_diff, bins=40, color='k')
        ax.set_yscale('log')
        fig.savefig('processed_data/ven03/ven03-r-diff-a-v.png', bbox_inches='tight')

        fig, ax = plt.subplots()
        r_diff = np.sqrt(np.sum(position_difference_n_v[:, :-1, 0]**2, axis=1))
        ax.hist(r_diff, bins=40, color='k')
        ax.set_yscale('log')
        fig.savefig('processed_data/ven03/ven03-r-diff-n-v.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
