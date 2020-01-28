def BowShock_IMB(x0, x1, half=False):
    """Defines the IMB and Bow shock position for a given x range. Adapted from Martinecz et al. [2008]."""
    import matplotlib.patches as patch
    import numpy as np
    # IMB calculation
    x_lims = np.arange(x0 - 0.01, x1 + 0.01, 0.01)
    IMB = []
    for x in x_lims:
        k = -0.097
        d = 1.109
        if x <= 0.:
            r = k * x + d
        elif 0. < x <= d:
            # A circle with radius 1.109
            r = np.sqrt(d * d - x * x)
        else:
            r = 0.
        IMB.append(r)
    IMB = np.array(IMB)
    if not half:
        IMB_circle = patch.Arc((0, 0), 2 * 1.109, 2 * 1.109, angle=0.0, theta1=0.0, theta2=90.0, color='k', lw=2, ls='--', alpha=.2)
    else:
        IMB_circle = patch.Arc((0, 0), 2 * 1.109, 2 * 1.109, angle=0.0, theta1=-90.0, theta2=90.0, color='k', lw=2, ls='--', alpha=.2)

    # Bow Shock calculation
    l = 1.303
    e = 1.056
    xref = 0.788
    angle = np.arange(0, 180, 0.01) * np.pi / 180
    BS_radius = l / (1 + e * np.cos(angle))
    BS_x = BS_radius * np.cos(angle) + xref
    BS_y = BS_radius * np.sin(angle)
    BS = np.array([BS_x[BS_x < BS_radius + xref], BS_y[BS_x < BS_radius + xref]])

    return IMB, IMB_circle, BS, x_lims


def plot_venus_bs_imb(ax):
    import matplotlib.pyplot as plt
    circle = plt.Circle((0, 0), 1, color='k', fill=False, lw=2, alpha=.2)
    ax.add_artist(circle)

    ICB_line, ICB_circle, BS, x_lims = BowShock_IMB(-6, 0, half=True)
    ax.plot(x_lims, ICB_line, '--k', lw=2, alpha=.2)
    ax.plot(BS[0, :], BS[1, :], 'k', lw=2, alpha=.2)
    ax.add_artist(ICB_circle)

    ax.plot(x_lims, -ICB_line, '--k', lw=2, alpha=.2)
    ax.plot(BS[0, :], -BS[1, :], 'k', lw=2, alpha=.2)
    ax.add_artist(ICB_circle)


def plot_venus_bs_imb_yz(ax):
    import matplotlib.pyplot as plt
    # Plot the Venus surface
    circle_venus = plt.Circle((0, 0), 1, color='k', fill=False, lw=5)
    ax.add_artist(circle_venus)

    # Plot the Bow shock and IMB at the terminator
    IMB_circle = plt.Circle((0, 0), 1.109, color='k', ls='--', fill=False, lw=5)
    BS_circle = plt.Circle((0, 0), 2.091, color='g', ls='-', fill=False, lw=5)
    ax.add_artist(IMB_circle)
    ax.add_artist(BS_circle)


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


def import_dicts():
    import IO
    # Import user dictionaries
    # Alex
    a_region = IO.read_file(filepath='regions', filename='region_a')
    a_dict = a_region.get()

    # Nicolas
    n_region = IO.read_file(filepath='regions', filename='region_n')
    n_dict = n_region.get()

    # Viktor
    v_region = IO.read_file(filepath='regions', filename='region_v')
    v_dict = v_region.get()

    user_dicts = {'a': a_dict, 'n': n_dict, 'v': v_dict}

    return user_dicts


def import_boundary_locations(orbits, plim):
    import numpy as np
    user_dicts = import_dicts()
    users = ['a', 'n', 'v']

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
        for b_l in boundary_location:
            for user in users:
                pos_x = []
                pos_y = []
                pos_z = []
                pos_r = []
                pos_diff = []
                for orbit in orbits:
                    timesDt = []
                    for b_pp in boundary_prepost:
                        boundary = b_l + ' ' + b_pp + ' ' + boundary_name
                        try:
                            timesDt.append(user_dicts[user][orbit][boundary])
                        except KeyError:
                            continue
                    if len(timesDt) < 2:
                        continue
                    px, py, pz, pr, pdiff = pre_post_average_location(timesDt)
                    if pdiff < plim:
                        pos_x.append(px)
                        pos_y.append(py)
                        pos_z.append(pz)
                        pos_r.append(pr)
                        pos_diff.append(pdiff)
                users_locations[user][boundary_name][b_l] = {'px': np.array(pos_x), 'py': np.array(pos_y), 'pz': np.array(pos_z), 'pr': np.array(pos_r), 'pdiff': np.array(pos_diff)}

    return users_locations


def import_orbit_numbers(condition, limit, EUV_condition):
    """Import which orbits to use for this condition.
    """
    import h5py
    orbit_list = []
    filename = 'processed_data/ssp01-orbitnumberfile.hdf5'

    class ContinueLoop(Exception):
        pass
    continueloop = ContinueLoop()
    EUV_value = 0.007
    with h5py.File(filename, 'r') as fp:
        for key in fp.keys():
            try:
                # Import the relevant condition to use and check if within the limit
                data = fp[key][condition][:]
                EUV_data = fp[key]['EUV'][:]
                if EUV_condition == 'highEUV':
                    EUV_bool = EUV_data[0][1] > EUV_value
                elif EUV_condition == 'lowEUV':
                    EUV_bool = EUV_data[0][1] < EUV_value
                try:
                    for (_, value) in data:
                        if (limit[0] < value < limit[1]) & EUV_bool:
                            break
                    else:
                        raise continueloop
                except ContinueLoop:
                    continue

                # If non of the previous failed and continued the loop earlier,
                # then this orbit matches all conditions and should be used:
                orbit_list.append(int(key))
            except KeyError:
                # If all of these values are not there then don't use this orbit
                continue
    return orbit_list
