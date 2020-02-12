"""
Divide the boundaries determined with the upstream conditions. Also constrict
the boundaries with how big difference of pre and post times (or distance).
"""

import numpy as np
import time
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import scipy.optimize as opt
import extra
import spiceypy as spice
from irfpy.vexpvat import vexspice as vspice
vspice.init()
spice.furnsh('/Volumes/VenusDataStorage/VExData/spice/kernels/lsk/NAIF0010.TLS')

# TODO: Take care using only one boundary for each orbit (they have crosscalibrated among them)
# TODO: Make some statistics on the difference between them for each orbit they have both made
# TODO: How do I choose which one?
# TODO: Make some random picks of boundaries and check if I agree with their determinations of the boundaries
# TODO: Make some plot of the average location of the boundary at terminator (X = 0.) and maybe X = -2.0 to check the size of the obstacle
# TODO: Make the energy spectra for the 10 upstream bins instead of 5 to calculate the coupling function with this.


def rms_fitting(data, parameter, error):
    initial_guess = [1.5, 1.01]
    # print('initial_guess: ', initial_guess)
    # print('data: ', data[error > 0.])
    # print('parameter: ', parameter[error > 0.])
    # print('error: ', error[error > 0.])
    # print(error_function(initial_guess, data[error > 0.], parameter[error > 0.], error[error > 0.]))
    fit_ret = opt.leastsq(error_function, initial_guess, args=(data[error > 0.], parameter[error > 0.], error[error > 0.]), full_output=1)
    return fit_ret[0]


def rms_fitting_icb(data, parameter, error, d):
    initial_guess = [-1]
    # print('initial_guess: ', initial_guess)
    # print('data: ', data.shape)
    # print('parameter: ', parameter.shape)
    # print('error: ', error.shape)
    # print(error_function_icb(initial_guess, data, parameter, error, d))
    fit_ret = opt.leastsq(error_function_icb, initial_guess, args=(data, parameter, error, d), full_output=1)
    return fit_ret[0]


def error_function(p, data, parameter, error):
        return (data - (p[0] + p[1] * parameter)) / error


def error_function_icb(p, data, parameter, error, d):
        return (data - (d + p * parameter)) / error


def linear_regression(x, m, k):
    return m + k * x


def plot_venus_fitted_bs_imb(ax, ICB_line, ICB_circle, BS, x_lims):
    circle = plt.Circle((0, 0), 1, color='k', fill=False, lw=2)
    ax.add_artist(circle)

    ax.plot(x_lims, ICB_line, '--k', lw=2, alpha=.2)
    ax.plot(BS[0, :], BS[1, :], 'k', lw=2, alpha=.2)
    ax.add_artist(ICB_circle)

    ax.plot(x_lims, -ICB_line, '--k', lw=2, alpha=.2)
    ax.plot(BS[0, :], -BS[1, :], 'k', lw=2, alpha=.2)
    ax.add_artist(ICB_circle)


def fit_bow_shock(users_locations):
    users = ['a', 'n', 'v']
    px = []
    py = []
    pz = []
    pr = []
    pdiff = []
    bd_name = 'bow shock'
    for user in users:
        for b_l in ['inbound', 'outbound']:
            px.extend(users_locations[user][bd_name][b_l]['px'])
            py.extend(users_locations[user][bd_name][b_l]['py'])
            pz.extend(users_locations[user][bd_name][b_l]['pz'])
            pr.extend(users_locations[user][bd_name][b_l]['pr'])
            pdiff.extend(users_locations[user][bd_name][b_l]['pdiff'])

    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)
    pr = np.array(pr)
    pdiff = np.array(pdiff)
    x0 = 0.788  # Use the one determined from Martinecz et al. (2008) to make it simpler
    r_fit = np.sqrt((px - x0)**2 + py**2 + pz**2)
    y_fit = 1 / r_fit
    costheta_fit = (px - x0) / r_fit
    error = pdiff  # distance between pre and post to be used as an error?
    par = rms_fitting(y_fit, costheta_fit, error)
    return par, error


def fit_ion_composition_boundary(users_locations):
    users = ['a', 'n', 'v']
    px = []
    py = []
    pz = []
    pr = []
    pdiff = []
    bd_name = 'ion composition boundary'
    for user in users:
        for b_l in ['inbound', 'outbound']:
            px.extend(users_locations[user][bd_name][b_l]['px'])
            py.extend(users_locations[user][bd_name][b_l]['py'])
            pz.extend(users_locations[user][bd_name][b_l]['pz'])
            pr.extend(users_locations[user][bd_name][b_l]['pr'])
            pdiff.extend(users_locations[user][bd_name][b_l]['pdiff'])

    px = np.array(px)
    py = np.array(py)
    pz = np.array(pz)
    pr = np.array(pr)
    pdiff = np.array(pdiff)

    # Find the average distance on the dayside for the fitted circle using weighted average
    r_data = np.sqrt(px**2 + py**2 + pz**2)[px > 0.]
    error = pdiff[px >= 0.]  # distance between pre and post to be used as an error?
    if len(error) < 1:
        r_fit = 1.109
    else:
        r_fit = np.average(r_data, weights=(1 / error))

    # Fit a straight line from the location of the circle end at X=0 and back
    exclude = (px < 0.) & (pdiff > 0.01)  # I don't trust the errors of less than 0.01. They make too big effect on the fitting scheme and I don't think we can determine to such a high degree. Therefore, remove these errors.
    error = pdiff[exclude]  # distance between pre and post to be used as an error?

    par = rms_fitting_icb(pr[exclude], px[exclude], error, r_fit)

    return par, r_fit, error


def main():
    # Setup variables
    users = ['a', 'n', 'v']
    users_colors = {'a': 'C0', 'n': 'C1', 'v': 'C2'}

    colors = {'highEUV': 'C3', 'lowEUV': 'k'}

    conditions = ['SWene', 'SWmom']
    # conditions = ['SWene']
    EUV_conditions = ['highEUV', 'lowEUV']
    fc = 1.67e-27 * .5 / 1.602e-19
    for condition in conditions:
        fig_radius, AX_radius = plt.subplots(2, 2, figsize=(12, 6))
        for EUV_condition in EUV_conditions:
            print(condition)
            tic0 = time.time()
            with open('processed_data/ssp13-conditionslimits.txt', 'r') as fx:
                for line in fx:
                    linedata = line.split()
                    if linedata[0] == condition:
                        cond_limits = [float(value) for value in linedata[1:]]
            Nr_cd_bins = len(cond_limits)

            # Make one plot for each condition range
            fig, AX = plt.subplots(2, Nr_cd_bins - 2, figsize=(24, 12))

            fig_fit, AX_fit = plt.subplots(2, Nr_cd_bins - 2, figsize=(24, 12))

            for idx in range(1, Nr_cd_bins - 1):
                limit = [cond_limits[idx], cond_limits[idx + 1]]
                print(idx, '/', Nr_cd_bins - 2, limit)
                orbits = extra.import_orbit_numbers(condition, limit, EUV_condition)
                limit = [cond_limits[idx] * fc, cond_limits[idx + 1] * fc]
                # print(orbits)

                users_locations = extra.import_boundary_locations(orbits, plim=0.5)

                # fit boundary locations
                par, error = fit_bow_shock(users_locations)
                x_range = np.arange(-1, 1, 0.01)
                y_fit = linear_regression(x_range, par[0], par[1])
                # Add fitted bow shock to all BS plots
                AX_fit[0, idx - 1].plot(x_range, y_fit, 'r--')

                # Convert fitted bow shock to "regular" units and add to plot
                l_fit = 1 / par[0]
                e_fit = par[1] * l_fit
                print('L: {:.4}, e: {:.4}'.format(l_fit, e_fit))
                xref = 0.788
                angle = np.arange(0, 180, 0.01) * np.pi / 180
                BS_radius = l_fit / (1 + e_fit * np.cos(angle))
                angle = angle[BS_radius > 0.]
                BS_radius = BS_radius[BS_radius > 0.]
                BS_x = BS_radius * np.cos(angle) + xref
                BS_y = BS_radius * np.sin(angle)
                AX[0, idx - 1].plot(BS_x, BS_y, 'r--', lw=2)
                AX[0, idx - 1].set_xlabel('L: {:.4}, e: {:.4}'.format(l_fit, e_fit))

                par, r_fit, error = fit_ion_composition_boundary(users_locations)
                print('k: {:.4}, d: {:.4}'.format(par[0], r_fit))
                x_range = np.arange(-6, 0., 0.01)
                y_fit = linear_regression(x_range, r_fit, par)
                AX[1, idx - 1].plot(x_range, y_fit, 'r--')
                IMB_circle = patch.Arc((0, 0), 2 * r_fit, 2 * r_fit, angle=0.0, theta1=-90.0, theta2=90.0, color='r', lw=2, ls='--')
                AX[1, idx - 1].add_artist(IMB_circle)
                AX[1, idx - 1].set_xlabel('k: {:.4}, d: {:.4}'.format(par[0], r_fit))

                # Plot the radial distance of the boundaries for X = 0 and X = -2.
                AX_radius[1, 0].plot(np.nanmean(limit), r_fit, 'o', color=colors[EUV_condition])  # IMB at -0.
                BS_radius_X0 = l_fit / (1 + e_fit * np.cos(np.deg2rad(90)))  # BS at -0.
                AX_radius[0, 0].plot(np.nanmean(limit), BS_radius_X0, 'o', color=colors[EUV_condition])
                r_fit_X2 = linear_regression(-2., r_fit, par)  # IMB at -2.
                AX_radius[1, 1].plot(np.nanmean(limit), r_fit_X2, 'o', color=colors[EUV_condition])
                idr = np.argmin(np.abs(BS_x + 2.))
                BS_radius_X2 = BS_y[idr]  # BS at -2.
                AX_radius[0, 1].plot(np.nanmean(limit), BS_radius_X2, 'o', color=colors[EUV_condition])

                for ibd, bd_name in enumerate(['bow shock', 'ion composition boundary']):
                    for user in users:
                        for b_l in ['inbound', 'outbound']:
                            px = users_locations[user][bd_name][b_l]['px']
                            py = users_locations[user][bd_name][b_l]['py']
                            pz = users_locations[user][bd_name][b_l]['pz']
                            pr = users_locations[user][bd_name][b_l]['pr']
                            pdiff = users_locations[user][bd_name][b_l]['pdiff']
                            AX[ibd, idx - 1].errorbar(px, pr, yerr=pdiff, fmt='.', c=users_colors[user], capsize=3, capthick=1, lw=1)
                            AX[ibd, idx - 1].plot(px, pr, '.', color=users_colors[user])
                            AX[ibd, idx - 1].set_title('{:.2} to {:.2}'.format(*limit))

                            # Convert to y = 1/r, x = cos(theta) coord. system
                            x0 = 0.788  # Use the one determined from Martinecz et al. (2008) to make it simpler
                            r_fit = np.sqrt((px - x0)**2 + py**2 + pz**2)
                            y_fit = 1 / r_fit
                            costheta_fit = (px - x0) / r_fit
                            AX_fit[ibd, idx - 1].errorbar(costheta_fit, y_fit, yerr=pdiff, fmt='.', c=users_colors[user], capsize=3, capthick=1, lw=1)
                            AX_fit[ibd, idx - 1].plot(costheta_fit, y_fit, '.', color=users_colors[user])
                            AX_fit[ibd, idx - 1].set_title('{:.2} to {:.2}'.format(*limit))

            for id, ax in enumerate(AX[0, :].ravel()):
                extra.plot_venus_bs_imb(ax)  # Add the theoretical boundaries from Martinecz et al., 2008, and the Venus disk
                ax.set_ylim(0., 6.)
                ax.set_xlim(-5., 2.)
                ax.set_aspect('equal')
            for id, ax in enumerate(AX[1, :].ravel()):
                extra.plot_venus_bs_imb(ax)  # Add the theoretical boundaries from Martinecz et al., 2008, and the Venus disk
                ax.set_ylim(0., 3.)
                ax.set_xlim(-3., 1.)
                ax.set_aspect('equal')

            AX_fit[0, 0].set_ylabel('Bow Shock\n(1/R) [Rv$^{-1}$]')
            AX_fit[1, 0].set_ylabel('IMB\n(1/R) [Rv$^{-1}$]')
            AX[0, 0].set_ylabel('Bow Shock')
            AX[1, 0].set_ylabel('IMB')

            filename = 'processed_data/ven02/ven02-boundary_locations_{}_{}.pdf'.format(EUV_condition, condition)
            fig.savefig(filename, bbox_inches='tight')

            filename_fit = 'processed_data/ven02/ven02-boundary_locations_fitcoords_{}_{}.pdf'.format(EUV_condition, condition)
            fig_fit.savefig(filename_fit, bbox_inches='tight')
            print('Plotted boundaries for {}, {} in {:.4} s.'.format(EUV_condition, condition, time.time() - tic0))

        AX_radius[0, 0].set_ylim(0.9, 1.7)
        AX_radius[0, 1].set_ylim(2.9, 3.7)
        AX_radius[1, 0].set_ylim(0.9, 1.3)
        AX_radius[1, 1].set_ylim(0.9, 1.3)
        for ax in AX_radius.ravel():
            ax.set_xscale('log')

        AX_radius[0, 0].set_ylabel('Bow Shock\nRadial distance [Rv]')
        AX_radius[1, 0].set_ylabel('IMB\nRadial distance [Rv]')
        AX_radius[0, 0].set_title('X = 0. Rv')
        AX_radius[0, 1].set_title('X = -2. Rv')
        AX_radius[1, 0].set_xlabel('SW energy flux')
        AX_radius[1, 1].set_xlabel('SW energy flux')
        AX_radius[1, 1].plot(1e15, 0, 'o', color=colors['highEUV'], label='high EUV')
        AX_radius[1, 1].plot(1e15, 0, 'o', color=colors['lowEUV'], label='low EUV')
        AX_radius[1, 1].legend()
        filename_radius = 'processed_data/ven02/ven02-boundary_radialdistance_{}.pdf'.format(condition)
        fig_radius.savefig(filename_radius, bbox_inches='tight')


if __name__ == '__main__':
    main()
