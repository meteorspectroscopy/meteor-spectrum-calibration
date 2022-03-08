import math
import logging
import numpy as np
import PySimpleGUI as sg
import m_specfun as m_fun


def tau(lambda_nm, h=0.0, aod=0.1):
    """
    from: Hayes et Latham dans Ap. J, 197, 593 (1975)
    in http://www.astrosurf.com/aras/extinction/calcul.htm
    :param lambda_nm: wavelength [nm]
    :param h: height observatory [m]
    :param aod: Aerosol Optical Depth (AOD) tau = -ln(transmission(550nm)) = A*ln(10)
        Un air très sec de montage correspond à un AOD de 0,02.
        Dans un désert sec le AOD vaut 0.04.
        En France, le AOD est de 0,07 en hiver, de 0,21 en été, et en moyenne sur l'année de 0,13.
        Lorsque le temps est très chauds et orageux, le AOD peut atteindre 0,50.
    :return tau: optical depth for air mass 1 (zenith)
            = 0.921 * absorption in magnitudes for air mass 1 (zenith)
    """
    lm = lambda_nm/1000
    h_km = h/1000
    # absorbance measured at sea_level, air_mass 1, from Buil
    a_rayleigh = 9.4977e-3 / lm**4 * (0.23465 + 1.076e2/(146 - lm**-2) + 0.93161/(41 - lm**-2))**2
    tau_r = 0.4*math.log(10)*math.exp(-h_km/7.996) * a_rayleigh
    tau_oz = 0.0168 * math.exp(-15.0 * abs(lm - 0.59))
    tau_ae = aod * (lm/0.55)**-1.3
    tau_0 = tau_r + tau_oz + tau_ae  # for air mass 1 (AM1) zenith
    return tau_0


def transmission(lambda_nm, elevation_deg=90.0, h=0.0, aod=0.1):
    """
    in http://www.astrosurf.com/aras/extinction/calcul.htm
    :param lambda_nm: wavelength [nm]
    :param elevation_deg: elevation of star, meteor above horizon [°]
    :param h: height observatory [m]
    :param aod: Aerosol Optical Depth (AOD) tau = -ln(transmission(550nm)) = A*ln(10)
        Un air très sec de montage correspond à un AOD de 0,02.
        Dans un désert sec le AOD vaut 0.04.
        En France, le AOD est de 0,07 en hiver, de 0,21 en été, et en moyenne sur l'année de 0,13.
        Lorsque le temps est très chauds et orageux, le AOD peut atteindre 0,50.
    :return transmission: transmission for air mass(elevation)
    """
    hrad = math.pi/180*elevation_deg
    air_mass = 1.0/(math.sin(hrad + 0.025 * math.exp(-11.0 * math.sin(hrad))))
    trans = math.exp(-air_mass * tau(lambda_nm, h, aod))
    return trans


def extinction_tool(file, elevation_deg=90.0, h=0.0, aod=0.1, resp_flag=False, trans_flag=True):
    """
    in http://www.astrosurf.com/aras/extinction/calcul.htm
    :param file: spectrum with atmospheric extinction
    :param elevation_deg: elevation of star, meteor above horizon [°]
    :param h: height observatory [m]
    :param aod: Aerosol Optical Depth (AOD) tau = -ln(transmission(550nm)) = A*ln(10)
        Un air très sec de montage correspond à un AOD de 0,02.
        Dans un désert sec le AOD vaut 0.04.
        En France, le AOD est de 0,07 en hiver, de 0,21 en été, et en moyenne sur l'année de 0,13.
        Lorsque le temps est très chauds et orageux, le AOD peut atteindre 0,50.
    :param resp_flag: if True, correction is applied to response
    :param trans_flag: if True, transmission is plotted after return
    :return new_file: file with appendix '_AM0', elevation_deg, h, aod, info, resp_flag, trans_flag
    """
    # do not apply atmospheric correction twice, operate on the original file, strip appendix '_AM0'
    file = m_fun.m_join(file).replace('_AM0', '')
    layout = [[sg.Text('Input File'), sg.InputText(file, key='file', size=(50, 1)),
               sg.Button('Load File')],
              [sg.Frame('Atmospheric transmittance', [[sg.Text('Elevation [°]:'),
                                           sg.InputText(elevation_deg, size=(19, 1), key='elev_deg')],
                                          [sg.T('AOD'), sg.In(aod, size=(10, 1), key='AOD'),
                                           sg.T('Height Obs. [m]'), sg.In(h, size=(10, 1), key='height')]])],
                                          [sg.B('Apply'), sg.B('Cancel'),
                                           sg.Checkbox('Save as response', default=resp_flag, key='resp'),
                                           sg.Checkbox('Plot transmission', default=trans_flag, key='trans')]]
    window = sg.Window('Atmospheric transmission correction', layout, keep_on_top=True).Finalize()
    info = ''
    new_file = ''

    while True:
        event, values = window.read()
        if event in (None, 'Cancel'):
            window.close()
            return new_file, elevation_deg, h, aod, info, False, False

        if event == 'Load File':
            window.Minimize()
            file, info = m_fun.my_get_file(file, title='Load uncorrected spectrum',
                                           file_types=(('Spectrum Files', '*.dat'), ('ALL Files', '*.*'),),
                                           default_extension='*.dat')
            window['file'].update(file)
            window.Normal()

        if event == 'Apply':
            t = []
            l_corr = []
            i_corr = []
            file = values['file']
            resp_flag = values['resp']
            trans_flag = values['trans']
            try:
                l_ori, i_ori = np.loadtxt(file, unpack=True, ndmin=2)
                new_file = m_fun.change_extension(file, '_AM0.dat')
                if l_ori != []:
                    elevation_deg = float(values['elev_deg'])
                    h = float(values['height'])
                    aod = float(values['AOD'])
                    for k in range(0, len(l_ori)):
                        if 900 > l_ori[k] > 300:
                            l_corr.append(l_ori[k])
                            trans_air = transmission(l_ori[k], elevation_deg, h, aod)
                            t.append(trans_air)
                            i_corr.append(i_ori[k] / trans_air)
                    if resp_flag:
                        # normalize response to peak value
                        i_corr = i_corr / np.max(i_corr)
                    np.savetxt(new_file, np.transpose([l_corr, i_corr]), fmt='%8.3f %8.5f')
                    info = f'corrected spectrum {new_file} saved for elev. = {elevation_deg}°, h= {h}m, AOD= {aod}'
                    if trans_flag:
                        np.savetxt('transmission_atmos.dat', np.transpose([l_corr, t]), fmt='%8.3f %8.5f')
                        info += f'\ntransmission_atmos.dat saved for elev. = {elevation_deg}°, h= {h}m, AOD= {aod}'
                    logging.info(info)
                else:
                    sg.PopupError('no file or invalid file loaded', title='Input Error', keep_on_top=True)
                    file = ''
                    trans_flag = False
            except Exception as e:
                sg.PopupError(f'error with {file}\n{e}', title='Input Error', keep_on_top=True)
                trans_flag = False
            finally:
                window.close()
                return new_file, elevation_deg, h, aod, info, resp_flag, trans_flag

