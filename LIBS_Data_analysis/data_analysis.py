import numpy as np
import pandas as pd
import os.path as os_path
from datetime import date
from scipy.signal import find_peaks
from scipy.signal import peak_widths


class data_analysis:
    """
    Description de la classe...
    """

    def __init__(self, spectra_path='', element_name='', unit_vector=False, level=1e-4, fwhm=1 / np.e ** 2, res=1e-1,
                 calibration_path=''):
        """
        spectra_path:
        element_name:
        unit_vector:
        level:
        fwhm:
        res:
        calibration_path:
        """

        data = pd.read_csv(spectra_path)
        key_1 = data.keys()[0]
        key_2 = data.keys()[1]
        self.waves = np.array(data[key_1])
        intensities = np.array(data[key_2])
        self.intensities = intensities / np.max(intensities)
        peaks = find_peaks(self.intensities, height=level)
        self.peaks_wavelgth = self.waves[np.array(peaks[0], dtype=int)]  # lambda_p
        self.peaks_values = peaks[1]['peak_heights']  # Ap
        results_half = peak_widths(self.intensities, peaks[0], rel_height=fwhm)
        self.peaks_widths = results_half[0]  # sigma
        self.classes = []  # liste des noms des classes(penser à enlever le underscore pour les classes qui se répètent)

        self.element_name = element_name
        self.calib_dir = calibration_path
        self.res = res

        print("Lambda des peaks = ", self.peaks_wavelgth)
        print("Amplitude des peaks = ", self.peaks_values)
        print("la largeur à mis hauteur des peaks: ", self.peaks_widths)

        if unit_vector:
            self.file_path = ""
            if os_path.isfile(calibration_path):
                self.file_path = calibration_path
                self.save_unit_vector(self.file_path)
            elif os_path.isdir(calibration_path):
                todays = date.today().strftime("%b-%d-%Y")
                self.file_path = os_path.join(calibration_path, 'LIBS_Calibration_' + todays + '.csv')
                os.path.normpath(self.file_path)
                print("New calibration file: ", self.file_path)
                self.save_unit_vector(self.file_path)
            else:
                print("Calibration directory path ERROR: please give a correct path")

        else:
            self.u_vec, self.svd, self.vh_vec, self.y_cal, self.elmt, self.projection, self.classes = \
                self.new_data_analysis(self.calib_dir)
            # self.projection = self.new_data_analysis(self.calib_dir)

    def delta_peaks(self):
        """
        Function to get the distance from one peak to the closest one
        :return: numpy array (Delta of all peaks detected in actual signal)
        """
        a = np.diff(self.peaks_wavelgth, append=self.peaks_wavelgth[-2])
        b = np.flip(np.diff(np.flip(self.peaks_wavelgth), append=self.peaks_wavelgth[1]))
        sigma_peak = np.minimum(np.abs(a), np.abs(b))

        return sigma_peak

    def importance(self):
        """
        Importance of a peak p is: Ap / Sum(all peak amplitudes)
        :return: numpy array (importance of all peaks detected in actual signal)
        """
        importance = - np.log10(1 - self.peaks_values / np.sum(self.peaks_values))

        return importance

    def selectivity(self):
        """
        Selectivity of a peak is: -Log[ peak_width / delta_peak ]
        :return: numpy array (selectivity of all peaks detected in actual signal)
        """
        # My calculation
        sel = 20 * np.log10(1 + self.delta_peaks() / self.peaks_widths)

        # Calculation as the article of Amato et al.
        # sel_list = list()
        # K = 10000
        # for lbd in self.peaks_wavelgth:
        #     sigma = lbd / K
        #     n = np.size(np.nonzero(abs(lbd-self.peaks_wavelgth)<sigma))
        #     sel_list.append( np.log10(np.size(self.peaks_wavelgth) / n))
        #
        # sel = np.array(sel_list)

        return sel

    def save_unit_vector(self, file_pth):
        """
        Save data if the actual signal is a calibration signal
        :return: bool (1: done, 0: Fails)
        """
        # By default, we supposed the calibration file exists, so No headers to be written and accessing mode is
        # appended
        new_file = False
        m = "w+"
        try:
            data = pd.read_csv(file_pth)
        except FileNotFoundError:
            data = pd.DataFrame()
            # If calibration file doesn't exist, then write headers and mode accessing is Write
            new_file = True
            m = "w"
        finally:
            imp = self.importance()
            sel = self.selectivity()
            # s = pd.Series([self.peaks_wavelgth, imp, sel, imp*sel],
            #               index=['Peaks', 'Importance', 'Selectivity', 'Weight'])
            if new_file:
                s = pd.Series(
                    {'Peaks': self.peaks_wavelgth, 'Importance': imp, 'Selectivity': sel, 'Weight': imp * sel},
                    dtype=float)
            else:
                s = pd.Series([self.peaks_wavelgth, imp, sel, imp * sel], dtype=float)

            data[self.element_name] = s
            data.to_csv(file_pth, mode=m, header=True, index=new_file)

    def new_data_analysis(self, calib_path):
        try:
            calib = pd.read_csv(calib_path, index_col=0)
            projection = {}
        except FileNotFoundError:
            print('Please give a correct calibration file path')
        else:
            waves = self.peaks_wavelgth
            weight = self.importance() * self.selectivity()
            # weights = weights / np.sqrt(np.sum(weights**2))
            m = np.zeros((waves.size, calib.columns.size))  # m a le même nombre de lignes que les longueurs d'ondes qui
            # nous interessent et le même nombre de colonnes que le nombre d'éléments chimiques dans la base
            ind = 0
            for element in calib.columns:
                x_cal = np.array(calib.loc['Peaks', element].strip('()').split(','), dtype=float)
                y_cal = np.array(calib.loc['Weight', element].strip('()').split(','), dtype=float)
                # y_cal = y_cal / np.sqrt(np.sum(y_cal**2))

                # sum = 0
                for j in x_cal:
                    # sum = sum + np.sum(weights[np.where(abs(waves - i) < self.res)] * y_cal[np.where(abs(x_cal - i)
                    # < self.res)]) m[np.where(waves == i), ind] = y_cal[np.where(x_cal == i)]
                    m[np.nonzero(abs(waves - j) < self.res), ind] = y_cal[np.nonzero(x_cal == j)]

                # projection[element] = sum # / np.sum(weights ** 2)
                ind = ind + 1

            # centered_m = m - np.mean(m, axis=0)
            u, s, vh = np.linalg.svd(m, full_matrices=False)
            x_u = np.dot(weight, u)
            s_s = np.where(abs(s) > 1e-6, s, 0)
            s_t = np.diag(np.where(s_s == 0.0, s_s, 1 / s_s))
            # s_t = np.diag(np.where(s == 0, s, 1 / s))
            x_us = np.dot(x_u, s_t)
            proj = np.dot(x_us, vh)
            proj = proj / np.sum(abs(proj))

            # class_list = [e.split("_")[0] for e in calib.columns]
            # class_list = list(set(class_list))
            ind = 0
            for element in calib.columns:
                prefix = element.split("_")[0]
                if not (prefix in projection.keys()):
                    projection[prefix] = proj[ind]
                else:
                    projection[prefix] = projection[prefix] + proj[ind]

                ind = ind + 1

        return u, s, vh, m, weight, projection, list(calib.columns)
        # return projection


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    # O for LIBS_calibration, 1 for LIBS_measurement
    dir_for_calib = './LIBS_Data_analysis/NIST_data/Unit_vectors_spectra/Mg_NIST/'

    # ### Add New unit vector LIBS Data to the calibration file ####
    # case = 0
    # calibration_file = './LIBS_Data_analysis/Calibration_files/LIBS_Calibration_Jan-18-2025.csv'

    # ## For new LIBS data measurement
    case = 1
    raw_data = './LIBS_Data_analysis/NIST_data/Alloys_spectra/AlSi_80-20.txt'
    calibration_file = './LIBS_Data_analysis/Calibration_files/LIBS_Calibration_Jan-18-2025.csv'

    # Default parameters for peak identification
    detect_level = 1e-3
    width_level = 1 / (np.e ** 2)
    spectro_res = 1e-4

    if case == 1:
        # ########## Make LIBS measurement
        libs = data_analysis(raw_data, 'Aluminium', unit_vector=False, level=detect_level, fwhm=width_level,
                             res=spectro_res,
                             calibration_path=calibration_file)

        sorted_projection = sorted(libs.projection.items(), key=lambda x: x[1], reverse=True)
        sorted_projection = dict(sorted_projection)
        fig, ax = plt.subplots()
        ax.bar(sorted_projection.keys(), sorted_projection.values())
        for i, v in enumerate(sorted_projection.values()):
            ax.text(i, v, str(round(v, 3)), fontweight='bold', rotation=45)
        plt.show()

        fig1, ax1 = plt.subplots()
        ax1.plot(libs.waves, libs.intensities, color='C0', label='NIST Spectra')
        weights = libs.importance() * libs.selectivity()
        ax2 = ax1.twinx()
        ax2.bar(libs.peaks_wavelgth, weights, align='edge', bottom=1, color='C1', label='Weights')
        ax1.set_ylabel('Libs Spectra (AU)', color='C0', fontsize=16)
        ax1.tick_params(axis='y', color='C0', labelcolor='C0')

        ax2.set_ylabel('Weights', color='C1', fontsize=16)
        ax2.tick_params(axis='y', color='C1', labelcolor='C1')
        ax2.spines['right'].set_color('C1')
        ax2.spines['left'].set_color('C0')

        ax1.set_xlabel('Wavelength (nm)', fontsize=16)
        ax1.legend(prop={"size": 16})

        plt.show()

    elif case == 0:
        # ############## Add new calibration vector
        files = os.listdir(dir_for_calib)
        files = [f for f in files if os.path.isfile(dir_for_calib + '/' + f)]
        for idx, f in enumerate(files):

            name = f.split(sep='_')[0] + '_' + str(idx)
            d = dir_for_calib + f
            libs = data_analysis(d, name, unit_vector=True, level=detect_level, fwhm=width_level, res=spectro_res,
                                 calibration_path=calibration_file)
            calibration_file = libs.file_path
