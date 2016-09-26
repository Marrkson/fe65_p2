import numpy as np
import yaml
import matplotlib.pyplot as plt
import tables as tb
import fe65p2.plotting as plotting
from fe65p2.scan_base import ScanBase
from glob import glob
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import fe65p2.analysis as analysis
import time
from matplotlib.backends.backend_pdf import PdfPages
from bokeh.charts import HeatMap, bins, output_file, vplot, hplot, show, save
from bokeh.palettes import RdYlGn6, RdYlGn9, BuPu9, Spectral11
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import logging
from scipy.optimize import curve_fit
from scipy.special import erf

def scurve(x, A, mu, sigma):
    return 0.5 * A * erf((x - mu) / (np.sqrt(2) * sigma)) + 0.5 * A

def fit_scurve(scurve_data, PlsrDAC):  # data of some pixels to fit, has to be global for the multiprocessing module
    index = np.argmax(np.diff(scurve_data))
    max_occ = np.median(scurve_data[index:])
    threshold = PlsrDAC[index]
    noise=100
    if abs(max_occ) <= 1e-08:  # or index == 0: occupancy is zero or close to zero
        popt = [0, 0, 0]
    else:
        try:
            popt, _ = curve_fit(scurve, PlsrDAC, scurve_data, p0=[max_occ, threshold, noise], check_finite=False) #0.01 vorher
            logging.info('Fit-params-scurve: %s %s %s ', str(popt[0]),str(popt[1]),str(popt[2]))
        except RuntimeError:  # fit failed
            popt = [0, 0, 0]
            logging.info('Fit did not work scurve: %s %s %s', str(popt[0]),
                         str(popt[1]), str(popt[2]))

    if popt[1] < 0:  # threshold < 0 rarely happens if fit does not work
        popt = [0, 0, 0]
    return popt


def scurveplot(h5_filename):
    with tb.open_file(h5_filename, 'r') as in_file_h5:
        tdc_data = in_file_h5.root.tdc_data[:]
    charge=()
    hits=()
    old=tdc_data[0][1]
    print old
    for i in tdc_data:
        if old != i[1]:
            break
        charge=np.append(charge,i[0])
        hits=np.append(hits,i[2])
        plot_range=np.arange(charge[0],charge[-1], (charge[-1]-charge[0])/1000)
    A,mu,sigma = fit_scurve(hits[:-3],charge[:-3])
    plt.plot(charge[:-3], hits[:-3], 'bo', Label= 'measured data')
    #plt.errorbar(charge[:-3], hits[:-3], fmt='bo', yerr=np.sqrt(hits[:-3]))
    plt.plot(plot_range, scurve(plot_range, A, mu, sigma), Label= ('fitted data A: ' + str(round(A,2)) + ', mu: ' + str(round(mu,2)) + ', sigma: ' + str(round(sigma,2))))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, mode="expand", borderaxespad=0.)
    plt.xlabel("Injection [e-]")
    plt.ylabel("Hits")
    plt.xlim(0,2000)
    plt.show()  # boom

def intime_thresh_sqrt(x, *parameters):
    tau,param1 = parameters
    return param1+tau/np.sqrt(x)

def fit_intime_thresh_sqrt(x_data,y_data,tau):
    params_guess=np.array([tau,y_data[-1]])
    params_from_fit = curve_fit(intime_thresh, x_data, y_data, p0=params_guess)
    return params_from_fit[0][0], params_from_fit[0][1]

def intime_thresh(x, *parameters):
    tau,thresh, param1 = parameters
    return -tau*np.log((1-thresh/4/x))+param1

def fit_intime_thresh(x_data,y_data,tau, thresh):
    params_guess=np.array([tau,thresh, y_data[-1]])
    params_from_fit = curve_fit(intime_thresh, x_data, y_data, p0=params_guess)
    return params_from_fit[0][0], params_from_fit[0][1], params_from_fit[0][2]

def plot_timewalk(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        try:
            tdc_data = in_file_h5.root.tdc_data[:]
            td_threshold=in_file_h5.root.td_threshold[:]
        except RuntimeError:
            logging.info('tdc_data not present in file')
            return
        param, index = np.unique(tdc_data['pixel_no'], return_index=True)
        x_data=tdc_data['charge'][index[0]:index[1]]
        y_data1=tdc_data['hits'][index[0]:index[1]]
        y_data2=tdc_data['delay_ns'][index[0]:index[1]]
        print x_data
        print y_data1
        sigma,mu,A=fit_scurve(y_data1, x_data)
        x_data = x_data[y_data2 >0]
        y_data2= y_data2[y_data2>0]
        tau,thresh,param1=fit_intime_thresh(x_data, y_data2, 100, mu)
        print tau,thresh,param1
        plt.plot(x_data,y_data2,'ro')
        plt.plot(x_data, intime_thresh(x_data,100,399,50))
        plt.xlabel('Charge (electrons)')
        plt.ylabel('Delay (ns)')
        plt.draw()
        plt.show()

def plot_timewalk2(h5_file_name):
    with tb.open_file(h5_file_name, 'r') as in_file_h5:
        try:
            tdc_data = in_file_h5.root.tdc_data[:]
            td_threshold=in_file_h5.root.td_threshold[:]
        except RuntimeError:
            logging.info('tdc_data not present in file')
            return
        #param, index = np.unique(tdc_data['pixel_no'], return_index=True)
        time_thresh = td_threshold['td_threshold']
        expfit0 = td_threshold['expfit0']
        expfit1 = td_threshold['expfit1']
        expfit2 = td_threshold['expfit2']
        expfit3 = td_threshold['expfit3']

        tot = tdc_data['tot_ns']
        tot_err = tdc_data['err_tot_ns']
        delay = tdc_data['delay_ns']
        delay_err = tdc_data['err_delay_ns']
        pixel_no = tdc_data['pixel_no']
        pulse = tdc_data['charge']
        hits = tdc_data['hits']
        pix, stop = np.unique(pixel_no, return_index=True)
        stop = np.sort(stop)
        print "pix: ", pix
        single_scan = figure(title="Single pixel scan ")
        single_scan.xaxis.axis_label="Charge (electrons)"
        single_scan.yaxis.axis_label="Hits"

        TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select"
        p1 = figure(title="In time threshold", tools=TOOLS)
        p1.xaxis.axis_label="Charge (electrons)"
        p1.yaxis.axis_label="Delay (ns)"
        p2 = figure(title="TOT linearity", tools=TOOLS)
        p2.xaxis.axis_label="Charge (electrons)"
        p2.yaxis.axis_label="TOT (ns)"

        stop = list(stop)
        stop.append(len(tot))
        for i in range(1):#len(stop)-1):
            s1 = int(stop[i])
            s2 = int(stop[i+1])
            if time_thresh[i]==0:
                continue
            single_scan.diamond(x=pulse[s1:s2], y=hits[s1:s2], size=5, color=Spectral11[i-1], line_width=2)
            A, mu, sigma = analysis.fit_scurve(hits[s1:s2], pulse[s1:s2], np.max(hits[s1:s2]))
            for values in range(s1,s2):
                if pulse[values] >=5/4*mu:
                    s1=values
                    break
            p1.circle(pulse[s1:s2], delay[s1:s2], legend=str("pixel " + str(pixel_no[s1])), color=Spectral11[i-1], size=8)
            if len(time_thresh)!=0:
                n=0
                delay_min=np.min(delay[s1:s2])
                for entries in pulse[s1:s2]:
                    print delay[s1+n]
                    if delay[s1+n]<=delay_min+25:
                        in_time_thresh=delay[s1+n]
                        break
                    n=n+1
                p1.asterisk(pulse[s1+n],in_time_thresh,color=Spectral11[i-1], size=20,legend="Time dependent Threshold: "+str(round(time_thresh[i],2)))
            else:
                logging.info('No fit possible only Data plotted')
            err_x1 = [(pulse[s], pulse[s]) for s in range(s1,s2)]
            err_y1 = [[float(delay[s]-delay_err[s]), float(delay[s]+delay_err[s])] for s in range(s1,s2)]
            p1.multi_line(err_x1, err_y1, color=Spectral11[i-1], line_width=2)

            p2.circle(pulse[s1:s2], tot[s1:s2], legend=str("pixel "+str(pix[i])), color=Spectral11[i-1], size = 8)
            err_x1 = [(pulse[s], pulse[s]) for s in range(s1,s2)]
            err_y1 = [[float(tot[s]-tot_err[s]), float(tot[s]+tot_err[s])] for s in range(s1,s2)]
            p2.multi_line(err_x1, err_y1, color=Spectral11[i-1], line_width=2)

        return p1, p2, single_scan


def single_timewalk_plot(h5_filename):
    output_file('/home/mark/Desktop/Stuff/single.html')
    p1,p2,single_scan = plot_timewalk2(h5_filename)
    save(hplot(vplot(p1, p2), single_scan))

def Heatmap():
    x=np.arange(0,64,1)
    y=np.arange(0,64,1)
    intensity=np.ones((64,64))
    #for i in range(0,8):
        #print i * 8,(i + 1) * 8
        #intensity[:,i * 8:(i + 1) * 8]=i/0.16
    #intensity[:, 2 * 8:(2 + 1) * 8] = 0
    plt.pcolormesh(x, y, intensity)
    plt.xlabel("Columns")
    plt.xlim(0,63)
    plt.ylim(0,63)
    plt.ylabel("Rows")
    cbar=plt.colorbar()  # need a colorbar to show the intensity scale
    cbar.set_label('Powerconsumption', rotation=270, labelpad=10)
    plt.clim(0, 1)
    plt.show()  # boom
    plt.close()


if __name__ == "__main__":
    #Heatmap()
    single_timewalk_plot('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/external/PrmpVbpDac/20160628_162052_noise_scan_PrmpVbpDac_02/160628_162052_noise_scan_PrmpVbpDac_02_076_threshold_timewalk.h5')
    #scurveplot('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/external/PrmpVbpDac/20160628_162052_noise_scan_PrmpVbpDac_02/160628_162052_noise_scan_PrmpVbpDac_02_076_threshold_timewalk.h5')

    #scurveplot('/media/mark/1TB/Scanresults/output_data/old_chips/chip2/external/vthin2Dacnew/20160803_000502_noise_scan_vthin2Dac_01/160803_000502_noise_scan_vthin2Dac_01_000_threshold_timewalk.h5')
    #folder=glob('/media/mark/1TB/Scanresults/output_data/old_chips/chip1/external/vthin2Dacnew/20160802_173639_noise_scan_vthin2Dac_00/*_timewalk.h5')
    #for files in folder:
    #    single_timewalk_plot(files)