from fe65p2.scan_base import ScanBase
import fe65p2.plotting as  plotting
import time
import fe65p2.analysis as analysis
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - [%(levelname)-8s] (%(threadName)-10s) %(message)s")

import numpy as np
import bitarray
import tables as tb
from bokeh.charts import output_file, show, vplot, hplot, save
from progressbar import ProgressBar
from basil.dut import Dut
import os


def analyze(h5_filename, trigger, clock1, clock2, measuredirection, ranges):
    get_bin = lambda x, n: format(x, 'b').zfill(n)
    ranges1 = ranges[1] + 1
    ranges0 = ranges[0] - 1
    attrs_tdc = {}
    with tb.open_file(h5_filename, 'r+') as in_file_h5:
        raw_data = in_file_h5.root.raw_data[:]
        meta_data = in_file_h5.root.meta_data[:]
        en_mask = in_file_h5.root.scan_results.en_mask[:]
        hit_noise = np.zeros(ranges[1] - ranges[0] + 2)
        new_meta = []
        new_meta_trigger = []
        if len(meta_data):
            for elements in meta_data:
                if elements[7] > 0:
                    if len(new_meta) < 1:
                        new_meta = elements
                    else:
                        new_meta = np.vstack((new_meta, elements))
                if elements[7] == 0:
                    if len(new_meta_trigger) < 1:
                        new_meta_trigger = elements
                    else:
                        new_meta_trigger = np.vstack((new_meta_trigger, elements))
            param = []
            noise = []
            TOT = []
            if len(new_meta):
                param = np.arange(ranges0, ranges1)
                for numbers in param:
                    noise_per_param = 0
                    TOT_per_param = 0
                    counter=0
                    for entries in new_meta:
                        if numbers == entries['scan_param_id']:
                            for raw_entries in raw_data[entries['index_start']:entries['index_stop']]:
                                if int(get_bin(int(raw_entries), 32)[1]) == 1:
                                    tdc_data = raw_entries & 0xFFF  # take last 12 bit
                                    #print int(get_bin(int(raw_entries), 32))," tdc_data: ",  tdc_data, " tdc_delay: ", tdc_delay
                                    noise_per_param = noise_per_param + 1
                                    TOT_per_param=TOT_per_param+tdc_data
                                    counter=counter+1
                                    # noise_per_param = noise_per_param + entries['data_length']
                    noise = np.append(noise, noise_per_param / 5)
                    if counter<1:
                        TOT = np.append(TOT, 0)
                        continue
                    TOT=np.append(TOT,TOT_per_param/counter * 1.5625)
            if trigger == 0:
                noise2 = np.zeros(ranges[1] - ranges[0] + 2)
            en_mask1 = np.array(en_mask, dtype=bool)
            pixels = np.count_nonzero(en_mask1)
        attrs_tdc['trigger'] = trigger
        attrs_tdc['clock1'] = clock1
        attrs_tdc['clock2'] = clock2
        attrs_tdc['direction'] = measuredirection
        attrs_tdc['pixels'] = pixels
    print param
    print noise
    print TOT
    # np.set_printoptions(threshold=np.inf)
    # print new_meta_trigger['scan_param_id']
    # test=new_meta_trigger['scan_param_id'][:,0]
    # params = np.bincount(test)
    # get_bin = lambda x, n: format(x, 'b').zfill(n)
    # test=raw_data[meta_data['index_start'][2717]:meta_data['index_stop'][2717]]#2717
    # for elements in test:
    # print get_bin(elements,32), (elements & 0b111100000000000000000) >> 17, (elements & 0b11111100000000000) >>11, (elements & 0b10000000000) >> 10, (elements & 0b11110000) >> 4, (elements & 0b1111)
    # print get_bin(elements,32)
    # print test.shape
    # test=raw_data[meta_data['index_start'][3078]:meta_data['index_stop'][3078]]#2717
    # for elements in test:
    # print get_bin(elements,32), (elements & 0b111100000000000000000) >> 17, (elements & 0b11111100000000000) >>11, (elements & 0b10000000000) >> 10, (elements & 0b11110000) >> 4, (elements & 0b1111)
    # print get_bin(elements,32)
    # print test.shape


if __name__ == "__main__":
    analyze('/media/mark/1TB/Scanresults/output_data/new_chips/chip1/clock_influence/high_to_low_1/trigger_0_clock1_0_clock2_0.h5',0, 0, 0, 0, [5,21])