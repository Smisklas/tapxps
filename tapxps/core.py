import numpy as np
from datetime import datetime, timedelta
import re
from lmfit import Model
from lmfit.models import ExponentialGaussianModel, LinearModel
from scipy.integrate import simps
import glob

class TAPXPS_set():
    """
    Class to handle the tapxps data. At initialisation, the class loads a combined tapxps data file
    and separates it into different pulses based on the separation in timestamps. The offset and
    start time for each pulse is sotored in a list of dictionaries.
    """
    def __init__(self, file):

        self.pulses = []
        self.splits = []
        self.headers = []

        self.loader(file)

        
    def find_splits(self, UTC, split_level = 2):
        """
        Finds the splits in the data based on the difference in the timestamps
        as found from UTC and determined by split_level.
        
        """
        
        prevsplit = 0
        for index, tapxps in enumerate(UTC):
            if index == 0:
                pass
            else:
                diff = UTC[index] - UTC[index-1]
                if diff.total_seconds() > split_level:
                    self.splits.append((prevsplit,index-1))
                    prevsplit = index
                if index == len(UTC)-1:
                    self.splits.append((prevsplit, index))


    def make_pulses(self):
        """
        Generates the pulses from the data between the splits.
        """
        
        for split in self.splits:
            start, end = split
            ticks = [self.data[j].ticks for j in range(start, end+1)]
            UTC = self.data[start].UTC
            offset = self.tick_to_offset(ticks)
            start_tick = self.data[start].ticks
            spectrum = np.array([sum(self.data[j].spectrum) for j in range(start, end+1)])
            #image = np.array([self.data[j].spectrum for j in range(start, end+1)])
            self.pulses.append(TAPXPS_pulse(offset,self.data[start:end+1] , UTC, start_tick, spectrum))

        
    def parse_chunk(self, chunk):
        """
        Parses the information found during loading into spectra.
        """
        ticks, *xy_data = chunk.strip().strip('\n').split('\n')
        #im = np.zeros((1024, 1024))
        spectrum = np.zeros(1024)
        for line in xy_data:
            x, y, time = line.split()
            #im[int(x)][int(y)] += 1
            spectrum[int(x)] += 1
        return int(ticks), spectrum#, im

    
    def loader(self, filename):
        """
        Loads the TAPXPS data from the file filename into a tapxps object.
        """
        try:
            with open(filename, 'r') as f:
                file = f.read()
        # Define re patterns.
            date_p = re.compile(r'New exposure at:\s+(.*)\n', re.MULTILINE)
            data_p = re.compile(r'(?<=ticks:)[\d\.\s]+(?=[^\S\n]*)', re.DOTALL)
        # Read data from patterns
            dates = date_p.findall(file)
            data_chunk = data_p.findall(file)
            
            UTC = [datetime.strptime(date, "%a %b %d %H:%M:%S %Y") for date in dates]# timestamps into datetimeobjects
            self.find_splits(UTC)

            # the data and ticks are in chunks, which should have the same indexing as the splits.
            # Thus the splits can be used to separate the chunks into pulses before parsing into spectra

            for split in self.splits:
                header = {}
                start, end = split
                chunks_to_parse = data_chunk[start:end]
                data = np.zeros((end-start, 1024))
                ticks = []
                for index, chunk in enumerate(chunks_to_parse):
                    tick, spectrum = self.parse_chunk(chunk)
                    data[index] = spectrum
                    ticks.append(tick)
                    
                offset= self.tick_to_offset(ticks)
                self.pulses.append(data)
                header['UTC'] = (UTC[start], UTC[end])
                header['offset']=offset
                self.headers.append(header)
                
                
                # UTC = datetime.strptime(date, "%a %b %d %H:%M:%S %Y")
                #self.data.append(TAPXPS_spectrum(UTC, ticks, spectrum))
        except MemoryError:
            print(len(self.data))


    def tick_to_offset(self, ticks):
        """
        Parses the ticks from the tapxps data into offsets.
        """
        offset = []
        for index in range(len(ticks)):
        
            td = timedelta(microseconds = ticks[index]/10)-timedelta(microseconds = ticks[0]/10)
            offset.append(td.total_seconds())
        return np.array(offset)*3.65


class qms_set():
    """
    Class that holds a set of QMS data which can contain multiple trends
    """

    def __init__(self, filename):
        self.header, self.date, self.offset, self.data = self.qms_loader(filename)

    def qms_loader(self, filename, skip=2):
        """
        Loads the qms data located in the file specified by filename. The
        skip parameter tells the load method the number of lines to skip in the
        begining of the file
        """
        date_list = []
        time_list = []
        offset_list = []
        data_list = []
        header = []
        counter  = 0
        with open(filename) as f:
            for line in f:
                if counter == 0:
                    header = line.strip('\n').split(';')
                counter += 1
                if counter > skip:
                    date, time, offset, *data = line.split(';')
                    date_list.append(date.strip())
                    time_list.append(time.strip())
                    offset_list.append(float(offset.replace(',','.')))
                    for index in range(len(data)):
                        if counter == skip+1:
                            data_list.append([float(data[index].replace(',','.').strip('\n'))])
                        else:
                            data_list[index].append(float(data[index].replace(',','.').strip('\n')))
               

    
        date_times = [datetime.strptime(' '.join([date_list[j], time_list[j]]),
                                    '%Y-%m-%d %H:%M:%S.%f') for j in range(len(date_list))]

        qms_data = []
        qms_data.append(header)
        qms_data.append(date_times)
        qms_data.append(np.array(offset_list))
        qms_data.append(np.array(data_list))

        return qms_data

    
    def find_qms_splits(self, tapxps_set):
        """
        Uses the timestamps from the tapxps object to divide the data into pulses.
        """

        # split times from the tapxps_set
        # match split times with the closest qms times to find split index
        # make a list of np arrays in which the 

        self.pulses = []
        self.splits = []
        self.pulse_headers = []
        for header in tapxps_set.headers:
            start, end  = header['UTC']
            self.splits.append((self.get_time_index(start, self.date),
                                self.get_time_index(end, self.date)))

            #split the pulses and store them in separated files things.
        for split in self.splits:
            start, end = split
            self.pulses.append(np.array([self.data[j][start:end] for j in range(len(self.data))]))
            self.pulse_headers.append({'date': (self.date[start], self.date[end]), 'offset': self.offset[start:end]-self.offset[start]})
 

    def get_time_index(self, time, time_array):

        """
        Method used to correlate the qms and xps data.
        """
        deltas = []
        for point in time_array:
            td = point - time
            deltas.append(abs(int(td.total_seconds())))
        #deltas = np.array(deltas)
        minimum = min(deltas)
    
        return deltas.index(minimum)

    
    def pulse_areas(self):

        """
        Obtains the area of a pulse by fitting an ExponentialGaussian
        and with a linear background to the data.
        """
        shape = ExponentialGaussianModel()
        line = LinearModel()
        model = shape + line

        result = []

        for index, pulse in enumerate(self.pulses):
            if len(pulse[0]) < 10:
                pass
            else:
                x = self.pulse_headers[index]['offset']
                pars = model.make_params(gamma = 0.02, sigma = 5,
                                center = 15, amplitude = max(pulse[0])*1000,
                                intercept = pulse[0][0], slope = 1e-12)
                out = model.fit(pulse[0], pars, x=x)
                comps = out.eval_components()
                result.append(simps(comps['expgaussian'],x))

        return result


class compresser():
    """
    Class to compress the .dat files generated by the DLD software.
    """
    def __init__(self, filename):
        self.filename = filename
        self.compress()

        
    def compress(self):
        filelist = glob.glob('*.dat')
        filenum = len(filelist)

        with open(self.filename+'.dat','a') as f:
            print('Compacting files:')
            for index, file in enumerate(filelist):
                with open(file, 'r') as data_file:
                    contents = data_file.read()
                f.write(contents+'\n')
                self.progbar(index, filenum, 20)


    def progbar(self, curr, total, full_progbar):
        frac = curr/total
        filled_progbar = round(frac*full_progbar)
        print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar),
          '[{:>7.2%}]'.format(frac), end='')
        


        
