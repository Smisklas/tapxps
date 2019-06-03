import glob



class compresser():
    """
    Class to compress the .dat files generated by the DLD software.
    """
    def __init__(self, filename):
        self.filename = filename

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

    def progbar(curr, total, full_progbar):
    frac = curr/total
    filled_progbar = round(frac*full_progbar)
    print('\r', '#'*filled_progbar + '-'*(full_progbar-filled_progbar),
          '[{:>7.2%}]'.format(frac), end='')
