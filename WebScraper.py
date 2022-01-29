# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import os
import re
from urllib.parse import urljoin
import concurrent.futures



class FedMinScraper(object):
    '''
    The purpose of this class is to extract monthly US federal reserve minutes
    
    Parameters
    ----------
    dates: list('yyyy'|'yyyy-mm')
        List of strings/integers referencing dates for extraction
        Example:
        dates = [min_year] -> Extracts all transcripts for this year
        dates = [min_year,max_year] -> Extracts transactions for a set of years
        dates = ['2020-01'] -> Extracts transcripts for a single month/year

    nthreads: int
        Set of threads used for multiprocessing
        defaults to None

    Returns
    --------
    transcripts: txt files

    '''
    url_parent = r'https://www.federalreserve.gov/monetarypolicy/'
    url_current = r'fomccalendars.htm'

    #historical transcripts are stored differently
    url_historical = r'fomchistorical{}.htm'
    #each transcript has a unique address, gathered from url_current or url_historical
    url_transcript = r'fomcminutes{}.htm'
    href_regex = re.compile('(?i)/fomc[/]?minutes[/]?\d{8}.htm')

    def __init__(self,dates,nthreads=None,save_path=None):

        #make sure user has given list with strings
        if not isinstance(dates,list):
            raise TypeError("dates should be a list of yyyy or yyyymm str/int")
    
        elif not all([bool(re.search(r'^\d{4}$|^\d{6}$',str(d))) for d in dates]):
            raise ValueError("dates should be in a yyyy or yyyymm format")

        self.dates = dates
        self.nthreads = nthreads
        self.save_path = save_path

        self.ndates = len(dates)
        self.years = [int(d[:4]) for d in dates]
        self.min_year,self.max_year = min(self.years),max(self.years)
        self.transcript_dates = []
        self.transcripts = {}
        self.historical_date = None
        #Each date is treated seperately, where we have yyyymm this will go directly
        #Find minimum year and if month is included, find
        self._get_transcript_dates()

        self.start_threading()

        if save_path:
            self.save_transcript()
        
    def _get_transcript_dates(self):
        '''
        Extract all links for
        '''

        r = requests.get(urljoin(FedMinScraper.url_parent,FedMinScraper.url_current))
        soup = BeautifulSoup(r.text,'lxml')
        #dates are given by yyyymmdd
        
        tdates = soup.findAll("a",href=self.href_regex)
        tdates = [re.search('\d{8}',str(t))[0] for t in tdates]
        self.historical_date = int(min(tdates)[:4])
        #find minimum year

        # extract all of these and filter
        #tdates can only be applied to /fomcminutes
        #historical dates need to be applied to federalreserve.gov

        if self.min_year < self.historical_date:
            #just append the years i'm interested in
            for y in range(self.min_year,self.historical_date + 1):
                
                r = requests.get(urljoin(FedMinScraper.url_parent,FedMinScraper.url_historical.format(y)))
                soup = BeautifulSoup(r.text,parser='lxml')
                hdates = soup.find_all("a",href=self.href_regex)
                tdates.extend([re.search('\d{8}',str(t))[0] for t in hdates])
        
        self.transcript_dates = tdates
  
    def get_transcript(self,transcript_date):

        transcript_url = urljoin(FedMinScraper.url_parent,FedMinScraper.url_transcript.format(transcript_date))
        r = requests.get(transcript_url)

        if not r.ok:
            transcript_url = urljoin(FedMinScraper.url_parent.replace('/monetarypolicy',''),r'fomc/minutes/{}.htm'.format(transcript_date))
            r = requests.get(transcript_url)

        soup = BeautifulSoup(r.content,'lxml')
        main_text = soup.findAll(name='p')

        #clean_main_text = '\n\n'.join(unicodedata.normalize("NFKD",t.get_text(strip=True)) for t in main_text)
        clean_main_text = '\n\n'.join(t.text for t in main_text)
        #clean_main_text = re.sub(r'\r\n',' ',clean_main_text)
        #lean_main_text = re.sub(r'(?<!\n)\n(?!\n)',r'\n\n',clean_main_text) 

        self.transcripts[transcript_date] = clean_main_text



    def start_threading(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.get_transcript,self.transcript_dates)
    
    def save_transcript(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for fname,txt in self.transcripts.items():
            with open(os.path.join(self.save_path,fname+'.txt'),"w",encoding='utf-8') as o:
                o.write(txt)
                o.close()
    
    
if __name__ == '__main__':

    dates = ['2004','2021']
    #Assumes run from ipython
    save_path = os.path.join(os.getcwd(),"Minutes")
    FMS = FedMinScraper(dates=dates,
                        save_path=save_path)
    
