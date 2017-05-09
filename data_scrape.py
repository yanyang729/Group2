import json
import requests
import datetime
import HTMLParser
import pandas as pd
import re
import time

import dateutil.parser
import datetime


# parse body
def clean_body(s):
    """Return a string
    Remove code, link, img, and tag from the string
    :param s String needs to remove control charaters
    :return cleaned string
    """
    code_match = re.compile('<pre>(.*?)</pre>', re.MULTILINE | re.DOTALL)
    link_match = re.compile('<a href="http://.*?".*?>(.*?)</a>', re.MULTILINE | re.DOTALL)
    img_match = re.compile('<img(.*?)/>', re.MULTILINE | re.DOTALL)
    tag_match = re.compile('<[^>]*>', re.MULTILINE | re.DOTALL)

    # remove code? link? img? tag?
    for match_str in code_match.findall(s):
        s = code_match.sub("", s)

    links = link_match.findall(s)

    s = re.sub(" +", " ", tag_match.sub('', s)).replace("\n", "")

    for link in links:
        if link.lower().startswith("http://"):
            s = s.replace(link, '')

    return s


class DataScraper:
    """Class for data scraper
    """

    #BASE_URL the base of API link
    BASE_URL = "https://api.stackexchange.com/2.2/questions?page={}&pagesize=100&fromdate={}&todate={}&order=desc&sort=creation&site=stackoverflow&filter=withbody"

    def __init__(self,size=10000, from_date=1451606400, to_date=1483228800):
        """
        Init the class DataScraper. Use method scrape to start scrape.
        :type size: int default 10000
        :type from_date: long
        :type to_date: long
        :param size: API requests of scrape data. Default 100000
        :param from_date: The start of creation date of questions. Type is datetime. timestamp. Default 1451606400 means 2016-01-01
        :param to_date: The end of creation date of questions. Type is datetime. timestamp. Default 1483228800 means 2017-01-01
-       """
        self.data_size = size
        self.fromDate = from_date
        self.toDate = to_date

    @staticmethod
    def load_vals(page=1, from_date=1451606400, to_date=1483228800):
        """Return a list(max 100) of items
        Scrape one API link
        :param page: the number of page, start from 1, default 1
        :param from_date: The start of creation date of questions. Type is datetime. timestamp. Default 1451606400 means 2016-01-01
        :param to_date: The end of creation date of questions. Type is datetime. timestamp. Default 1483228800 means 2017-01-01
        :returns: has_more: Argument from API return
                    final_reults: A list of list of items
        """
        url = DataScraper.BASE_URL.format(page, from_date, to_date)
        json_result = json.loads(requests.get(url).content)
        try:
            has_more = json_result["has_more"]
            json_result = HTMLParser.HTMLParser().unescape(json_result["items"])
        except Exception as e1:
            print e1
            return False,None
        if len(json_result) < 1:
            return False, None
        final_reults = []
        for item in json_result:
            title = item["title"]
            time = datetime.datetime.fromtimestamp(item["creation_date"]).strftime('%b %d, %Y')
            body = clean_body(item["body"])
            tags = item["tags"]
            final_reults.append([title, body, time, tags])
        return has_more, final_reults

    @staticmethod
    def load_sets(size=10000, from_date=1451606400, to_date=1483228800):
        """Return a lists of items
        :type size: int default 10000
        :type from_date: long
        :type to_date: long
        :param size: API requests of scrape data. Default 100000
        :param from_date: The start of creation date of questions. Type is datetime. timestamp. Default 1451606400 means 2016-01-01
        :param to_date: The end of creation date of questions. Type is datetime. timestamp. Default 1483228800 means 2017-01-01
        :return: df_set: lists of items. Maxim= size*100 items
        """
        for_size = size / 100
        if size % 100 == 0:
            for_size -= 1
        df_set = []
        for page in range(for_size+1):
            time.sleep(0.1)
            has_more, set100 = DataScraper.load_vals(page + 1, from_date, to_date)
            if not set100:
                break

            df_set += set100
            if not has_more:
                break
        return df_set

    def scrape(self):
        """Retrun dataframe of items with columns: "title", "body","time","tags"
        Start scrape data. Export dataframe with columns: "title", "body","tags"
        :return: A dataframe of items with columns: "title", "body","time","tags"
        """
        dataset=DataScraper.load_sets(self.data_size,self.fromDate,self.toDate)
        df=pd.DataFrame(dataset,columns=["title", "body","time","tags"])
        df_out=df[["title", "body","tags"]]
        df_out.to_csv("./data/processed/scape_data.csv",header=False,index=False,encoding="utf8")
        return df

#Test
if __name__ == '__main__':
    start="2016-01-01"
    end="2017-01-01"
    convertDT=lambda dt:int((dateutil.parser.parse(dt) - datetime.datetime(1970, 1, 1)).total_seconds())
    scraper=DataScraper(size=100000,from_date=convertDT(start),to_date=convertDT(end))
    df_full=scraper.scrape()
    df_fullo_csv("./data/scape_data_time.csv",header=False,index=False,encoding="utf8")

