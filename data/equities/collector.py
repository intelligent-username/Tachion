"""
Stock prices and S&P500 data collector
Calls the TwelveData API to collect information for the past 5 years
(or less if not available) for 100 randomly selected stocks.
It then collects the past 5 years of S&P500 index data.
All data is written into CSVs in the raw/ directory.
"""

import os






if __name__ == "__main__":
    # 5 companies from each of the 11 S&P sectors
    # 9 more "important" large S&P companies for broader market conditions
    # And 15 "smaller" companies for variance injection
    # And of course the S&P data

    # For a total of 80

    with open("companies.txt", "r") as f:
        companies = [line.rstrip("\n") for line in f if line[0] != "#" and line != "\n"]
    
    
    

