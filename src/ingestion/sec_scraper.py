import os
from pathlib import Path
from typing import List
from sec_edgar_downloader import Downloader

class SECScraper:
    """
    A scraper utility to download SEC filings (10-k, 10-Q) for given tickers.
    Respects SEC rate limits and directory structures.
    """
    def __init__(self, company_name: str, email_address: str, download_dir: str = "data/raw"):
        """
        Initialises the SEC Downloader.

        Args:
            company_name: Used for the user agent string (SEC requirement).
            email_address: Used for the user agent string (SEC requirement).
            download_dir: Target directory for the raw downlaoded files
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # putting it in the SEC required format
        self.downloader = Downloader(company_name, email_address, str(self.download_dir))
    
    def fetch_filings(self, tickers: List[str], form_types: List[str] = ["10-K"], limit: int = 1) -> None:
        """
        Downlaods specific SEC forms for a list of ticker symbols

        Args:
            tickers: List of stock tickers.
            form_types: List of SEC form types.
            limit: Number of recent filings to download per ticker per form type.
        """
        for ticker in tickers:
            for form in form_types:
                print(f"Downloading {limit} {form} filing(s) for {ticker}...")
                try:
                    self.downloader.get(form, ticker, limit=limit)
                    print(f"Successfully downloaded {form} for {ticker}.")
                except Exception as e:
                    print(f"Failed to download {form} for {ticker}. Error: {e}")

if __name__ == "__main__":
    scraper = SECScraper(company_name="Bhargav Nath", email_address="bhargavkumarnathh@gmail.com")
    scraper.fetch_filings(tickers=["NVDA"], form_types=["10-K"], limit=1)
