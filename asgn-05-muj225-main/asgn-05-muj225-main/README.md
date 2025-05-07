# Midterm - 10-K Sentiment and Stock Return Analysis
Muzammil Jawed

## Purpose
This repository contains an analysis of firm-level data focused on the relationship between sentiment from 10-K filings and stock returns. The analysis explores how positive and negative sentiments extracted from firms' 10-K filings correlate with stock performance around the filing date and subsequent days. The study evaluates the effectiveness of two sentiment dictionaries, LM (Loughran-McDonald) and ML (Machine Learning), in predicting financial market behavior.

The key objectives of this analysis are:
- To analyze the relationship between 10-K sentiment (positive and negative) and firm stock returns around the filing date.
- To explore the impact of contextual sentiment (e.g., financial, geopolitical, and competitive sentiment) on stock returns.
- To compare the predictive power of the LM and ML sentiment dictionaries.
- To visualize the correlation between sentiment measures and stock returns through scatter plots and summary statistics.

## Key Inputs
- **10-K Filings**: HTML files containing firm filings from the SEC EDGAR database. These filings are parsed to extract relevant text for sentiment analysis.
- **Sentiment Dictionaries**: Two sentiment word lists—LM (Loughran-McDonald) and ML (Machine Learning)—are used to compute the sentiment for each firm’s 10-K filing.
- **Stock Return Data**: Daily stock returns are used to measure how sentiment influences firm performance around the filing date (t0) and over subsequent windows (t0 to t+2 and t+3 to t+10).

## Running the Analysis
The analysis is organized in the following order:
1. **Download 10-K Filings**: 
   - Run the `get_text_files_zipAFTER.ipynb` notebook first to download the 10-K filings of firms for the year 2022. 
   - **Note**: This step may download several gigabytes of data depending on the number of firms and filings processed.
2. **Sentiment Analysis**:
   - Run the `sentiment_analysis.py` file to process the downloaded 10-K files, clean the text, calculate sentiment scores using the LM and ML dictionaries, and store the results in a CSV file.
3. **Stock Return Analysis**:
   - Use the sentiment data and stock return data (from a separate source) to perform the correlation and regression analysis. The resulting figures and tables will provide insights into the relationships between sentiment and stock returns.
4. **Visualizations**:
   - The `visualizations.py` script will generate scatter plots and correlation matrices for both the LM and ML sentiment variables, and their relationship with stock returns.

**Important**: After downloading the 10-K filings, you will have a folder containing a large number of HTML files. Be sure to ensure you have enough storage space before beginning the download.

## Required Packages
This project requires the following Python packages:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations and calculations.
- `matplotlib`: For creating visualizations.
- `seaborn`: For advanced plotting (e.g., scatter plots, correlation matrices).
- `beautifulsoup4`: For parsing HTML files.
- `requests-html`: For web scraping.
- `sec_edgar_downloader`: For downloading SEC filings.
- `csv`: For reading and writing CSV files.

To install the necessary packages, run the following command:

```bash
pip install pandas numpy matplotlib seaborn beautifulsoup4 requests-html sec-edgar-downloader
