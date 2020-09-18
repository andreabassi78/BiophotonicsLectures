set root="C:\Users\Andrea Bassi\Anaconda3"

call %root%\Scripts\activate.bat %root%

call bokeh serve --show FilterImageServerApplication.py
