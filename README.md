# Spreaduler
Computational Experiment Scheduling System (for Machine Learning) using Google Spreadsheets as a dashboard.

## How to use
 1. Using [this](https://gspread.readthedocs.io/en/latest/oauth2.html) tutorial create API credentials.
 2. Create google spreadsheet with columns that corespond to arguments in your argparse parser
 3. Add special columns: 'time_started', 'last_update', 'progress_bar', 'server', 'status', 'comment'
 4. Add collumns with performance characteristics of your script (e.g. 'test accuracy')
 5. Write your worker script by Inheriting from `ParamsSheet` class, add credentials created on step 1 and give id of google spreadsheet that you created in step 2.
 6. Make sure your 'train' function takes parsed arguments as input (output of `parse_args` method)
 7. Add experiment parameters to a spreadsheet and watch worker do it's thing.
 
## Features

 * You can run as many workers in parallel as you like (the only limit is google api, 500 requests/minute)
 * Anything you can put as argparse optional argument can be a parameter in spreadsheet
 * If your program raises an exception status of experiment will be changed to 'error' and traceback will be copied to 'comment' column.
 * You can specify environment variable "SERVERNAME" to override servers hostname.
 * If you specify 'server' in spreadsheet only workers on that server will be able to take the task.
