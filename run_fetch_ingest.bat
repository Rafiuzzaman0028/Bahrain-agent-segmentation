@echo off
cd /d "C:\Users\Rafiuzzaman0028\Desktop\bahrain_stats_agent"
call venv\Scripts\activate.bat
REM Set dry to true for testing; remove / change as needed
::python scripts\fetch_and_ingest.py --run --dry
python scripts\fetch_and_ingest.py --run 
REM To allow writes and ingest, use: python scripts\fetch_and_ingest.py --run
