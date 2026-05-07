@echo off
setlocal

call "%~dp0scripts\\windows\\start_backend.bat"
exit /b %errorlevel%
