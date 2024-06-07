# Welcome to my project!

## Spec's (stable funtioning application)
- OS: running on Windows 11 Home
- Processor / RAM: Intel Core i7-1185G7, 16 GB RAM
- IDE used: Visual Studio Code (VSC) > developed by Microsoft

## If you are a MAC user (macOS)
Comment out pywin32 within the requirements.txt before installing, as it leads to issues.

## Instructions
1. Before running the code it is neccessary to create a virtual Python Environment (on the keyboard: ctrl+shift+p / choose "Python: Create environment") and install the requirements I added in 'requirements.txt'.
2. Then add the MongoDB credentials Noémie provided you in model.py and api.py (within the string: mongodb+srv://kaeseno1:<password>@cluster0.4pnoho7.mongodb.net/)
3. Run the app by running 'flask run' in the terminal. ATTENTION: this may take a while as the model is being loaded. Please be patient.
4.  I wish you good luck at the final exam and have fun exploring my project!

Start app: flask run
App starts on either your localhost http://localhost:5000/ or on 127.0.0.1 http://127.0.0.1:5000, depending on what you use.

## Access to MongoDB & API Key
Mongo DB access: please get in touch with kaeseno1@students.zhaw.ch to get the password (required in model.py & api.py). Please do not share this secret with anyone else.
API Key: please generate your own API key if you want to load more data into the mongo DB. But this is not neccessarily required for running the code.

## Example data for input
- Earth is safe: 
Absolute Magnitude: 23.62
Minimum Diameter (km): 0.0501828101
Maximum Diameter (km): 0.1122121746

- Potentially hazardous asteorid:
Absolute Magnitude: 20.68
Minimum Diameter (km): 0.1943367684
Maximum Diameter (km): 0.4345502246

## Questions?
https://letmegoogleforyou.com/ ... no, just joking, I hope that I added enough comments so the code is understandable and working for you. Otherwise, I might help you; just write a mail to kaeseno1@students.zhaw.ch



----
## Supporting command for Noémie
Update requirements: pip freeze > requirements.txt