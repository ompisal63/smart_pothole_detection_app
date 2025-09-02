1. Clone the repository
Open Command Prompt or PowerShell, then:
git clone https://github.com/ompisal63/smart_pothole_detection_app.git
cd smart_pothole_app

2. Create a virtual environment
Inside the cloned folder:

python -m venv venv

3. Activate the virtual environment
On PowerShell:

.\venv\Scripts\Activate.ps1


4. On Command Prompt (cmd.exe):

venv\Scripts\activate

5. Install dependencies

pip install -r requirements.txt


*If no requirements.txt is present, install manually: pip install streamlit tensorflow keras pillow numpy opencv-python plotly pandas

6. Run the app
streamlit run app.py


This will open the web app in the browser (default: http://localhost:8501).

7. (Optional) Use the .bat file for one-click run

Double-click run_app.bat if you included it in your repo.
If venv is already created + dependencies installed, this will auto-run your app.
