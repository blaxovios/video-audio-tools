# Video and audio tools

### Installation
1. Python version 3.12 required.
2. Secondly, create a virtual environment for Python 3 with `python3.12 -m venv venv`and then activate with `source venv/bin/activate`.
3. Then, we need to install dependencies based on [pyproject.toml](pyproject.toml) file. Use `pip install --upgrade --upgrade-strategy eager -e .`.
⚠️ Be aware that if you use the existing requirements file to install dependencies, you may have compatibility issues due to different machines.
4. To avoid pushing files larger than 100 MB, use `find . -size +100M | cat >> .gitignore` and `find . -size +100M | cat >> .git/info/exclude`.

### Project Description
Reads a video file and removes speech from audio.

### Scalene: a Python CPU+GPU+memory profiler with AI-powered optimization proposals
The following command runs Scalene on a provided example program. This helps to define performance issues in the machine learning code.

scalene `remove_speech.py`