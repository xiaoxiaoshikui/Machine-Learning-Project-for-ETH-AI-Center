# Machine Learning Project

This is the repository for the AI Center Projects in Machine Learning Research course in 2023 Spring.

<!-- TODO: check if pre-commit hooks work in VScode in WSL -->

## Setting up

*Note: if you are using Windows, you will need to either use a Linux VM or WSL (see [WSL setup instructions](#setting-up-wsl-recommended-for-windows) below). In both cases you will need to use `linux` as your OS identifier (e.g., for the Anaconda lock file name).*

### Setting up WSL (recommended for Windows)

1. Set up [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
1. Open WSL command line and clone the repo using it (to a path NOT starting with `/mnt/`) instead of using a Windows command prompt of PowerShell. This will make the development environment faster.
1. Add `export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0` to your shell profile (e.g., to the end of `~/.bashrc` or `~/.zshrc` or similar) to enable WSL to open windows and display GUIs.
1. If you are using VSCode, run `code .` inside the project directory to open it (or if you've opened the project before, you can access it from `File -> Open Recent`).
1. Do all further setup inside the WSL command line or from the terminal of VSCode opened from WSL.

### Installing dependencies

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (Anaconda also works, but is not necessary)
1. Navigate to the project directory, then run `conda create --name ml_project --file conda-<linux/osx>-64.lock` (replace `<linux/osx>` with your OS) to create the Anaconda environment.
1. Run `conda activate ml_project` to activate the virtual environment.
1. Run `poetry install` in the project directory to install dependencies of the project.
1. To install MuJoCo, follow the [instructions in the GitHub repo](https://github.com/openai/mujoco-py/#install-mujoco).
1. Add `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin` to your shell profile and start a new shell to make MuJoCo discoverable.
1. Make sure all the required libraries are installed by running `sudo apt install gcc libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf`
1. If you don't have `ffmpeg` installed yet, [install it on your system](https://ffmpeg.org/download.html) or run `pip install imageio-ffmpeg` to install it in the project locally.
1. Run `pre-commit install` to install pre-commit hooks (they will run some checks before each commit to the repo).

### Setting up VSCode (recommended)

1. Install and open [VSCode](https://code.visualstudio.com/download)
1. Open the command palette, choose `Python: Select Interpreter`, then select the virtual environment created by Anaconda.
   *Note: If the desired environment is not in the list, you can find the location of the environments by running `conda env list`, then add the interpreter as a new entry.*
1. Start a new terminal. VSCode will automatically activate the selected environment.
1. Install these VSCode extensions (by searching for them on the extensions tab): `charliermarsh.ruff`, `njpwerner.autodocstring`, `visualstudioexptteam.vscodeintellicode`, `ms-python.black-formatter`, `ms-python.isort`, `ms-python.vscode-pylance`, `ms-python.pylint`, `ms-python.python`, `kevinrose.vsc-python-indent`

*Note: for now VSCode does not use the selected interpreter for Git commands (see [issue](https://github.com/microsoft/vscode-python/issues/10165)), so you need to create commits from changes that contain Python code from the command line (pre-commit hooks need to run from the right Python environment).*

### Without VSCode

1. Run `conda activate ml_project` to activate the Anaconda environment.
1. It's recommended to set up the extensions in your IDE equivalent to those listed above in the VSCode setup section for a more convenient development.

## Managing dependencies

- To **add** a package to dependencies, use `poetry add <package>` for dependencies or `poetry add -D <package>` for development dependencies
- If adding a package using Poetry fails (or if a version from Anaconda is needed), add the package and its version to `environment.yml`, then run:
  - `conda-lock -k explicit --conda mamba` to update the Anaconda lock file
  - `mamba update --file conda-<linux/osx>-64.lock` (replace `<linux/osx>` with your OS) to update the packages in your current environment
- To **remove** a package from the dependencies, use `poetry remove <package>` or remove them from `environment.yml`. In the latter case, you will also need to run the commands above to update the Anaconda lock files and current environment packages.

## Running the code

You can run scripts specified in `pyproject.toml` with `poetry run <script name>`. For example, to run the `gridworld` script, run `poetry run gridworld` (you might also need to run `poetry install` before to update the dependencies).

To run the training of RL agents, run `poetry run train`. To generate videos of the trained agents, run `poetry run generate`.

To launch the web interface, run:

```bash
poetry run flask --app ml_project/web_interface run
```
