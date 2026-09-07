# openpilot tools

## System Requirements

openpilot is developed and tested on **Ubuntu 24.04**, which is the primary development target aside from the [supported embedded hardware](https://github.com/commaai/openpilot#running-on-a-dedicated-device-in-a-car).

Most of openpilot should work natively on macOS and, for development only, on Windows. On Windows you can also use WSL for a nearly native Ubuntu experience. Running natively on any other system is not currently recommended and will likely require modifications.

## Native setup on Ubuntu 24.04 and macOS

Follow these instructions for a fully managed setup experience. If you'd like to manage the dependencies yourself, just read the setup scripts in this directory.

**1. Clone openpilot**
``` bash
git clone https://github.com/commaai/openpilot.git
```

**2. Run the setup script**
``` bash
cd openpilot
tools/op.sh setup
```

**3. Activate a Python shell**
Activate a shell with the Python dependencies installed:
``` bash
source .venv/bin/activate
```

**4. Build openpilot**
``` bash
scons -u
```

## Native setup on Windows

The development tools (UI, cabana, replay, jotpluggler, the models and the unit tests) build natively on Windows; there is no on-road support and the comma device processes that need Linux do not run, exactly as on macOS. The build runs in an [MSYS2](https://www.msys2.org/) CLANG64 shell (clang, lld and libc++) with a native Python managed by uv, so it behaves like the macOS setup rather than WSL.

**1. Install [MSYS2](https://www.msys2.org/), then install git and clone openpilot from its CLANG64 shell.** This is MSYS2's native git; its Cygwin-style `git` package is the one git-lfs cannot drive. A fresh Windows also needs the [Visual C++ Redistributable](https://aka.ms/vc14/vc_redist.x64.exe), which pycapnp's wheel links against; most machines already have it.
``` bash
pacman -S mingw-w64-clang-x86_64-git
git clone https://github.com/commaai/openpilot.git
```

**2. Run the setup script** from that shell. It installs the toolchain with pacman, uv, the Python dependencies (comma's dependencies come as prebuilt Windows wheels) and the LFS files.
``` bash
cd openpilot
tools/op.sh setup
```

**3. Activate a Python shell**
``` bash
source .venv/Scripts/activate
```

**4. Build openpilot**
``` bash
scons -u
```

The tools run from the same shell, e.g. `openpilot/tools/cabana/cabana --demo` or `python openpilot/selfdrive/ui/ui.py`, and `tools/op.sh test` runs the unit tests.

## WSL on Windows

[Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/about) should provide a similar experience to native Ubuntu. [WSL 2](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) specifically has been reported by several users to be a seamless experience.

Follow [these instructions](https://docs.microsoft.com/en-us/windows/wsl/install) to setup the WSL and install the `Ubuntu-24.04` distribution. Once your Ubuntu WSL environment is setup, follow the Linux setup instructions to finish setting up your environment. See [these instructions](https://learn.microsoft.com/en-us/windows/wsl/tutorials/gui-apps) for running GUI apps.

**NOTE**: If you are running WSL 2 and experiencing performance issues with the UI or simulator, you may need to explicitly enable hardware acceleration by setting `GALLIUM_DRIVER=d3d12` before commands. Add `export GALLIUM_DRIVER=d3d12` to your `~/.bashrc` file to make it automatic for future sessions.

## CTF
Learn about the openpilot ecosystem and tools by playing our [CTF](/tools/CTF.md).

## Directory Structure

```
├── car_porting/        # Tools for porting new cars
├── release/            # Scripts for building openpilot releases
└── scripts/            # Miscellaneous scripts
```

Development tools such as cabana, plotjuggler, and replay live in [openpilot/tools/](/openpilot/tools/):

```
├── cabana/             # View and plot CAN messages from drives or in realtime
├── camerastream/       # Cameras stream over the network
├── joystick/           # Control your car with a joystick
├── lib/                # Libraries to support the tools and reading openpilot logs
├── plotjuggler/        # A tool to plot openpilot logs
├── replay/             # Replay drives and mock openpilot services
└── sim/                # Run openpilot in a simulator
```
